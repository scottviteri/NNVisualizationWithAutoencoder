"""
This contains code to train the various components.

TODO: Move this outside of src/ and instead import this as a python package with pip install.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import MSELoss, Linear, TransformerEncoderLayer, LayerNorm, TransformerEncoder
from torch.optim import Adam
from torch.cuda.amp import autocast
import copy
#from tqdm import tqdm
import openai
import numpy as np
import random
from torch import nn
import random
from IPython.display import clear_output
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import weightwatcher as ww
import code

from .autoencoder import LinearAutoEncoder, Gpt2Autoencoder, Gpt2AutoencoderBoth
from .utils import unembed_and_decode, get_sentence_similarity, gen_sentences, generate_sentence


class DeepDreamLLMTrainer:
    def __init__(self, hugging_face_model_name="distilgpt2", randomize_sentences=True, autoencoder_cls=LinearAutoEncoder):
        """
        A class for training an autoencoder and for optimizing a sentence in the latent
        space of that autencoder in order to activate a neuron.

        Args:
            randomize_sentences: bool - whether or not to train with randomized sentences
        """
        openai.api_key = input()
        # Check if CUDA is available and choose device accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(hugging_face_model_name).to(device)
        if randomize_sentences:
            self.sentences = gen_sentences(self.model, self.tokenizer, device=device)
        self.randomize_sentences = randomize_sentences

        autoencoder = autoencoder_cls('distilgpt2').to(self.device)
        try:
            autoencoder.load_state_dict(torch.load('transformer-autoencoder.pt'))
        except:
            print("No autencoder file found")
        self.autoencoder = autoencoder


    def calc_loss(self, sentence1, sentence2):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #padding='max_length', max_length=model.config.n_ctx
        input_ids_1 = self.tokenizer.encode(sentence1, return_tensors="pt").to(self.device)
        embeddings_1 = self.model.transformer.wte(input_ids_1)
        input_ids_2 = self.tokenizer.encode(sentence2, return_tensors="pt").to(self.device)
        embeddings_2 = self.model.transformer.wte(input_ids_2)
        #code.interact(local=locals())
        print("embeddings_shape:", embeddings_1.shape, embeddings_2.shape)

        if embeddings_1.shape != embeddings_2.shape:
        print("warning, shapes are not equal in loss calc")
        return torch.mean(torch.norm(embeddings_1 - embeddings_2, dim=2)).item()


    def train_autoencoder(self, optimizer, num_epochs=1000, print_every=1, save_path="transformer-autoencoder.pt", load_path=None, use_openai=True):
        loss_values = []
        similarity_values = []
        constructed_losses = []

        # Load the model's parameters from a checkpoint if provided
        if load_path is not None:
            self.autoencoder.load_state_dict(torch.load(load_path))

        pbar = tqdm(range(num_epochs))
        # Training loop
        for epoch in pbar:
            # Generate a sentence with the pretrained model
            if self.random_sentences:
                input_sentence = generate_sentence(self.model, self.tokenizer, max_length=50)
            else:
                input_sentence = self.sentences[0]
            # Prepare the inputs for the self.autoencoder
            # , max_length=self.model.config.n_ctx
            input_ids = self.tokenizer.encode(input_sentence, return_tensors="pt").to(device)
            original_embeddings = self.model.transformer.wte(input_ids)
            #print("orig:", original_embeddings.shape)
            # Run the self.autoencoder and compute the loss
            #with autocast():
            reconstructed_embeddings = self.autoencoder(original_embeddings)
            loss = torch.mean(torch.norm(original_embeddings-reconstructed_embeddings, dim = 2))
            #loss = criterion(reconstructed_embeddings, original_embeddings)
            #code.interact(local=locals())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Record the loss value for plotting
            loss_values.append(loss.item())

            #print("reconstructed embs:", reconstructed_embeddings.shape)
            # Compute the sentence similarity between the original and reconstructed sentences
            reconstructed_sentence = unembed_and_decode(reconstructed_embeddings)[0]
            # reconstructed_sentence has too many tokens?
            #print("recon:", len(reconstructed_sentence))
            # Print progress and save model every 'print_every' epochs
            if epoch % print_every == 0:
                if use_openai:
                    similarity = get_sentence_similarity(input_sentence, reconstructed_sentence)
                    similarity_values.append(similarity)
                    #constructed_loss = calc_loss(input_sentence, reconstructed_sentence)
                    #constructed_losses.append(constructed_loss)
                    pbar.set_description(
                        f"""Original: {input_sentence}
                        Reconstructed: {reconstructed_sentence}
                        Constructed Loss: constructed_loss
                        Epoch {epoch}/{num_epochs}, Loss: {loss.item()}, Similarity: {similarity}""")
                torch.save(self.autoencoder.state_dict(), save_path)

        # Plot the loss values, similarity values, and constructed losses
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(loss_values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 5)
        if use_openai:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:red'
            ax2.set_ylabel('Similarity', color=color)  # we already handled the x-label with ax1
            ax2.plot(range(0, num_epochs, print_every), similarity_values, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

        ## Add constructed losses to the plot
        #ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
        #ax3.spines['right'].set_position(('outward', 60))  # move the y-axis
        #color = 'tab:green'
        #ax3.set_ylabel('Constructed Losses', color=color)
        #ax3.plot(range(0, num_epochs, print_every), constructed_losses, color=color)
        #ax3.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


    def neuron_loss_fn(activation):
        """
        Default loss function
        """
        return -torch.sigmoid(activation)

    def optimize_for_neuron_whole_input(self, neuron_index=0, layer_num=1, mlp_or_attention="mlp", num_tokens=10, num_iterations=200, loss_fn=neuron_loss_fn):
        """
        Args:
            mlp_or_attention (str): 'mlp' or 'attention'
            num_tokens (int): the 
        Returns:
            a tuple of losses, log_dict
            losses - the list of losses of the modell
            log_dict:
                'original_sentence' - the original generated sentence
                'original_sentence_reconstructed' - the original sentence after reconstructing it
                'reconstructed_sentences' - a list of the reconstructed sentence as it chanegs throughout training
        """

        # Get the dimensionality of the latent space
        # latent_dim = self.autoencoder.latent_dim
        # latent_dim = 100

        log_dict = {}

        # TODO Start with a random sentence
        torch.manual_seed(42)
        sentence = generate_sentence(self.model, self.tokenizer, max_length=50)
        print("Original sentence is:")
        print(sentence)
        log_dict["original_sentence"] = sentence

        input_ids = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        original_embeddings = self.model.transformer.wte(input_ids)
        latent = self.autoencoder.encoder(original_embeddings)
        latent_vectors = latent.detach().clone().to(self.device)
        latent_vectors.requires_grad = True
        # latent_vectors = torch.randn((1, num_tokens, latent_dim), device=device, requires_grad=True)
        print("original reconstructed sentence is ")
        with torch.no_grad():
            og_reconstructed_sentence = unembed_and_decode(self.autoencoder.decoder(latent_vectors))[0]
            log_dict["original_sentence_reconstructed"] = og_reconstructed_sentence
        # Create an optimizer for the latent vectors
        optimizer = AdamW([latent_vectors], lr=0.1)  # You may need to adjust the learning rate

        if 'mlp' in mlp_or_attention:
            layer = self.model.transformer.h[layer_num].mlp.c_fc
        elif 'attention' in mlp_or_attention:
            layer = self.model.transformer.h[layer_num].attn.c_attn
        else:
            raise NotImplementedError("Haven't implemented attention block yet")

        activation_saved = [torch.tensor(0.0, device=self.device)]
        def hook(model, input, output):
            # The output is a tensor. We're getting the average activation of the neuron across all tokens.
            activation = output[0, :, neuron_index].mean()
            activation_saved[0] = activation
        handle = layer.register_forward_hook(hook)

        losses, log_dict["reconstructed_sentences"]= [], []
        for i in tqdm(range(num_iterations), position=0, leave=True):
            # Construct input for the self.model using the embeddings directly
            embeddings = self.autoencoder.decoder(latent_vectors)
            _ = self.model(inputs_embeds=embeddings) # the hook means outputs are saved to activation_saved
            # We want to maximize activation, which is equivalent to minimizing negative activation
            loss = loss_fn(activation_saved[0])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % (num_iterations//30) == 0:
                tqdm.write(f"Loss at step {i}: {loss.item()}\n", end='')
                reconstructed_sentence = unembed_and_decode(embeddings)[0]
                tqdm.write(reconstructed_sentence, end='')
                log_dict["reconstructed_sentences"].append(reconstructed_sentence)
            optimizer.zero_grad()

        handle.remove()  # Don't forget to remove the hook!
        return losses, log_dict
    


