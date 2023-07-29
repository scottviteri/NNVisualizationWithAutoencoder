"""
This contains code to train the various components.

TODO: Move this outside of src/ and instead import this as a python package with pip install.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import openai
from tqdm.auto import tqdm
from accelerate import Accelerator
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import TrainingConfig
from autoencoder import TAE, LinearAutoEncoder, Gpt2AutoencoderBoth

from utils import (
    unembed_and_decode,
    generate_sentence,
    generate_sentence_batched,
    update_plot,
    print_results,
)

class DeepDreamLLMTrainer:
    def __init__(self, config: TrainingConfig):
        """
        A class for training an autoencoder and for optimizing a sentence in the latent
        space of that autencoder in order to activate a neuron.

        Args:
            config (TrainingConfig): a configuration object containing various training parameters.

        """
        self.__dict__.update(vars(config))

        if self.use_openai:
            print("Please input your OpenAI API key in the terminal below:")
            openai.api_key = input()
        
        # Load the model's parameters from a checkpoint if provided
        if self.autoencoder is None:
            # Default autoencoder
            full_name = None
            model_names = ['LinearAutoEncoder', 'Gpt2AutoencoderBoth', 'TAE']
            if self.autoencoder_name not in model_names:
                full_name = self.autoencoder_name
                self.autoencoder_name = self.autoencoder_name.split('_')[0]
            if self.autoencoder_name == "LinearAutoEncoder":
                self.autoencoder = LinearAutoEncoder("distilgpt2", latent_dim=self.latent_dim)
            elif self.autoencoder_name == "Gpt2AutoencoderBoth":
                self.autoencoder = Gpt2AutoencoderBoth("distilgpt2", latent_dim=self.latent_dim)
            elif self.autoencoder_name == "TAE":
                self.autoencoder = TAE("distilgpt2", latent_dim=self.latent_dim)
            else:
                raise NotImplementedError(f"Autoencoder {config.autoencoder_name} not implemented")
            if full_name:
                loaded = torch.load("/content/NNVisualizationWithAutoencoder/Checkpoints/"+full_name)
                self.autoencoder.load_state_dict(loaded)

        accelerator = Accelerator()  # TODO actually use this other than just preparing stuff
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr=self.learning_rate)
        self.model, self.optimizer, self.autoencoder = accelerator.prepare(self.model, self.optimizer, self.autoencoder)
        self.device = accelerator.device
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        #print("Testing autoencoder shapes")
        #self.test_autoencoder_shapes()
        #print("Autoencoder shapes test passed")

    def get_embeddings(self, input_ids):
        return self.model.transformer.wte(input_ids)

    def encode_sentence(self, sentence):
        assert isinstance(sentence, str), "sentence must be a string"
        return self.tokenizer.encode(
            sentence,
            return_tensors="pt",
        ).to(self.device)

    # 3 kinds of loss: loss, openai_distance, and reencode_loss
    def calc_loss(self, original, reconstructed):
        return torch.mean(torch.norm(original - reconstructed, dim=2))

    def calc_openai_loss(self, sentence1, sentence2):
        # this loss is distinguished by using openai embeddings to measure similarity
        # this one should be the most dissimilar in nature
        response1 = openai.Embedding.create(
            input=sentence1, model="text-embedding-ada-002"
        )
        response2 = openai.Embedding.create(
            input=sentence2, model="text-embedding-ada-002"
        )
        embedding1 = np.array(response1["data"][0]["embedding"])
        embedding2 = np.array(response2["data"][0]["embedding"])
        distance = 1 - cosine_similarity(
            embedding1.reshape(1, -1), embedding2.reshape(1, -1)
        )
        return distance.item()

    def calc_reencode_loss(self, sentence1, sentence2):
        # this loss is distinguished from original loss by decoding, re-encoding and taking embeddings distance
        input_ids_1, input_ids_2 = self.encode_sentence(
            sentence1
        ), self.encode_sentence(sentence2)
        embeddings_1, embeddings_2 = self.get_embeddings(
            input_ids_1
        ), self.get_embeddings(input_ids_2)
        # truncate the longer embedding to be the size of the shorter one
        if embeddings_1.shape[1] > embeddings_2.shape[1]:
            embeddings_1 = embeddings_1[:, : embeddings_2.shape[1], :]
        elif embeddings_2.shape[1] > embeddings_1.shape[1]:
            embeddings_2 = embeddings_2[:, : embeddings_1.shape[1], :]
        return self.calc_loss(embeddings_1, embeddings_2).item()

    def train_autoencoder(self, num_epochs, print_every, save_path=None, num_sentences=None):
        if save_path is None:
            save_path = f"/content/NNVisualizationWithAutoencoder/Checkpoints/{self.autoencoder_name}_{num_epochs}_{print_every}.pt"
        losses, openai_losses, reencode_losses = [], [], []

        # Initialize numpy array for sentences
        sentences = np.array([])

        pbar = tqdm(range(num_epochs))
        # Training loop
        for epoch in pbar:
            # If there aren't enough sentences left, generate new ones
            if sentences.size < self.batch_size:
                print("Ran out of sentences, generating another batch")
                nsent = num_sentences if num_sentences else 1000
                new_sentences = generate_sentence_batched(
                    self.model, self.tokenizer, n=nsent
                )
                sentences = np.append(sentences, new_sentences)

            # Extract a batch of sentences and remove them from the array
            input_sentences = sentences[:self.batch_size]
            sentences = sentences[self.batch_size:]
            # input_ids, attention_mask = self.tokenizer.encode(
            #   input_sentences, padding=True, truncation=True, return_tensors="pt").to(
            #     self.device
            # )
            encoding = self.tokenizer(input_sentences.tolist(), padding=True, truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            original_embeddings = self.get_embeddings(input_ids)
            reconstructed_embeddings = self.autoencoder(original_embeddings, attention_mask.T==0)
            loss = self.calc_loss(original_embeddings, reconstructed_embeddings).sum()
            
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
            if epoch % print_every == 0:
                print("loss: ", loss)

            # if epoch % print_every == 0:
            #   # Record the loss value for plotting
            #     reconstructed_sentence = unembed_and_decode(
            #         model=self.model,
            #         tokenizer=self.tokenizer,
            #         embeds_input=reconstructed_embeddings,
            #     )
            #     reconstructed_sentences.append(reconstructed_sentence)

            #     reencode_loss = self.calc_reencode_loss(
            #         input_sentence, reconstructed_sentence
            #     )
            #     reencode_losses.append(reencode_loss)
            #     openai_loss = 0
            #     if self.use_openai:
            #         openai_loss = self.calc_openai_loss(
            #             input_sentence, reconstructed_sentence
            #         )
            #         openai_losses.append(openai_loss)
            #     if self.is_notebook:
            #         from IPython.display import clear_output

            #         clear_output(wait=True)
            #         update_plot(losses, openai_losses, reencode_losses, print_every)
            #     print_results(
            #         epoch,
            #         input_sentence,
            #         reconstructed_sentence,
            #         loss.item(),
            #         openai_loss,
            #         reencode_loss,
            #         num_epochs
            #     )
            #     if save_path: torch.save(self.autoencoder.state_dict(), save_path)
        return losses, openai_losses, reencode_losses, sentences, reconstructed_sentences

    def neuron_loss_fn(activation):
        """
        Default loss function
        """
        return -torch.sigmoid(activation)

    def optimize_for_neuron_whole_input(
        self,
        neuron_index=0,
        layer_num=1,
        mlp_or_attention="mlp",
        num_tokens=50,
        num_iterations=200,
        loss_fn=neuron_loss_fn,
        learning_rate=0.1,
        seed=42,
        verbose=True
    ):
        """
        Args:
            neuron_index (int): the index of the neuron to optimize for
            layer_num (int): the layer number to optimize for
            mlp_or_attention (str): 'mlp' or 'attention'
            num_tokens (int): the number of tokens to in the sentence that we optimize over
            num_iterations (int): the number of iterations to run the optimization for
            loss_fn (function): the loss function to use.
            learning_rate (float): the learning rate to use for the optimizer
            seed (int): the seed to use for reproducibility
        Returns:
            losses (list): the list of losses of the model
            log_dict (dict): has keys below
                original_sentence (str): the original generated sentence
                original_sentence_reconstructed (str) : the original sentence after reconstructing it
                reconstructed_sentences (list): Reconstructed sentences during training every 1/30th of the way through
                activations (list): Average activations of the neuron every 1/30th of the way through
        """
        log_dict = {}

        # Set the seed for reproducibility
        sentence = generate_sentence(
            self.model, self.tokenizer, max_length=num_tokens, seed=seed
        )
        if verbose: tqdm.write("Original sentence is:")
        if verbose: tqdm.write(sentence)
        log_dict["original_sentence"] = sentence

        input_ids = self.encode_sentence(sentence)
        original_embeddings = self.get_embeddings(input_ids)
        latent = self.autoencoder.encode(original_embeddings)
        latent_vectors = latent.detach().clone().to(self.device)
        latent_vectors.requires_grad = True

        if verbose: tqdm.write("original reconstructed sentence is ")
        with torch.no_grad():
            og_reconstructed_sentence = unembed_and_decode(
                self.model, self.tokenizer, self.autoencoder.decode(latent_vectors)
            )
            log_dict["original_sentence_reconstructed"] = og_reconstructed_sentence
        # Create an optimizer for the latent vectors
        optimizer = AdamW(
            [latent_vectors], lr=learning_rate
        )  # You may need to adjust the learning rate

        if "mlp" in mlp_or_attention:
            layer = self.model.transformer.h[layer_num].mlp.c_fc
        elif "attention" in mlp_or_attention:
            layer = self.model.transformer.h[layer_num].attn.c_attn
        else:
            raise NotImplementedError("Haven't implemented attention block yet")

        activation_saved = [torch.tensor(0.0, device=self.device)]

        def hook(model, input, output):
            # The output is a tensor. We're getting the average activation of the neuron across all tokens.
            activation = output[0, :, neuron_index].mean()
            activation_saved[0] = activation

        handle = layer.register_forward_hook(hook)

        losses, log_dict["reconstructed_sentences"], log_dict["activations"] = [], [], []
        if verbose:
            pbar = tqdm(range(num_iterations), position=0, leave=True)
        else:
            pbar = range(num_iterations)
        for i in pbar:
            # Construct input for the self.model using the embeddings directly
            embeddings = self.autoencoder.decode(latent_vectors)
            _ = self.model(
                inputs_embeds=embeddings
            )  # the hook means outputs are saved to activation_saved
            # We want to maximize activation, which is equivalent to minimizing negative activation
            loss = loss_fn(activation_saved[0])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % (num_iterations // 30) == 0:
                if verbose: tqdm.write(f"Loss at step {i}: {loss.item()}\n", end="")
                reconstructed_sentence = unembed_and_decode(
                    self.model, self.tokenizer, embeddings
                )
                if verbose: tqdm.write(reconstructed_sentence, end="")
                log_dict["reconstructed_sentences"].append(reconstructed_sentence)
                log_dict["activations"].append(activation_saved[0].item())
            optimizer.zero_grad()

        handle.remove()  # Don't forget to remove the hook!
        return losses, log_dict

    def test_autoencoder_shapes(self):
        """
        1. Checks that the latent vector shape is the same as latent_dim
        2. Checks that the decoded shape is the same as the starting shape

        3. Unembed and decode has different number of tokens potentially :(
        """

        # 1
        sentence = generate_sentence(
            self.model, self.tokenizer, max_length=10
        )
        input_ids = self.encode_sentence(sentence)
        original_embeddings = self.get_embeddings(input_ids)
        latent = self.autoencoder.encode(original_embeddings)
        assert latent.shape[2] == self.autoencoder.latent_dim, (
            f"latent dim {latent.shape[2]} does not match autoencoder latent dim {self.autoencoder.latent_dim}"
        )

        # 2
        reconstructed_embeddings = self.autoencoder.decode(latent)
        assert reconstructed_embeddings.shape == original_embeddings.shape, (
            f"reconstructed_embeddings shape {reconstructed_embeddings.shape} does not match original_embeddings shape {original_embeddings.shape}"
        )

        
        return True
