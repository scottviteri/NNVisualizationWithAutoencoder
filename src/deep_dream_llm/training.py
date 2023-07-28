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
import random
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.metrics.pairwise import cosine_similarity

from deep_dream_llm.utils import (
    unembed_and_decode,
    get_sentence_similarity,
    generate_sentence,
    generate_sentence_batched,
    update_plot,
    print_results,
)

class DeepDreamLLMTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        autoencoder=None,
        load_path=None,
        optimizer=None,
        use_openai=True,
        print_every=150,
        lr_scheduler=None,
        is_notebook=False,
        num_sentences=100
    ):
        """
        A class for training an autoencoder and for optimizing a sentence in the latent
        space of that autencoder in order to activate a neuron.

        """
        if use_openai:
            print("Please input your OpenAI API key in the terminal below:")
            openai.api_key = input()
        self.use_openai = use_openai

        # Load the model's parameters from a checkpoint if provided
        self.autoencoder = autoencoder
        if load_path is not None:
            self.autoencoder.load_state_dict(torch.load(load_path))

        accelerator = (
            Accelerator()
        )  # TODO actually use this other than just preparing stuff
        self.model, self.optimizer, self.autoencoder = accelerator.prepare(
            model, optimizer, self.autoencoder
        )
        self.device = accelerator.device
        print("accelerator device: ", self.device)
        self.lr_scheduler = lr_scheduler

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token

        self.sentences = generate_sentence_batched(
            self.model, self.tokenizer, n=self.num_sentences
        )
        self.print_every = print_every
        self.is_notebook = is_notebook
        self.num_sentences = num_sentences

    def get_embeddings(self, input_ids):
        return self.model.transformer.wte(input_ids)

    def encode_sentence(self, sentence):
        return self.tokenizer.encode(
            sentence,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
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
        return self.calc_loss(embeddings_1, embeddings_2).item()

    def train_autoencoder(
        self,
        save_path,
        num_epochs=1000,
        print_every=100,
        use_openai=None,
    ):
        losses, openai_losses, reencode_losses = [], [], []
        reconstructed_sentences = []
        if use_openai is None:
            use_openai = self.use_openai

        pbar = tqdm(range(num_epochs))
        # Training loop
        for epoch in pbar:
            if len(self.sentences) == 0:
                print("Ran out of sentences, generating another batch")
                self.sentences = generate_sentence_batched(
                    self.model, self.tokenizer, n=self.num_sentences
                )
            input_sentence = np.random.choice(self.sentences, replace=False)

            input_ids = self.tokenizer.encode(input_sentence, return_tensors="pt").to(
                self.device
            )
            original_embeddings = self.get_embeddings(input_ids)
            reconstructed_embeddings = self.autoencoder(original_embeddings)
            loss = self.calc_loss(original_embeddings, reconstructed_embeddings)

            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())

            if epoch % print_every == 0:
              # Record the loss value for plotting
                reconstructed_sentence = unembed_and_decode(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    embeds_input=reconstructed_embeddings,
                )[0]
                reconstructed_sentences.append(reconstructed_sentence)

                reencode_loss = self.calc_reencode_loss(
                    input_sentence, reconstructed_sentence
                )
                reencode_losses.append(reencode_loss)
                openai_loss = 0
                if use_openai:
                    openai_loss = self.calc_openai_loss(
                        input_sentence, reconstructed_sentence
                    )
                    openai_losses.append(openai_loss)
                if self.is_notebook:
                    from IPython.display import clear_output

                    clear_output(wait=True)
                    update_plot(losses, openai_losses, reencode_losses)
                print_results(
                    epoch,
                    input_sentence,
                    reconstructed_sentence,
                    loss.item(),
                    openai_loss,
                    reencode_loss,
                )
                if save_path:
                    torch.save(self.autoencoder.state_dict(), save_path)
        return losses, openai_losses, reencode_losses, reconstructed_sentences

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
            a tuple of losses, log_dict
            losses - the list of losses of the modell
            log_dict:
                'original_sentence' - the original generated sentence
                'original_sentence_reconstructed' - the original sentence after reconstructing it
                'reconstructed_sentences' - a list of the reconstructed sentence as it chanegs throughout training
        """
        log_dict = {}

        # Set the seed for reproducibility
        sentence = generate_sentence(
            self.model, self.tokenizer, max_length=num_tokens, seed=seed
        )
        print("Original sentence is:")
        print(sentence)
        log_dict["original_sentence"] = sentence

        input_ids = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        original_embeddings = self.model.transformer.wte(input_ids)
        latent = self.autoencoder.encoder(original_embeddings)
        latent_vectors = latent.detach().clone().to(self.device)
        latent_vectors.requires_grad = True

        print("original reconstructed sentence is ")
        with torch.no_grad():
            og_reconstructed_sentence = unembed_and_decode(
                self.model, self.tokenizer, self.autoencoder.decoder(latent_vectors)
            )[0]
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

        losses, log_dict["reconstructed_sentences"] = [], []
        for i in tqdm(range(num_iterations), position=0, leave=True):
            # Construct input for the self.model using the embeddings directly
            embeddings = self.autoencoder.decoder(latent_vectors)
            _ = self.model(
                inputs_embeds=embeddings
            )  # the hook means outputs are saved to activation_saved
            # We want to maximize activation, which is equivalent to minimizing negative activation
            loss = loss_fn(activation_saved[0])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % (num_iterations // 30) == 0:
                tqdm.write(f"Loss at step {i}: {loss.item()}\n", end="")
                reconstructed_sentence = unembed_and_decode(
                    self.model, self.tokenizer, embeddings
                )[0]
                tqdm.write(reconstructed_sentence, end="")
                log_dict["reconstructed_sentences"].append(reconstructed_sentence)
            optimizer.zero_grad()

        handle.remove()  # Don't forget to remove the hook!
        return losses, log_dict

    def update_autoencoder_plot(self, losses, openai_losses, reencode_losses):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss", color="b")
        ax1.plot(losses, color="b")

        ax1.tick_params("y", colors="r")
        ax1.set_ylabel("Reencode Loss", color="r", labelpad=15)
        ax1.plot([self.print_every*i for i in range(len(reencode_losses))],
                reencode_losses, color="r")
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        ax2.set_ylabel("OpenAI Loss", color="g")
        ax2.plot([self.print_every*i for i in range(len(openai_losses))],
                  openai_losses, color="g")
        ax2.set_ylim(-1, 1)

        fig.tight_layout()
        plt.show()
