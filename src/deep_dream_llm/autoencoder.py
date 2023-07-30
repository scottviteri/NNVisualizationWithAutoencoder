"""
This is the file for defining the auto-encoder classes to experiment with.

Basic interface for each class:
    Should implement an encoder and a decoder. Encoder takes it to the latent space.
    Decoder takes it out of the latent space back the the embedding space.
"""


import torch
from torch.nn import (
    MSELoss,
    Linear,
    TransformerEncoderLayer,
    LayerNorm,
    TransformerEncoder,
)
import copy
import random
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class mock_transformer(torch.nn.Module):
    def __init__(self):
        """
        Mock transformer model for testing.
        Implements layers:
            wte
        """
        super().__init__()
        # vocab size to 768
        self.wte = Linear(50257, 768)

    def forward(self, input_ids, attention_mask=None):
        return self.wte(input_ids)


class mock_gpt2(torch.nn.Module):
    def __init__(self):
        """
        Mock gpt2 model for testing.
        Implements layers:
            lm_head,
            transformer.wte
        Implements methods:
            generate,
            normal forward,
            inputs_embeds forward,
        """
        super().__init__()
        self.lm_head = Linear(768, 768)
        self.transformer = mock_transformer()

    def forward(self, inputs_ids=None, input_embeds=None, attention_mask=None):
        """
        Accepts either input_ids or input_embeds. Returns logits.
        Args:
            input_ids: torch.tensor of shape (batch_size, sequence_length, vocab_size)
            input_embeds: torch.tensor of shape (batch_size, sequence_length, embedding_size)
        Returns:
            torch.tensor of shape (batch_size, sequence_length, embedding_size)
        """
        if input_embeds is None:
            input_embeds = self.transformer(inputs_ids)
        return self.lm_head(input_embeds)

class MockAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim=100):
        """
        Mock autoencoder model for testing.
        """
        super().__init__()
        self.encoder = Linear(768, latent_dim)
        self.decoder = Linear(latent_dim, 768)
        self.latent_dim = latent_dim

    def encode(self, input_embeds, **kwargs):
        return self.encoder(input_embeds, **kwargs)

    def decode(self, latent, **kwargs):
        return self.decoder(latent, **kwargs)

    def forward(self, inputs_embeds):
        """
        Args:
            input_embeds: torch.tensor of shape (batch_size, sequence_length, embedding_size)
        Returns:
            torch.tensor of shape (batch_size, sequence_length, embedding_size)
        """
        latent = self.encode(inputs_embeds)
        return self.decode(latent)


class TAE(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim, nhead=8, num_layers=8):
        """
        TODO: Test this
        """
        super().__init__()

        # Load the pretrained model
        base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

        # Create the encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_model.config.n_embd, nhead=nhead
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Create the decoder from scratch
        self.projection_1 = Linear(base_model.config.n_embd, latent_dim)
        self.projection_2 = Linear(latent_dim, base_model.config.n_embd)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=base_model.config.n_embd, nhead=nhead
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers
        )  # logits?

        self.latent_dim = latent_dim

    def encode(self, input_embeds, attention_mask):
        encoded_embeddings = self.encoder(input_embeds, src_key_padding_mask=attention_mask)
        latent = self.projection_1(encoded_embeddings)
        return latent

    def decode(self, latent, attention_mask):
        p2 = self.projection_2(latent)
        return self.decoder(p2, src_key_padding_mask=attention_mask)

    def forward(self, input_embeds, attention_mask=None):
        # Encode the input
        latent = self.encode(input_embeds, attention_mask)
        reconstructed_embeddings = self.decode(latent, attention_mask)
        return reconstructed_embeddings


class Gpt2Autoencoder(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=100, nhead=2, num_layers=6):
        super().__init__()

        # Load the pretrained model
        base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

        # Create the encoder
        self.encoder = copy.deepcopy(base_model)
        self.encoder.lm_head = Linear(base_model.config.n_embd, latent_dim)

        # Create the decoder from scratch
        self.projection = Linear(latent_dim, base_model.config.n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_model.config.n_embd, nhead=nhead
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )  # logits?

    def encode(self, input_embeds):
        return self.encoder(input_embeds).logits



    def forward(self, input_embeds, attention_mask=None):
        # Encode the input
        latent = self.encoder(
            inputs_embeds=input_embeds, attention_mask=attention_mask
        ).logits

        # Project the latent representation to the original embedding dimension
        projected = self.projection(latent)

        # Decode the projected representation
        reconstructed_embeddings = self.decoder(projected)

        return reconstructed_embeddings


class Gpt2Encoder(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=100):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        self.base_model.lm_head = Linear(self.base_model.config.n_embd, latent_dim)

    def forward(self, inputs_embeds):
        """
        Inputs_embeds - shape [batch_size, sequence_length, embedding_size]
        """
        latent = self.base_model(inputs_embeds=inputs_embeds).logits
        return latent


class Gpt2DecoderToEmbedding(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=100):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        self.base_model.lm_head = Linear(
            self.base_model.config.n_embd, self.base_model.config.n_embd
        )
        self.projection = Linear(latent_dim, self.base_model.config.n_embd)

    def forward(self, latent):
        """
        latent - shape [batch_size, sequence_length, embedding_size]
        output - shape [batch_size, sequence_length, embedding_size]
        """
        projected = self.projection(latent)
        decoded = self.base_model(inputs_embeds=projected).logits
        return decoded


class Gpt2DecoderToVocab(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=100):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        self.projection = Linear(latent_dim, self.base_model.config.n_embd)

    def forward(self, latent):
        """
        latent - shape [batch_size, sequence_length, embedding_size]
        output - shape [batch_size, sequence_length, vocab_size]
        """
        projected = self.projection(latent)
        decoded = self.base_model(inputs_embeds=projected).logits
        return decoded


class Gpt2AutoencoderBoth(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=100):
        """
        Implements GPt2Autoencoder with both encoder and decoder
        """
        super().__init__()
        self.encoder = Gpt2Encoder(model_checkpoint, latent_dim=latent_dim)
        self.decoder = Gpt2DecoderToEmbedding(model_checkpoint, latent_dim=latent_dim)

    def forward(self, input_embeds):
        latent = self.encoder(input_embeds)
        decoded = self.decoder(latent)
        return decoded


class LinearAutoEncoder(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=100):
        super().__init__()

        # Load the pretrained model
        base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        self.latent_dim = latent_dim

        # Create the encoder
        self.encoder = Linear(base_model.config.n_embd, latent_dim)
        self.decoder = Linear(latent_dim, base_model.config.n_embd)

    def encode(self, input_embeds, **kwargs):
        return self.encoder(input_embeds)

    def decode(self, latent, **kwargs):
        return self.decoder(latent)

    def forward(self, inputs_embeds, attention_mask=None):
        # Encode the input
        latent = self.encode(inputs_embeds)
        # Decode the projected representation
        reconstructed_embeddings = self.decode(latent)
        return reconstructed_embeddings
