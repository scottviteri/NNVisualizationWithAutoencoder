"""
This is the file for defining the auto-encoder classes to experiment with
"""


import torch
from torch.nn import MSELoss, Linear, TransformerEncoderLayer, LayerNorm, TransformerEncoder
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

        encoder_layer = nn.TransformerEncoderLayer(d_model=base_model.config.n_embd, nhead=nhead)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # logits?

    def forward(self, input_embeds, attention_mask=None):
        # Encode the input
        latent = self.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask).logits

        # Project the latent representation to the original embedding dimension
        projected = self.projection(latent)

        # Decode the projected representation
        reconstructed_embeddings = self.decoder(projected)

        return reconstructed_embeddings
    

class Gpt2AutoencoderBoth(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=100, nhead=2, num_layers=6):
        super().__init__()

        # Load the pretrained model
        base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        #self.encoder = nn.Linear(base_model.config.n_embd, latent_dim)
        self.encoder = copy.deepcopy(base_model)
        self.encoder.lm_head = Linear(base_model.config.n_embd, latent_dim)
        self.projection = nn.Linear(latent_dim, base_model.config.n_embd)
        self.decoder = Linear(base_model.config.n_embd, base_model.config.n_embd)
        #self.decoder = copy.deepcopy(base_model) # what about positional embeddings?
        #self.decoder.lm_head = Linear(base_model.config.n_embd, base_model.config.n_embd)

    def forward(self, input_embeds, attention_mask=None):
        latent = self.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask).logits
        #latent = self.encoder(input_embeds)
        projected = self.projection(latent)
        #return projected
        reconstructed_embeddings = self.decoder(projected)
        #reconstructed_embeddings = self.decoder(inputs_embeds = projected).logits
        #assert reconstructed_embeddings.shape == input_embeds.shape, f"shape must match of input and output, got {reconstructed_embeddings.shape} and {input_embeds.shape}"
        return reconstructed_embeddings
    

class LinearAutoEncoder(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=100):
        super().__init__()

        # Load the pretrained model
        base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        self.latent_dim = latent_dim

        # Create the encoder
        self.encoder = Linear(base_model.config.n_embd, latent_dim)
        self.decoder =  Linear(latent_dim, base_model.config.n_embd)

    def forward(self, inputs_embeds, attention_mask=None):
        # Encode the input
        latent = self.encoder(inputs_embeds)
        # Decode the projected representation
        reconstructed_embeddings = self.decoder(latent)
        return reconstructed_embeddings
