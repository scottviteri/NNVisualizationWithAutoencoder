"""
This is the file for defining the auto-encoder classes to experiment with
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
    def __init__(self, model_checkpoint, latent_dim=100, nhead=2, num_layers=6):
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
