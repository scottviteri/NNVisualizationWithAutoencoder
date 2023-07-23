"""
A file for useful functions and for plotting
"""

import openai
import numpy as np
import torch
from tqdm import tqdm
import random

from torch.cuda.amp import autocast
from sklearn.metrics.pairwise import cosine_similarity

def get_sentence_similarity(sentence1, sentence2):

    # Get embeddings for both sentences
    response1 = openai.Embedding.create(input=sentence1, model="text-embedding-ada-002")
    response2 = openai.Embedding.create(input=sentence2, model="text-embedding-ada-002")

    embedding1 = np.array(response1['data'][0]['embedding'])
    embedding2 = np.array(response2['data'][0]['embedding'])

    # Compute cosine similarity between embeddings
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

    return similarity[0][0]

def unembed_and_decode(model, tokenizer, embeds_input):
    """
    Given an embedding vector, decode each token by using the transpose of the embedding matrix
    and grabbing the vocab token with the highest probability on each token.

    Also do this with the unembedding matrix as well.

    Args:
        model - the model to use
        tokenizer - the tokenizer to use
        embeds_input - the embeddings to decode
    """
    with torch.no_grad():
        with autocast():
            # Get the pre-trained embeddings
            pretrained_embeddings = model.transformer.wte.weight
            # if pretrained_embeddings.dtype != embeds_input.dtype:
            # These types don't match so we use auto cast.
            #   print(f"types don't match, got for embeds inputs { embeds_input.dtype}, and {pretrained_embeddings.dtype} for embeddings matrix from gpt2 model")
            # Calculate dot product between input embeddings and pre-trained embeddings
            dot_product = torch.matmul(embeds_input, pretrained_embeddings.t())

            # Get the index of the highest value along dimension 2 (tokens)
            _, tokens = torch.max(dot_product, dim=2)

    tokenizer.pad_token = tokenizer.eos_token
    # Decode tokens into text using the tokenizer
    text = tokenizer.batch_decode(tokens.tolist(), skip_special_tokens=True, max_length=model.config.n_ctx, padding="max_length")
    return text


def generate_sentence(model, tokenizer, max_length=50, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    # Randomly select a token
    random_token = random.randint(0, 50256)

    # get the device of the model
    device = next(model.parameters()).device

    # Convert the token to a tensor
    input_ids = torch.tensor([[random_token]]).to(device)

    # Generate a sequence of tokens
    output = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=1.0, pad_token_id=tokenizer.eos_token_id)

    # Decode the tokens into a sentence
    sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return sentence

def gen_sentences(model, tokenizer, device="cpu"):
    """
    For debuggingg the autoendoer. Generates a set amount of sentences
    that makes training easier
    Args:
        model - the model to use
        tokenizer - the tokenizer to use
    """
    pbar = tqdm(range(10))
    sentences = []
    # Generate random pairs of sentences
    for _ in pbar:
        sentences.append(generate_sentence(model, tokenizer, max_length=50))
    return sentences
