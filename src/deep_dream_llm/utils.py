"""
A file for useful functions and for plotting
"""

import openai
import numpy as np
import torch
from tqdm import tqdm
import random
import textwrap

import matplotlib.pyplot as plt

from torch.cuda.amp import autocast
from sklearn.metrics.pairwise import cosine_similarity

PRINT_EVERY = 150


def get_sentence_similarity(sentence1, sentence2):
    # Get embeddings for both sentences
    response1 = openai.Embedding.create(input=sentence1, model="text-embedding-ada-002")
    response2 = openai.Embedding.create(input=sentence2, model="text-embedding-ada-002")

    embedding1 = np.array(response1["data"][0]["embedding"])
    embedding2 = np.array(response2["data"][0]["embedding"])

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
    # Decode tokens into text using the tokenizer
    text = tokenizer.batch_decode(
        tokens.tolist(), max_length=model.config.n_ctx, padding="max_length"
    )
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
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the tokens into a sentence
    sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return sentence


def generate_sentence_batched(model, tokenizer, sentence_length=50, n=100):
    # get the device of the model
    device = next(model.parameters()).device

    # Generate random tokens in batches
    random_tokens = torch.randint(0, 50256, size=(n, 1)).to(device)

    # Generate sequences of tokens in batches
    output = model.generate(
        random_tokens,
        max_length=sentence_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode the tokens into a sentence
    sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    return sentences


def gen_sentences(model, tokenizer, n=10, sentence_length=50):
    """
    For debuggingg the autoendoer. Generates a set amount of sentences
    that makes training easier
    Args:
        model - the model to use
        tokenizer - the tokenizer to use
        n - the number of sentences to generate
        sentence_length - the max length of each sentence
    """
    pbar = tqdm(range(n), desc="Generating Sentences")
    sentences = []
    # Generate random pairs of sentences
    for _ in pbar:
        sentences.append(
            generate_sentence(model, tokenizer, max_length=sentence_length)
        )
    return sentences


def update_plot(losses, openai_losses, reencode_losses):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss", color="b")
    ax1.plot(losses, color="b")
    if reencode_losses:
        ax1.tick_params("y", colors="r")
        ax1.set_ylabel("Reencode Loss", color="r", labelpad=15)
        ax1.plot([PRINT_EVERY*i for i in range(len(reencode_losses))], reencode_losses, color="r")
        ax1.set_ylim(bottom=0)
    if openai_losses:
        ax2 = ax1.twinx()
        ax2.set_ylabel("OpenAI Loss", color="g")
        ax2.plot([PRINT_EVERY*i for i in range(len(openai_losses))], openai_losses, color="g")
        ax2.set_ylim(-1, 1)
    fig.tight_layout()
    plt.show()


def print_results(
    epoch,
    original_sentence,
    reconstructed_sentence,
    loss,
    openai_loss,
    reencode_loss,
    total_epochs,
):
    print(f"Epoch {epoch}/{total_epochs}, Loss: {round(loss, 4)}")
    print(
        f"Openai Loss: {round(openai_loss, 4)}"
    )
    print(f")
    print(f"Original: {textwrap.fill(original_sentence, width=80)}")
    print(f"Reconstructed: {textwrap.fill(reconstructed_sentence[:100], width=80)}")
    print(
        f"Lengths: Orig = {len(original_sentence)}, Recon = {len(reconstructed_sentence)}"
    )
