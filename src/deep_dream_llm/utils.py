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

    Returns:
        text - the decoded text, which may contain more or less tokens than before
    """
    # original_shape = embeds_input.shape
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
            _, tokens = torch.max(dot_product, dim=-1)
    # Decode tokens into text using the tokenizer
    text = tokenizer.batch_decode(tokens.tolist())
    # # Encode the text again to verify number of tokens is the same
    # encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    # # Verify that the number of tokens is the same
    # assert encoded.input_ids.shape[:2] == original_shape[:2], (
    #     f"Number of tokens is not the same after decoding. "
    #     f"Expected {original_shape[:2]} but got {encoded.input_ids.shape[:2]}"
    # )
    return text


def optimize_for_neuron_whole_input(
    model,
    tokenizer,
    autoencoder,
    neuron_index=0,
    layer_num=1,
    mlp_or_attention="mlp",
    num_tokens=50,
    num_iterations=200,
    loss_fn=None,
    learning_rate=0.1,
    seed=42,
    sentence=None,
    verbose=False
):
    """
    Args:
        model (GPT2LMHeadModel): the model to optimize for
        tokenizer (GPT2Tokenizer): the tokenizer to use
        autoencoder (Autoencoder): the autoencoder to use
        neuron_index (int): the index of the neuron to optimize for
        layer_num (int): the layer number to optimize for
        mlp_or_attention (str): 'mlp' or 'attention'
        num_tokens (int): the number of tokens to in the sentence that we optimize over
        num_iterations (int): the number of iterations to run the optimization for
        loss_fn (function): the loss function to use.
        learning_rate (float): the learning rate to use for the optimizer
        seed (int): the seed to use for reproducibility
    Returns:
        losses (list): the list of losses of the model, shape [num_iterations]
        log_dict (dict): has shape [num_iterations // 30] and has keys below
            original_sentence (str): the original generated sentence
            original_sentence_reconstructed (str) : the original sentence after reconstructing it
            reconstructed_sentences (list): Reconstructed sentences during training every 1/30th of the way through
            activations (list): Average activations of the neuron every 1/30th of the way through
    """
    log_dict = {}
    device = next(model.parameters()).device
    if loss_fn is None:
        loss_fn = lambda x: -torch.sigmoid(x)

    # Set the seed for reproducibility
    if sentence is None:
        if verbose: tqdm.write("Generating sentence because no sentence was provided")
        sentence = generate_sentence(
            model, tokenizer, max_length=num_tokens, seed=seed
        )
    if verbose: tqdm.write("Original sentence is:")
    if verbose: tqdm.write(sentence)
    log_dict["original_sentence"] = sentence

    input_ids = tokenizer.encode(
            sentence,
            return_tensors="pt",
        ).to(device)
    original_embeddings = model.transformer.wte(input_ids)
    latent = autoencoder.encode(original_embeddings, attention_mask=None) # batch size 1, no mask needed
    latent_vectors = latent.detach().clone().to(device)
    latent_vectors.requires_grad = True

    if verbose: tqdm.write("original reconstructed sentence is ")
    with torch.no_grad():
        og_reconstructed_sentence = unembed_and_decode(
            model, tokenizer, autoencoder.decode(latent_vectors, attention_mask=None)
        )
        log_dict["original_sentence_reconstructed"] = og_reconstructed_sentence
    # Create an optimizer for the latent vectors
    optimizer = torch.optim.AdamW(
        [latent_vectors], lr=learning_rate
    )  # You may need to adjust the learning rate

    if "mlp" in mlp_or_attention:
        layer = model.transformer.h[layer_num].mlp.c_fc
    elif "attention" in mlp_or_attention:
        layer = model.transformer.h[layer_num].attn.c_attn
    else:
        raise NotImplementedError("Haven't implemented attention block yet")

    activation_saved = [torch.tensor(0.0, device=device)]

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
        embeddings = autoencoder.decode(latent_vectors, attention_mask=None)
        _ = model(
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
                model, tokenizer, embeddings
            )[0]
            if verbose: tqdm.write(reconstructed_sentence, end="")
            log_dict["reconstructed_sentences"].append(reconstructed_sentence)
            log_dict["activations"].append(activation_saved[0].item())
        optimizer.zero_grad()

    handle.remove()  # Don't forget to remove the hook!
    return losses, log_dict


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


def update_plot(losses, direct_losses, openai_losses, reencode_losses, print_every, save_path=None):
    """
    Args:
        losses - a list of losses
        openai_losses - a list of openai losses
        reencode_losses - a list of reencode losses
        print_every - how often to print the losses
        save_path - where to save the plot    
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss", color="b")
    xs = [i*print_every for i in range(len(losses)//print_every)]
    ax1.set_ylim(bottom=0, top=max(losses[-1*(len(losses)//2):]))
    ax1.plot(xs, [losses[x] for x in xs], color="b")
    if direct_losses:
        xs = [i*print_every for i in range(len(direct_losses)//print_every)]
        ax1.plot(xs, [direct_losses[x] for x in xs], color="y")
    if reencode_losses:
        ax1.tick_params("y", colors="r")
        ax1.set_ylabel("Reencode Loss", color="r", labelpad=15)
        ax1.plot([print_every*i for i in range(len(reencode_losses))], reencode_losses, color="r")
    if openai_losses:
        ax2 = ax1.twinx()
        ax2.set_ylabel("OpenAI Loss", color="g")
        ax2.plot([print_every*i for i in range(len(openai_losses))], openai_losses, color="g")
        ax2.set_ylim(-1, 1)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.draw()


def print_results(
    epoch,
    original_sentence,
    reconstructed_sentence,
    loss,
    direct_loss,
    openai_loss,
    reencode_loss,
    total_epochs,
    learning_rate = None
):
    print(f"Epoch {epoch}/{total_epochs}, Loss: {round(loss, 4)}, Direct Loss: {round(direct_loss, 4)}")
    if learning_rate: print(f"LR: {learning_rate}")
    if openai_loss: print(f"Openai Loss: {round(openai_loss, 4)}")
    if reencode_loss: print(f"Reencode Loss: {round(reencode_loss, 4)}")
    print(f"Original: {textwrap.fill(repr(original_sentence), width=80)}")
    print(f"Reconstructed: {textwrap.fill(repr(reconstructed_sentence), width=80)}")
    print(f"Lengths: Orig = {len(repr(original_sentence))} and Recon = {len(repr(reconstructed_sentence))}")
    