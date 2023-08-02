"""
Code for all our final experiments
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import pandas as pd
import tabulate
import argparse
import os
import sys
import json
import numpy as np
import time

from deep_dream_llm.utils import (
    unembed_and_decode,
    update_plot,
    optimize_for_neuron_whole_input,
    generate_sentence,
)
from deep_dream_llm.autoencoder import (
    LinearAutoEncoder,
    Gpt2AutoencoderBoth,
    TAE,
    MockAutoencoder,
)
from deep_dream_llm.training import DeepDreamLLMTrainer
import deep_dream_llm.config as config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=0,
        help="The number identifier for the experiment to run",
    )
    return parser.parse_args()


def experiment_0(args):
    """
    Trains a several linear autoencoders with a different number of latent dimensions
    Saves the final loss of this model
    """
    # make a trainer with the linear autoencoder
    # epochs to plot
    epochs_to_plot = [10, 32, 64, 128, 256]
    latent_dims, losses_total = {}, {}
    for latent_dim in tqdm(range(1, 102, 10), desc="Latent Dims"):
        batch_size = 1
        cfg = config.TrainingConfig(
            autoencoder_name="LinearAutoEncoder",
            latent_dim=latent_dim,
            # load_path=LOAD_PATH,
            # autoencoder=autoencoder,
            learning_rate=1e-4,
            use_openai=False,
            # lr_scheduler=lr_scheduler,
            is_notebook=False,
            batch_size=batch_size,
        )
        trainer = DeepDreamLLMTrainer(cfg)
        n_epochs = epochs_to_plot[-1]
        (
            losses,
            openai_losses,
            reencode_losses,
            sentences,
            reconstructed_sentences,
        ) = trainer.train_autoencoder(
            num_epochs=n_epochs,
            print_every=100000,
            save_path="Checkpoints/linear_testing.pt",
            num_sentences=n_epochs * batch_size,
        )
        for epoch in epochs_to_plot:
            losses_avg_10 = np.mean(losses[epoch - 10 : epoch])
            latent_dims.setdefault(epoch, []).append(latent_dim)
            losses_total.setdefault(epoch, []).append(losses_avg_10)
    plot_0(losses_total, latent_dims, "linear_autoencoder_latent_dim_vs_loss")


def plot_0(losses, latent_dims, plot_name):
    """
    Plots on the x axis the latent_dim, and
    on the y-axis the losses. Plots in a separate color each of the epochs

    Args:
        losses (dict): A dictionary mapping epoch to list of losses
        latent_dims (dict): A dictionary mapping epoch to list of latent_dims
    """
    fig, ax = plt.subplots()
    for epoch in losses:
        ax.plot(latent_dims[epoch], losses[epoch], label=f"Epoch {epoch}")
    ax.set_xlabel("Latent Dim")
    ax.set_ylabel("Loss")
    ax.set_title(plot_name)
    ax.legend()
    plt.savefig(f"experiments/plots/{plot_name}.png")
    plt.close(fig)


def experiment_1(args):
    """
    Trains a complete autoencoder.
    TODO Complete this function
    """
    cfg = config.TrainingConfig(
        autoencoder_name="LinearAutoEncoder",
        latent_dim=latent_dim,
        # load_path=LOAD_PATH,
        # autoencoder=autoencoder,
        learning_rate=1e-4,
        use_openai=False,
        # lr_scheduler=lr_scheduler,
        is_notebook=False,
        batch_size=batch_size,
    )
    trainer = DeepDreamLLMTrainer(cfg)


def experiment_2(args):
    """
    Optimizes random sentences for 10 neurons across all 6 mlp layers in GPT2.
    """
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    autoencoder = TAE("distilgpt2", latent_dim=20)

    seed = 42
    num_tokens = 50
    original_sentence = generate_sentence(
        model, tokenizer, max_length=num_tokens, seed=seed
    )
    print(f"Sentence we are optimizing: {original_sentence}")

    neurons = []
    output_optimized_sentences = []
    for layer_num in range(6):
        output_optimized_sentences.append([])
        for neuron_index in range(10):
            neurons.append({"layer_num": layer_num, "neuron_index": neuron_index})
            output_optimized_sentences[layer_num].append([])
    print(
        f"We have {len(output_optimized_sentences)} layers and {len(output_optimized_sentences[0])} neurons per layer"
    )
    start_time = time.time()
    for neuron in tqdm(neurons, desc="Neurons", total=len(neurons)):
        layer_num = neuron["layer_num"]
        neuron_index = neuron["neuron_index"]
        losses, log_dict = optimize_for_neuron_whole_input(
            model=model,
            tokenizer=tokenizer,
            autoencoder=autoencoder,
            neuron_index=neuron_index,
            layer_num=layer_num,
            num_iterations=64,
            learning_rate=0.1,
        )
        final_reconstructed_sentence = log_dict["reconstructed_sentences"][-1]
        output_optimized_sentences[layer_num][neuron_index].append(
            (original_sentence, final_reconstructed_sentence)
        )
    end_time = time.time()
    print(output_optimized_sentences)
    print(f"Total time: {end_time - start_time}")
    # save output_optimized_sentences to a json file
    with open("experiments/demo/optimized_sentences_test1.json", "w") as file:
        json.dump(output_optimized_sentences, file)


def main():
    args = parse_args()
    try:
        func = getattr(sys.modules[__name__], f"experiment_{args.experiment_id}")
        print(f"Running experiment_{args.experiment_id}")
        func(args)
    except AttributeError as error:
        print(f"Caught error: {error}")
        print(f"Potentialaly no such experiment: experiment_{args.experiment_id}")


if __name__ == "__main__":
    main()
