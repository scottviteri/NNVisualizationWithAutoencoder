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
from collections import defaultdict

from deep_dream_llm.utils import (
    unembed_and_decode,
    update_plot,
    optimize_for_neuron_whole_input,
    generate_sentence,
    generate_sentence_batched,
    optimize_for_neuron_whole_input_batched,
    get_device
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
    batch_size = 1
    cfg = config.TrainingConfig(
        autoencoder_name="LinearAutoEncoder",
        latent_dim=20,
        base_model_name="distilbert-base-uncased",
        # load_path=LOAD_PATH,
        # autoencoder=autoencoder,
        learning_rate=1e-4,
        use_openai=False,
        # lr_scheduler=lr_scheduler,
        is_notebook=False,
        batch_size=batch_size,
    )
    trainer = DeepDreamLLMTrainer(cfg)
    n_epochs = 10
    (
        losses,
        openai_losses,
        reencode_losses,
        sentences,
        reconstructed_sentences,
    ) = trainer.train_autoencoder(
        num_epochs=n_epochs,
        print_every=1,
        save_path="Checkpoints/linear_testing.pt",
        num_sentences=n_epochs * batch_size,
    )


def experiment_2(args):
    """
    Optimizes random sentences for 10 neurons across all 6 mlp layers in GPT2.
    """
    DEBUG_MODE = False
    TEXT_OUTPUT_PATH = "experiments/demo/optimized_sentences_TAE_l20_l8_h8.json"
    TRY_BATCHED_OPTIMIZATION = False
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    autoencoder = TAE("distilgpt2", latent_dim=20)
    repo_id = "Scottviteri/TransformerAutoencoderLatentDim20"
    model_file_name = "TAE_l20_l8_h8.pt"
    download_path = hf_hub_download(repo_id=repo_id, filename=model_file_name)
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"
    autoencoder.load_state_dict(torch.load(download_path, map_location=map_location))
    device = get_device()
    autoencoder.to(device)
    model.to(device)
    
    seed = 42
    num_tokens = 50
    number_of_sentences = 5
    original_sentences = generate_sentence_batched(
        model, tokenizer, sentence_length=num_tokens, n=number_of_sentences
    )
    print(f"Sentence we are optimizing: {original_sentences}")

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
        neuron_dict = defaultdict(list)
        neuron_dict["Original Sentences"].extend(original_sentences)
        output_optimized_sentences[layer_num][neuron_index] = neuron_dict
        if DEBUG_MODE and neuron_index == 3:
            break
        if TRY_BATCHED_OPTIMIZATION:
            losses, log_dict = optimize_for_neuron_whole_input_batched(
                model=model,
                tokenizer=tokenizer,
                autoencoder=autoencoder,
                neuron_index=neuron_index,
                layer_num=layer_num,
                num_iterations=64,
                learning_rate=0.1,
                sentences=original_sentences,
            )
            final_reconstructed_sentence = log_dict["final_reconstructed_sentences"]
            neuron_dict["TAE_l20_l8_h8 final reconstructed sentences"].extend(final_reconstructed_sentence)
        else:
            for sentence in original_sentences:
                losses, log_dict = optimize_for_neuron_whole_input(
                    model=model,
                    tokenizer=tokenizer,
                    autoencoder=autoencoder,
                    neuron_index=neuron_index,
                    layer_num=layer_num,
                    num_iterations=64,
                    learning_rate=0.1,
                    sentence=sentence,
                )
                neuron_dict["TAE_l20_l8_h8 final reconstructed sentences"].append(log_dict["reconstructed_sentences"][-1])
        
    end_time = time.time()
    print(output_optimized_sentences)
    print(f"Total time: {end_time - start_time}")
    # save output_optimized_sentences to a json file
    with open(TEXT_OUTPUT_PATH, "w") as file:
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
