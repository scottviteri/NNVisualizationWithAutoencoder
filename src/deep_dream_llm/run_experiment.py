"""
A sample experiment runner.
Each function is a self contained experiment.

TODO: Move this file outside src and into an experiments folder
TODO: Fix the whole thing where generated sentences include a ton of \n tokens.
"""

import torch
from torch.optim import AdamW
import accelerate
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import pandas as pd
import tabulate
import argparse
import os
import sys

from deep_dream_llm.utils import unembed_and_decode, update_plot, optimize_for_neuron_whole_input
from autoencoder import LinearAutoEncoder, Gpt2AutoencoderBoth, TAE, MockAutoencoder
from training import DeepDreamLLMTrainer
import config

LOAD_PATH = "Checkpoints"
    

def train_autoencoder_experiment(args):
    train_autoencoder = args.train_autoencoder
    save_path = os.path.join("Checkpoints", args.save_path)
    n_epochs = args.n_epochs
    autoencoder_name = args.autoencoder_name
    lr_scheduler = args.lr_scheduler
    print_every=100

    base_model_name = args.base_model_name

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    if "LinearAutoEncoder" == autoencoder_name:
        autoencoder = LinearAutoEncoder(base_model_name)
    elif "Gpt2AutoencoderBoth" == autoencoder_name:
        autoencoder = Gpt2AutoencoderBoth(base_model_name)
    elif "TAE" == autoencoder_name:
        autoencoder = TAE(base_model_name)
    elif "mock" == autoencoder_name:
        autoencoder = MockAutoencoder()
    else:
        raise NotImplementedError(f"Autoencoder {autoencoder_name} not implemented")
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.autoencoder_lr)
    if lr_scheduler == "LinearWarmup":
        """
        This scales the lr coeff from 0 to 1 over the first 10% of the epochs,
        and then scales it from 1 to 0 over the remaining 90% of the epochs.
        """
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 10 * epoch / n_epochs if epoch / n_epochs < 0.1 else 1 - epoch / n_epochs)
    else:
        print("Not using a lr_scheduler")
        lr_scheduler = None

    cfg = config.TrainingConfig(
        autoencoder_name=autoencoder_name,
        load_path=LOAD_PATH,
        tokenizer=tokenizer,
        model=model,
        autoencoder=autoencoder,
        optimizer=optimizer,
        use_openai=False,
        lr_scheduler=lr_scheduler,
        save_path=save_path,
        is_notebook=False,
        batch_size=args.batch_size,
    )

    trainer = DeepDreamLLMTrainer(cfg)
    if train_autoencoder:
        output = trainer.train_autoencoder(
            num_epochs=n_epochs, print_every=print_every, save_path=save_path
        )
        losses, openai_losses, reencode_losses, sentences, reconstructed_sentences = output
        update_plot(losses, openai_losses, reencode_losses, print_every, save_path=f"{autoencoder_name}-{n_epochs}-epochs-lr-{args.autoencoder_lr}.png")

    return trainer


def optimize_encoding_average(args, trainer: DeepDreamLLMTrainer, plot=True):
    """
    Loads a trained autoencoder and then optimizes the encoding for a random
    sentence in order to maximally activate a neuron in the model.
    """
    seed = 42
    if args.optimize_n_sentences > 1:
        seed = None

    losses, log_dict = trainer.optimize_for_neuron_whole_input(
        neuron_index=args.neuron_index,
        layer_num=args.layer_num,
        num_tokens=args.num_tokens,
        num_iterations=100,
        seed=seed,
        verbose=False
    )
    
    # display in a table the sentences in log_dict using tabulate
    df = pd.DataFrame.from_dict({"reconstructed_sentences": log_dict["reconstructed_sentences"], "activations": log_dict["activations"]})
    if plot:
        print("Original sentence: ")
        print(log_dict["original_sentence"])
        print("Reconstructed sentence: ")
        print(log_dict["original_sentence_reconstructed"])
        print(tabulate.tabulate(df, headers="keys", tablefmt="psql"))

        # Generate x-axis values
        loss_iterations = range(1, len(losses) + 1)

        # Plot the losses
        plt.plot(loss_iterations, losses, "-o")

        # Set the plot title and labels
        plt.title("Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # Show the plot
        plt.show()
    return {"original_sentence": log_dict["original_sentence"], "original_sentence_reconstructed": log_dict["original_sentence_reconstructed"], "activations": log_dict["activations"], "final_sentence": log_dict["reconstructed_sentences"][-1]}


def baseline_optimize_encoding():
    def optimize_for_neuron(
        starting_sentence, layer_num=1, neuron_index=0, mlp_or_attention="mlp"
    ):
        """
        Args:
            neuron_indices: List of indices.
            mlp_or_attention (str): 'mlp' or 'attention'
        """
        model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
        inputs = tokenizer(starting_sentence, return_tensors="pt").to(device)

        # Get embeddings
        with torch.no_grad():
            embeddings = model.transformer.wte(inputs["input_ids"])

        # Make embeddings require gradient
        embeddings.requires_grad_(True)

        # Create an optimizer for the embeddings
        optimizer = AdamW(
            [embeddings], lr=0.1
        )  # You may need to adjust the learning rate
        pre_embeddings = embeddings.detach().clone()
        print(embeddings)
        print(unembed_and_decode(pre_embeddings))
        len_example = embeddings.shape[1] - 1

        if "mlp" in mlp_or_attention:
            layer = model.transformer.h[layer_num].mlp
        else:
            raise NotImplementedError("Haven't implemented attention block yet")
        activation_saved = [torch.tensor(0.0)]

        def hook(model, input, output):
            # The output is a tensor. You can index it to get the activation of a specific neuron.
            # Here we're getting the activation of the 0th neuron.
            # TODO: Figure out what neruon this is actually grabbing. Why is it
            activation = output[0, len_example, neuron_index]
            activation_saved[0] = activation

        handle = layer.register_forward_hook(hook)

        losses = []
        dist = 0.0
        for i in tqdm(range(100)):
            outputs = model(
                inputs_embeds=embeddings, attention_mask=inputs.attention_mask
            )
            loss = -torch.sigmoid(activation_saved[0])
            loss.backward()
            optimizer.step()
            dist = torch.sum(embeddings - pre_embeddings).item()
            losses.append(loss)
            if i % 25 == 0:
                tqdm.write(f"\n{dist} and then {loss}\n")
                tqdm.write(unembed_and_decode(embeddings)[0])
            optimizer.zero_grad()

        return losses

    input_sentence_1 = "In the midst of a vibrant summer morning, with the sun casting its golden rays upon the lush green meadows and the fragrant wildflowers swaying gently in the warm breeze, a multitude of birds chirped melodiously while gracefully soaring across the clear blue sky, their wings glimmering like tiny diamonds as they embraced the boundless freedom of the open air, and nearby, a majestic oak tree stood tall and proud, its branches extending outward in a magnificent display of nature's artistry, providing shade and shelter for a variety of creatures that sought solace beneath its protective canopy, including a family of squirrels playfully darting between the branches, their bushy tails serving as vibrant accents against the backdrop of verdant leaves, and as the day progressed, the distant rumble of thunder gradually grew louder, heralding the imminent arrival of a summer storm, as dark clouds gathered overhead, casting an ephemeral gloom over the once vibrant landscape, yet even in the face of this impending tempest, there was an undeniable beauty in the contrast between the electric flashes of lightning that briefly illuminated the sky and the cascading raindrops that danced upon the earth, breathing life into the thirsty soil and rejuvenating the flora and fauna, and as the storm subsided, a mesmerizing rainbow emerged, arching gracefully across the horizon, its vibrant hues painting a breathtaking scene that filled hearts with awe and wonder, reminding us of the ever-present magic and resilience of nature, and in that fleeting moment, as the world basked in the afterglow of the storm, a profound sense of gratitude and harmony washed over everything, reminding us of our intricate connection to the vast tapestry of existence."
    input_sentence_2 = "The fundamental principles of calculus provide a powerful framework for understanding and analyzing the rates of change and accumulation of quantities in various fields of mathematics and science, enabling us to model and solve complex real-world problems with precision and rigor."
    input_sentence_3 = "I'm sorry for the misunderstanding, but as an AI developed by OpenAI, I don't have direct access to individual sentences or documents from my training data. I was trained on a mixture of licensed data, data created by human trainers, and publicly available data. These sources may contain a wide range of data, including books, websites, and other texts, so I don't have the ability to recall or generate any specific sentence from the training data. I generate responses based on patterns and information in the data I was trained on."
    losses = optimize_for_neuron(input_sentence_3, neuron_index=2, layer_num=5)
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot([loss.cpu().detach() for loss in losses])
    plt.title("Loss curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def baseline_optimize_encoding_average():
    def optimize_for_neuron_whole_input(
        neuron_index=0,
        layer_num=1,
        mlp_or_attention="mlp",
        num_tokens=10,
        num_iterations=200,
    ):
        """
        Args:
        neuron_indices: List of indices.
        mlp_or_attention (str): 'mlp' or 'attention'
        """
        model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

        # Start with random embeddings
        embeddings = torch.randn(
            (1, num_tokens, model.config.n_embd), device=device, requires_grad=True
        )

        # Create an optimizer for the embeddings
        optimizer = AdamW(
            [embeddings], lr=0.1
        )  # You may need to adjust the learning rate

        if "mlp" in mlp_or_attention:
            layer = model.transformer.h[layer_num].mlp
        else:
            raise NotImplementedError("Haven't implemented attention block yet")

        activation_saved = [torch.tensor(0.0, device=device)]

        def hook(model, input, output):
            # The output is a tensor. We're getting the average activation of the neuron across all tokens.
            activation = output[0, :, neuron_index].mean()
            activation_saved[0] = activation

        handle = layer.register_forward_hook(hook)

        pbar = tqdm(range(num_iterations), position=0, leave=True)
        losses = []
        for i in pbar:
            # Construct input for the model using the embeddings directly
            outputs = model(inputs_embeds=embeddings)
            # We want to maximize activation, which is equivalent to minimizing negative activation
            loss = -torch.sigmoid(activation_saved[0])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % (num_iterations // 30) == 0:
                pbar.set_description(
                    f"""Loss at step {i}: {loss.item()}\n\n
                                        Current sentence: {unembed_and_decode(embeddings)[0]}\n"""
                )
            optimizer.zero_grad()

        handle.remove()  # Don't forget to remove the hook!
        return losses

    losses = optimize_for_neuron_whole_input(neuron_index=2, layer_num=5, num_tokens=20)
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Loss curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def optimize_bunch_of_sentences(args, trainer):
    og_sentences = []
    og_reconstructed_sentences = []
    reconstructed_sentences = []
    for _ in tqdm(range(args.optimize_n_sentences), desc="Num_sentences"):
        log_out = optimize_encoding_average(args, trainer, plot=False)
        og_sentences.append(log_out["original_sentence"])
        og_reconstructed_sentences.append(log_out["original_sentence_reconstructed"])
        reconstructed_sentences.append(log_out["final_sentence"])
    # make this into a table and print the table
    df = pd.DataFrame.from_dict({"og_sentences": og_sentences, "og_reconstructed_sentences": og_reconstructed_sentences, "final_sentences": reconstructed_sentences})
    table = tabulate.tabulate(df, headers="keys", tablefmt="simple_grid")
    print(table)
    root_dir = args.experiment_save_dir
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    filename = os.path.join(root_dir, f"{args.autoencoder_name}-layer-{args.layer_num}-neuron_index-{args.neuron_index}-table")
    # check if filename exists
    if os.path.exists(filename):
        filename += "-1"
    filename += ".txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(table)
    print(f"Saved table to {filename}")


def experiment_0(args):
    """
    Optionaly trains an autoencoder, and then optimizes it for a bunch of sentences.
    Saves the optimized sentences to a folder.
    """
    trainer = train_autoencoder_experiment(args)
    optimize_bunch_of_sentences(args, trainer)

def experiment_1(args):
    REPO_ID = "Scottviteri/TransformerAutoencoderLatentDim7"
    FILENAME = args.autoencoder_name + ".pt"

    ae = TAE("distilgpt2", latent_dim=7)
    download_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(download_path)
    ae.load_state_dict(torch.load(download_path, map_location=torch.device('cpu')))

    cfg = config.TrainingConfig(
        autoencoder_name=args.autoencoder_name,
        autoencoder=ae,
        learning_rate=0.0001,
        latent_dim=7,
        batch_size=64,
        use_openai=False,
        is_notebook=False
    )
    trainer = DeepDreamLLMTrainer(cfg)
    optimize_bunch_of_sentences(args, trainer)

def experiment_2(args):
    REPO_ID = "Scottviteri/TransformerAutoencoderLatentDim7"
    FILENAME = args.autoencoder_name + ".pt"

    ae = TAE("distilgpt2", latent_dim=7)
    download_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(download_path)
    ae.load_state_dict(torch.load(download_path, map_location=torch.device('cpu')))

    cfg = config.TrainingConfig(
        autoencoder_name=args.autoencoder_name,
        autoencoder=ae,
        learning_rate=0.0001,
        latent_dim=7,
        batch_size=64,
        use_openai=False,
        is_notebook=False
    )
    trainer = DeepDreamLLMTrainer(cfg)
    neurons = []
    for layer_id in range(5):
        for neuron_id in range(1,3):
            neurons.append({"layer_num": layer_id, "neuron_index": neuron_id})
    for neuron in tqdm(neurons, desc="Neurons", total=len(neurons)):
        args.layer_num = neuron["layer_num"]
        args.neuron_index = neuron["neuron_index"]
        optimize_bunch_of_sentences(args, trainer)

def experiment_3(args):
    cfg = config.TrainingConfig(
        autoencoder_name=args.autoencoder_name,
        learning_rate=0.0001,
        latent_dim=100,
        batch_size=64,
        use_openai=False,
        is_notebook=False,
        load_path=LOAD_PATH,
    )
    trainer = DeepDreamLLMTrainer(cfg)
    neurons = []
    for layer_id in range(5):
        for neuron_id in range(1,3):
            neurons.append({"layer_num": layer_id, "neuron_index": neuron_id})
    for neuron in tqdm(neurons, desc="Neurons", total=len(neurons)):
        args.layer_num = neuron["layer_num"]
        args.neuron_index = neuron["neuron_index"]
        optimize_bunch_of_sentences(args, trainer)

def experiment_4(args):
    """
    Test for optimize for neuron whole input
    """
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print("Device:", device)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    autoencoder = LinearAutoEncoder("distilgpt2")
    autoencoder, model = accelerator.prepare(autoencoder, model)
    losses, log_output = optimize_for_neuron_whole_input(
        model=model,
        tokenizer=tokenizer,
        autoencoder=autoencoder,
    )
    # graph the loss
    plt.plot(losses)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_autoencoder",
        action="store_true",
        help="Whether to train the autoencoder or not",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="test_1.pt",
        help="Path to save the autoencoder to",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=2,
        help="Number of epochs to train the autoencoder for",
    )
    parser.add_argument(
        "--autoencoder_name",
        type=str,
        default="LinearAutoEncoder",
        help="Name of the autoencoder to use",
    )
    parser.add_argument(
        "--autoencoder_lr",
        type=float,
        default=0.001,
        help="Learning rate for the autoencoder",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="",
        help="Learning rate lr_scheduler for the autoencoder",
    )
    parser.add_argument(
        "--autoencoder_path",
        type=str,
        default="",
        help="Path to a pretrained autoencoder",
    )
    parser.add_argument(
        "--optimize_n_sentences",
        type=int,
        default=1,
        help="Whether to optimize many sentences or not",
    )
    parser.add_argument(
        "--neuron_index",
        type=int,
        default=0,
        help="Index of the neuron to optimize for",
    )
    parser.add_argument(
        "--layer_num",
        type=int,
        default=1,
        help="Index of the layer to optimize for",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=20,
        help="Number of tokens to optimize for in a sentence",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="distilgpt2",
        help="Name of the base model to use",
    )
    parser.add_argument(
        "--experiment_num",
        type=int,
        default=0,
        help="Which test to run",
    )
    parser.add_argument(
        "--experiment_save_dir",
        type=str,
        default="experiments",
        help="Directory to save experiments to",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training the autoencoder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        func = getattr(sys.modules[__name__], f'experiment_{args.experiment_num}')
        print(f"Running experiment_{args.experiment_num}")
        func(args)
    except AttributeError as e:
        print(f"Caught error: {e}")
        print(f"No such experiment: {args.experiment_num}")

if __name__ == "__main__":
    main()
