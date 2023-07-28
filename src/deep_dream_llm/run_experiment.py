"""
A sample experiment runner.
Each function is a self contained experiment.

TODO: Move this file outside src and into an experiments folder
TODO: Fix the whole thing where generated sentences include a ton of \n tokens.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import tabulate
import argparse

from utils import unembed_and_decode, update_plot
from autoencoder import LinearAutoEncoder, Gpt2AutoencoderBoth, TAE, MockAutoencoder
from training import DeepDreamLLMTrainer


def shape_test_autoencoder(autoencoder):
    """
    Args:

    """
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=0.01)
    # Generate a random sentence
    trainer = DeepDreamLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        autoencoder=autoencoder,
        use_openai=False,
    )
    


def train_autoencoder_experiment(args):
    train_autoencoder = args.train_autoencoder
    save_path = args.save_path
    n_epochs = args.n_epochs
    autoencoder_name = args.autoencoder_name

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    if "LinearAutoEncoder" == autoencoder_name:
        autoencoder = LinearAutoEncoder("distilgpt2")
    elif "Gpt2AutoencoderBoth" == autoencoder_name:
        autoencoder = Gpt2AutoencoderBoth("distilgpt2")
    elif "TAE" == autoencoder_name:
        autoencoder = TAE("distilgpt2")
    elif "mock" == autoencoder_name:
        autoencoder = MockAutoencoder()
    else:
        raise NotImplementedError(f"Autoencoder {autoencoder_name} not implemented")
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=0.01)

    trainer = DeepDreamLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        autoencoder=autoencoder,
        use_openai=False,
    )
    # tokenizer.pad_token = tokenizer.eos_token
    # TODO I think a smaller lr will do better
    if train_autoencoder:
        (
            losses,
            openai_losses,
            reencode_losses,
            reconstructed_sentences,
        ) = trainer.train_autoencoder(
            num_epochs=n_epochs, print_every=100, use_openai=False, save_path=save_path
        )
        update_plot(losses, openai_losses, reencode_losses)

    return trainer


def optimize_encoding_average(trainer: DeepDreamLLMTrainer):
    """
    Loads a trained autoencoder and then optimizes the encoding for a random
    sentence in order to maximally activate a neuron in the model.
    """

    losses, log_dict = trainer.optimize_for_neuron_whole_input(
        neuron_index=2,
        layer_num=5,
        num_tokens=20,
        num_iterations=100,
    )
    
    # display in a table the sentences in log_dict using tabulate
    df = pd.DataFrame.from_dict({"reconstructed_sentences": log_dict["reconstructed_sentences"], "activations": log_dict["activations"]})
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


# prompt: Generate random pairs of sentences using gpt2, and get the average ada embedding distance

# def calc_average_emb_distance(num_pairs):
#     # Initialize the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

#     # Initialize gpt2
#     model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
#     model.eval()

#     pbar = tqdm(range(num_pairs))
#     similarity_values = []
#     # Generate random pairs of sentences
#     for i in pbar:
#         sentence1 = generate_sentence(model, tokenizer, max_length=50)
#         sentence2 = generate_sentence(model, tokenizer, max_length=50)
#         # Compute cosine similarity between embeddings
#         similarity = get_sentence_similarity(sentence1, sentence2)
#         # Store the similarity value
#         similarity_values.append(similarity)
#         pbar.set_description(f"Running average: {np.mean(similarity_values)}")
#     # Print the average similarity value
#     return np.mean(similarity_values)

# # calc_average_emb_distance(10) #0.6956303872374302


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
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = train_autoencoder_experiment(args)
    optimize_encoding_average(trainer)


if __name__ == "__main__":
    main()
