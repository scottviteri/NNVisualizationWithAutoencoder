import transformers

print(transformers.__version__)

import torch
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check if CUDA is available and choose device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_checkpoint = "distilgpt2"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
from transformers import AutoModelForCausalLM


def unembed_and_decode(embeds_input):
  """
  Given an embedding vector, decode each token by using the transpose of the embedding matrix
  and grabbing the vocab token with the highest probability on each token.

  Also do this with the unembedding matrix as well.
  """
  with torch.no_grad():
      # Get the pre-trained embeddings
      pretrained_embeddings = model.transformer.wte.weight
      # Calculate dot product between input embeddings and pre-trained embeddings
      dot_product = torch.matmul(embeds_input, pretrained_embeddings.t())

      # Get the index of the highest value along dimension 2 (tokens)
      _, tokens = torch.max(dot_product, dim=2)

  # Decode tokens into text using the tokenizer
  text = tokenizer.batch_decode(tokens.tolist(), skip_special_tokens=True)

  return text



def optimize_for_neuron(starting_sentence, layer_num=1, neuron_index=0, mlp_or_attention="mlp"):
  """
  Args:
    neuron_indices: List of indices.
    mlp_or_attention (str): 'mlp' or 'attention'
  """
  model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)
  inputs = tokenizer(starting_sentence, return_tensors="pt").to(device)

  # Get embeddings
  with torch.no_grad():
      embeddings = model.transformer.wte(inputs["input_ids"])

  # Make embeddings require gradient
  embeddings.requires_grad_(True)

  # Create an optimizer for the embeddings
  optimizer = AdamW([embeddings], lr=0.1)  # You may need to adjust the learning rate
  pre_embeddings = embeddings.detach().clone()
  print(embeddings)
  print(unembed_and_decode(pre_embeddings))
  len_example = embeddings.shape[1] - 1

  if 'mlp' in mlp_or_attention:
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
    outputs = model(inputs_embeds=embeddings, attention_mask=inputs.attention_mask)
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



# input_sentence_1 = "In the midst of a vibrant summer morning, with the sun casting its golden rays upon the lush green meadows and the fragrant wildflowers swaying gently in the warm breeze, a multitude of birds chirped melodiously while gracefully soaring across the clear blue sky, their wings glimmering like tiny diamonds as they embraced the boundless freedom of the open air, and nearby, a majestic oak tree stood tall and proud, its branches extending outward in a magnificent display of nature's artistry, providing shade and shelter for a variety of creatures that sought solace beneath its protective canopy, including a family of squirrels playfully darting between the branches, their bushy tails serving as vibrant accents against the backdrop of verdant leaves, and as the day progressed, the distant rumble of thunder gradually grew louder, heralding the imminent arrival of a summer storm, as dark clouds gathered overhead, casting an ephemeral gloom over the once vibrant landscape, yet even in the face of this impending tempest, there was an undeniable beauty in the contrast between the electric flashes of lightning that briefly illuminated the sky and the cascading raindrops that danced upon the earth, breathing life into the thirsty soil and rejuvenating the flora and fauna, and as the storm subsided, a mesmerizing rainbow emerged, arching gracefully across the horizon, its vibrant hues painting a breathtaking scene that filled hearts with awe and wonder, reminding us of the ever-present magic and resilience of nature, and in that fleeting moment, as the world basked in the afterglow of the storm, a profound sense of gratitude and harmony washed over everything, reminding us of our intricate connection to the vast tapestry of existence."
# input_sentence_2 = "The fundamental principles of calculus provide a powerful framework for understanding and analyzing the rates of change and accumulation of quantities in various fields of mathematics and science, enabling us to model and solve complex real-world problems with precision and rigor."
# input_sentence_3 = "I'm sorry for the misunderstanding, but as an AI developed by OpenAI, I don't have direct access to individual sentences or documents from my training data. I was trained on a mixture of licensed data, data created by human trainers, and publicly available data. These sources may contain a wide range of data, including books, websites, and other texts, so I don't have the ability to recall or generate any specific sentence from the training data. I generate responses based on patterns and information in the data I was trained on."
# losses = optimize_for_neuron(input_sentence_3, neuron_index=2, layer_num=5)
# # Plot losses
# plt.figure(figsize=(10,6))
# plt.plot([loss.cpu().detach() for loss in losses])
# plt.title('Loss curve')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

from tqdm import tqdm

def optimize_for_neuron_whole_input(neuron_index=0, layer_num=1, mlp_or_attention="mlp", num_tokens=10, num_iterations=200):
    """
    Args:
      neuron_indices: List of indices.
      mlp_or_attention (str): 'mlp' or 'attention'
    """
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)

    # Start with random embeddings
    embeddings = torch.randn((1, num_tokens, model.config.n_embd), device=device, requires_grad=True)

    # Create an optimizer for the embeddings
    optimizer = AdamW([embeddings], lr=0.1)  # You may need to adjust the learning rate

    if 'mlp' in mlp_or_attention:
        layer = model.transformer.h[layer_num].mlp
    else:
        raise NotImplementedError("Haven't implemented attention block yet")

    activation_saved = [torch.tensor(0.0, device=device)]
    def hook(model, input, output):
        # The output is a tensor. We're getting the average activation of the neuron across all tokens.
        activation = output[0, :, neuron_index].mean()
        activation_saved[0] = activation
    handle = layer.register_forward_hook(hook)

    losses = []
    for i in tqdm(range(num_iterations), position=0, leave=True):
        # Construct input for the model using the embeddings directly
        outputs = model(inputs_embeds=embeddings)
        # We want to maximize activation, which is equivalent to minimizing negative activation
        loss = -torch.sigmoid(activation_saved[0])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i % (num_iterations//30) == 0:
            tqdm.write(f"Loss at step {i}: {loss.item()}\n", end='')
            tqdm.write(unembed_and_decode(embeddings)[0], end='')
        optimizer.zero_grad()

    handle.remove()  # Don't forget to remove the hook!
    return losses


import copy

class TransformerAutoencoder(torch.nn.Module):
    def __init__(self, model_checkpoint, latent_dim=10):
        super().__init__()

        # Load the pretrained model
        base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

        # Create the encoder
        self.encoder = copy.deepcopy(base_model)
        self.encoder.lm_head = torch.nn.Linear(base_model.config.n_embd, latent_dim)

        self.up_project = torch.nn.Linear(latent_dim, base_model.config.n_embd)
        # Create the decoder
        self.decoder = copy.deepcopy(base_model)
        # self.decoder.transformer.wte = torch.nn.Linear(latent_dim, base_model.config.n_embd)
        # self.decoder.transformer.wpe = torch.nn.Embedding(base_model.config.max_position_embeddings, base_model.config.n_embd)
        self.decoder.lm_head = torch.nn.Linear(base_model.config.n_embd, base_model.config.n_embd)

    def forward(self, input_ids, attention_mask=None):
        # Encode the input
        latent = self.encoder(input_ids, attention_mask=attention_mask).logits

        # Project the latent representation to the original dimension
        latent = self.up_project(latent)
        # Decode the latent representation
        outputs = self.decoder(inputs_embeds=latent, attention_mask=attention_mask)

        return outputs

from torch.nn import MSELoss
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast

# Initialize autoencoder
autoencoder = TransformerAutoencoder('distilgpt2').to(device)

# Initialize loss function, optimizer, and gradient scaler for mixed-precision training
criterion = MSELoss()
optimizer = Adam(autoencoder.parameters())
scaler = GradScaler()

# Number of training steps
n_steps = 1000

# Size of each sentence
sentence_size = 20

base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)

for step in range(n_steps):
    # Generate a sentence from GPT-2
    generated = base_model.generate(max_length=sentence_size)
    sentence = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(sentence)

    # Embed the sentence
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        original_embeddings = base_model.transformer.wte(input_ids)

    # Run the autoencoder and compute the loss
    with autocast():  # Enables mixed-precision training
        reconstructed_embeddings = autoencoder(input_ids).logits
        loss = criterion(reconstructed_embeddings, original_embeddings)

    # Backpropagate the loss and update the weights
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Every 100 steps, print the loss and a reconstructed sentence
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}")
        print(f"Reconstructed sentence: {unembed_and_decode(reconstructed_embeddings)[0]}")
