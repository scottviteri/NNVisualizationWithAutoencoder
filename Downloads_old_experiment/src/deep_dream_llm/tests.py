"""
A file for tests to ensure code works as expect

TODO This file is a Work in progress
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm.auto import tqdm
import torch

import autoencoder
#import run_experiment
import utils
import config
import training

# def test_calc_loss():
#     assert calc_loss("there", "there") == 0.0
#     assert calc_loss("hello", "there") > 0.0
#     try:
#         calc_loss("hello there", "there")
#         print("THis doesn't faile which is weird")
#     except:
#         print("success")

def test_injective_tokenizer_inner(tokens=None, verbose=False):
    # load gpt2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # first we test Tokens to sentence to tokens is the identity
    if tokens is None:
        tokens = np.random.randint(0, 1000, size=(20,))
    sentence = tokenizer.decode(tokens)
    tokens2 = tokenizer.encode(sentence)
    TST = False
    if len(tokens) != len(tokens2) or np.all(tokens != tokens2):
        if verbose: print(f"T -> S -> T is not the identity. tokens: {tokens}\ntokens2: {tokens2}")
    else:
        if verbose: print("T -> S -> T is the identity")
        if verbose: print(tokens)
        TST = True
    # now we test that the sentence to tokens to sentence is not the identiy
    sentence2 = tokenizer.decode(tokens2)
    if verbose: print("------------------")
    STS = False
    if sentence == sentence2:
        if verbose: print("S -> T -> S is the identity")
        if verbose: print(sentence)
        STS = True
    else:
        if verbose: print("S -> T -> S is not the identity")
        if verbose: print(f"Sentence1: {sentence}\nSentence2: {sentence2}")
    if verbose: print("Done")
    return TST, STS

def test_injective_tokenizer():
    ones = np.ones((10,), dtype=np.int32)
    TST, STS = test_injective_tokenizer_inner(ones)
    for i in tqdm(range(10)):
        tmp1, tmp2 = test_injective_tokenizer_inner()
        TST = TST and tmp1
        STS = STS and tmp2
    assert(not(TST) and STS)
    
def test_valid_distance_metric():
    cfg = config.TrainingConfig(
            autoencoder_name="TAE",
            learning_rate=0.0001,
            latent_dim=20,
            batch_size=4,
            use_openai=False,
            is_notebook=True,
    )
    trainer = training.DeepDreamLLMTrainer(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t1 = torch.randn((3,20,768), device=device)
    t2 = t1.clone()
    assert trainer.model_embed_loss(t1,t2) < 1e-4

if __name__ == "__main__":
    test_injective_tokenizer()
    test_valid_distance_metric()
    