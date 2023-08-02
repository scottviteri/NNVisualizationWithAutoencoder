"""
A file for tests to ensure code works as expect

TODO This file is a Work in progress
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

from autoencoder import *
from run_experiment import *
from training import *
from utils import *

# def test_calc_loss():
#     assert calc_loss("there", "there") == 0.0
#     assert calc_loss("hello", "there") > 0.0
#     try:
#         calc_loss("hello there", "there")
#         print("THis doesn't faile which is weird")
#     except:
#         print("success")

def test_injective_tokenizer(tokens=None, verbose=False):
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

if __name__ == "__main__":
    ones = np.ones((10,), dtype=np.int32)
    TST, STS = test_injective_tokenizer(ones)
    for i in tqdm(range(100)):
        tmp1, tmp2 = test_injective_tokenizer()
        TST = TST and tmp1
        STS = STS and tmp2
    print(f"TST: {TST}\nSTS: {STS}")
    