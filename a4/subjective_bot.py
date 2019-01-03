"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""
import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
spacy.load('en')

torch.manual_seed(0)
import random
random.seed(0)

import argparse
import os
nlp = spacy.load('en')
text = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
train, val, test = data.TabularDataset.splits(
    path='./data/', train='train.tsv',
    validation='validation.tsv', test='test.tsv', format='tsv',
    fields=[('', None), ('text', text)], skip_header=True)

rnn = torch.load("model_rnn.pt")
cnn = torch.load('model_cnn.pt')
baseline = torch.load('model_baseline.pt')

text.build_vocab(train, vectors="glove.6B.100d")
vocab = text.vocab


def tokenizer(text): # create a tokenizer function
    tokenlist = []
    for token in nlp.tokenizer(text):
        tokenlist.append(token.text)
    return tokenlist

while True:
    myinput = input("Enter a sentence: ")

    tokens = tokenizer(myinput)
    indices = torch.tensor([vocab.stoi[tok] for tok in tokens]).unsqueeze(dim=0)
    rnn_predict = rnn(indices, [len(indices)]).squeeze()
    cnn_pred = cnn(indices, [len(indices)]).squeeze()
    baseline_p = baseline(indices, [len(indices)]).squeeze()
    print("Model rnn:{} ({})".format("subjective" if rnn_predict > 0.5 else "objective", round(float(rnn_predict),3)))
    print("Model cnn:{} ({})".format("subjective" if cnn_pred > 0.5 else "objective", round(float(cnn_pred),3)))
    print("Model baseline:{} ({})".format("subjective" if baseline_p > 0.5 else "objective", round(float(baseline_p),3)))




