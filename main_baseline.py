#!/bin/python3

import pandas as pd
import json
import numpy as np
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils

from utils.baseline import BaseProbabilities

def parse_args():
    parser = argparse.ArgumentParser(description="Create data for baseline")

    # add arguments
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('--datafile', type=str, help="Path + name to formatted data file with sentences", required='--convert' not in sys.argv)
    requiredNamed.add_argument('--word2idxfile', type=str, help="Path + name to word2idx file", required=True)
    parser.add_argument('--convert', action='store_true', help="If specified, probabilities table is created from the datafile.")
    parser.add_argument('--file', type=str, help="Path + name to data file with sentences", required='--convert' in sys.argv)
    parser.add_argument('--window', type=int, help="Window size", required='--convert' in sys.argv)
    parser.add_argument('--direction', choices=['forward', 'backward'], type=str, help="How to create the pairs. Forward: look at what comes after the center. Backward: look at what comes before the center.", required='--convert' in sys.argv)
    parser.add_argument('--randomize', action='store_true', help="Randomizes the data if added")
    parser.add_argument('--predict', action='store_true', help="Predict data to analyze accuracy")
    parser.add_argument('--probfile', type=str, help="Path + name to formatted data file with sentences", required='--predict 'in sys.argv)
    return parser.parse_args()

args = parse_args()

# load in the word2idx file
with open(args.word2idxfile, 'r') as f:
    word2idx = json.load(f)

idx2word = {idx: w for w, idx in word2idx.items()}

if args.convert:
    print("CONVERSION")
    print("Data file:", args.file)
    print("Window:", args.window)
    print("Direction:", args.direction)
    print("Randomized:", args.randomize)
    
    testset = BaseProbabilities(direction=args.direction, window=args.window, unordered=args.randomize)
    testset.create_pairs(filename=args.file)
    testset.convert_word2idx(word2idx=args.word2idxfile)
    prob_df = testset.df
    
    if args.predict:
        datafile = testset.pair_data
else:
    print("Preformatted")
    if args.predict:
        print("Data file:", args.datafile)
        #load in formatted data
        with open(args.datafile, 'r') as f:
            datafile = json.load(f)

    # load probabilities
    print("Probability file:", args.probfile)
    prob_df = pd.read_excel(args.probfile)

def perplexity(column):
    column = column * np.log(column)

    perplexity = np.exp(-np.sum(column))

    return perplexity

print("Total Perplexity:", np.mean(prob_df.apply(perplexity, axis=0)))

if args.predict:
    vocab_size = len(prob_df.index)
    total_accuracy = 0, 0
    N = 0
    # log_every = 50000

    # criterion = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        criterion.cuda()
    print("TRAINING")
    # # ehh something
    for i, seq in enumerate(datafile):
    #     # print("{}/{}".format(i, N), end="\r")
        x = str(seq[0])
        y = seq[1]

        output = prob_df[x]
        y_pred = output.idxmax()

        y = torch.as_tensor(np.array([y]))
        output = torch.tensor(output.values).float().unsqueeze(0)

        if use_cuda:
            y = y.cuda()
            output = output.cuda()

      # accuracy
        if seq[1] == y_pred:
            total_accuracy += 1  

        N += 1

    print("Accuracy: {:.3f}".format(total_accuracy/N))

#something magical is happening