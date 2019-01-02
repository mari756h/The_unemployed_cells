#!/bin/python

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--data', type=str, default='proteins.test.txt')
    parser.add_argument('--model', type=str, help="Path + name to saved model")

    return parser.parse_args()

def plot_tSNE(X_test, directory):
    model = TSNE(n_components=2, perplexity=10, n_iter=5000, method='exact', verbose=1, learning_rate=5.0)

    words_test = sorted(preprocess_test.wc, key=preprocess_test.wc.get, reverse=True)
    words_test_array = np.array(words_test)
    
    X_test=model.fit_transform(X_test)

    plt.figure(figsize=(10,10))

    plt.legend()
    plt.savefig(directory + '/')