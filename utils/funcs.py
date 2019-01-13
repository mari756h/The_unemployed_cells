#!/usr/bin/python3

import torch 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tSNE(idx2vec, word2idx, words, filename):    
    """ Creates the t-SNE plot with the embeddings in 2 dimensions

    Parameters
    ----------
    idx2vec: array
        embeddings
    word2idx: dict
        convert words to their indices
    words: list
         unique words / amino acids
    filename: str
        destination + name of file

    Returns
    ----------
    the plot
    """

    # initialize tSNE model
    model = TSNE(n_components=2, perplexity=10, n_iter=5000, method='exact', verbose=1, learning_rate=5.0, random_state=1)
    
    # transform
    X = np.array([idx2vec[word2idx[word]] for word in words])
    X_fit=model.fit_transform(X)

    target_ids = range(len(words))

    x=X_fit[:,0]
    y=X_fit[:,1]

    # set colors
    coloring_scheme = {'R': 'green', 'H': 'green', 'K': 'green', #positively charged side chains 
                        'D': 'red', 'E': 'red',  #negatively charges side chains
                        'S': 'blue', 'T': 'blue', 'N': 'blue', 'Q': 'blue', #polar uncharged side chains
                        'C': 'orange', 'U': 'orange', 'G': 'orange', 'P': 'orange', # special cases
                        'A': 'purple', 'V': 'purple', 'I': 'purple', 'L': 'purple', 'M': 'purple', 'F': 'purple', 'Y': 'purple', 'W': 'purple', #hydrophobic side chains
                        '_': 'white'}

    custom_legend = [plt.Line2D([0], [0], color='green', marker='o', linestyle=''), 
                    plt.Line2D([0], [0], color='red', marker='o', linestyle=''), 
                    plt.Line2D([0], [0], color='blue', marker='o', linestyle=''), 
                    plt.Line2D([0], [0], color='orange', marker='o', linestyle=''), 
                    plt.Line2D([0], [0], color='purple', marker='o', linestyle='')]
    
    # # plot
    plt.figure(figsize=(20, 20))
    plt.rcParams.update({'font.size': 50, 'axes.edgecolor': 'black'})
    for i, label in zip(target_ids, words):
        if label == '_' or label == 'padding': continue # skip
        plt.scatter(x[i], y[i], c=coloring_scheme[label], s=500)
        plt.annotate(label, (x[i]+0.05, y[i]+0.05))
    
    plt.legend(custom_legend, ['positively charged', 'negatively charged', 'polar uncharged', 'special cases', 'hydrophobic'], loc='best', title='Side chain properties', bbox_to_anchor=(1, 0.5))
    # plt.xlim(-3.5, 3.3)
    # plt.ylim(-2, 3.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)
