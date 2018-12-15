import torch 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_tSNE(idx2vec, word2idx, words, filename):    
    """
    idx2vec: embeddings
    word2idx: convert words to their indices
    words: unique words / amino acids
    filename: destination + name of file
    """

    # initialize tSNE model
    model = TSNE(n_components=2, perplexity=10, n_iter=5000, method='exact', verbose=1, learning_rate=5.0)

    # get unique words
    # words = sorted(preprocess_test.wc, key=preprocess_test.wc.get, reverse=True)
    # words_array = np.array(words)
    
    # transform
    X = np.array([idx2vec[word2idx[word]] for word in words])
    X_fit=model.fit_transform(X)

    target_ids = range(len(words))

    x=X_fit[:,0]
    y=X_fit[:,1]

    # set colors
    num_colors = len(words)
    colors = cm.rainbow(np.linspace(0, 1, num_colors))

    # plot
    plt.figure(figsize=(10,10))
    for i, c, label in zip(target_ids, colors, words):
        plt.scatter(x[i], y[i], c=c, label=label)
    plt.legend()
    plt.savefig(filename)
    plt.show()