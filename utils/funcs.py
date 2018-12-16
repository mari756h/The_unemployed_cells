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
    #num_colors = len(words)
    #colors = cm.rainbow(np.linspace(0, 1, num_colors))

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

    # plot
    plt.figure(figsize=(10,8))
    for i, label in zip(target_ids, words):
        if label == '_': continue # skip
        plt.scatter(x[i], y[i], c=coloring_scheme[label])
        plt.annotate(label, (x[i]+0.05, y[i]+0.05))
    plt.legend(custom_legend, ['positively charged', 'negatively charged', 'polar uncharged', 'special cases', 'hydrophobic'], loc='center left', bbox_to_anchor=(1, 0.5), title='Side chain properties')
    plt.savefig(filename)
    plt.show()
