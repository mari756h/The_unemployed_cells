#!/usr/bin/python3
# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils

import argparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


#########################
# # Data loading
#########################
# Data loader class
class DataLoader: 
    def __init__(self, padding='_'):
        self.corpus = []
        self.padding = '_'
    
    # Load words
    def load_corpus(self, path): 
        print('# Loading corpus...')
        with open(path, 'r') as infile: 
            for line in infile: 
                line = line[:-1].split()
                self.corpus.append(line)
        print('\tDone\n')
    
    # Make dict
    def count_corpus(self, padding=True, verbose=False): 
        print('# Building vocabulary...')
        # Count occurrences
        unique, counts = np.unique(np.array([item for sublist in self.corpus for item in sublist]), return_counts=True)
        self.corpus_counts = dict(zip(unique, counts))

        if verbose: 
            for v, k in sorted(zip(counts, unique), reverse=True): 
                print('Key is "{0}" with count {1}'.format(k, v))

        # Build vocabulary
        if padding: 
            indices = list(range(1,len(unique)+1))
            self.word_to_idx = dict(zip(sorted(unique), indices))
            self.word_to_idx[self.padding] = 0
        else: 
            indices = list(range(len(unique)))
            self.word_to_idx = dict(zip(sorted(unique), indices))
        
        # Make reverse dict
        self.idx_to_word = {w: idx for idx, w in self.word_to_idx.items()}
        
        print('\tDone\n')
            
    # Function to make context pairs
    def make_context_pairs(self, window_size=2, padding=True, direction='both'): 
        print('# Making context pairs...') 
        self.window_size = window_size
        self.direction = direction
        
        # Run through each sample
        self.word_data = []
        window_slide = {'both': [-window_size, window_size], 'before': [-window_size, 0], 'after': [0, window_size]}
        ranges = {'both': [window_size, window_size], 'before': [window_size, 0], 'after': [0, window_size]}
        for line in self.corpus: 
            if padding: 
                # Add padding corresponding to the size of the window on either side
                padding = [self.padding]*window_size

                # Set direction of padding
                if self.direction=='both': 
                    line = padding+line+padding
                elif self.direction=='before': 
                    line = padding+line
                elif self.direction=='after': 
                    line = line+padding
                else: 
                    print('Specify window direction!')
                    sys.exit(1)

                # Window ranges
                start, end = ranges[self.direction]
            else: 
                start = 0
                end = 0

            # Make contexts
            for i in range(start, len(line)-end):
                c, end_slide = window_slide[self.direction]

                # Run through window
                context = []
                while c <= end_slide:
                    if c != 0: 
                        context.append(line[i+c])
                    c += 1
                self.word_data.append((context, line[i]))
        print('\tDone\n')

    # Convert word_data to numpy array tuples
    def words_to_index(self, word2idx=None):
        if hasattr(self, 'window_size') and hasattr(self, 'direction'): 
            print('# Converting words to indices...')
            
            if word2idx is not None:
                self.word_to_idx = word2idx

            # Pre-allocate
            if self.direction == 'both': 
                columns = self.window_size*2
            else: 
                columns = self.window_size

            data = np.empty((len(self.word_data), columns), dtype=int)
            labels = np.empty((len(self.word_data)), dtype=int)
            
            # Run through context pairs and fill arrays
            i = 0
            for d, l in self.word_data: 
                data[i, :] = np.array([self.word_to_idx[w] for w in d])
                labels[i,] = self.word_to_idx[l]
 
                i += 1
                
            # Save as tuple
            self.context_array = (data, labels)
            
            # Remove word_data
            del self.word_data
            
            print('\tDone\n')
        else: 
            print('# Make context pairs, first!')
            sys.exit(1)

####################
# # CBOW class
####################
class cbow(nn.Module):

    def __init__(self, vocab_size, embedding_dim=20, padding=True):
        super(cbow, self).__init__()
        # num_embeddings is the number of words in your train, val and test set
        # embedding_dim is the dimension of the word vectors you are using
        if padding: 
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, 
                                          padding_idx=0)
        else: 
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, 
                                          padding_idx=None)
        
        self.linear_out = nn.Linear(in_features=embedding_dim, out_features=vocab_size, bias=False)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        
        # To not care about the order of the words we take the mean of the time dimension
        means = torch.mean(embeds, dim=1)
        
        # Softmax on output
        #probs = F.log_softmax(out, dim=1)
        probs = F.log_softmax(self.linear_out(means), dim=1)
        
        return probs

# Estimate performance
def accuracy(y_true, y_pred):
    # Make y_pred for the word with max probability
    _, indices = torch.max(input=y_pred, dim=1)
    
    # Check if indices match
    check = torch.eq(indices, y_true)
    
    # Estimate accuracy
    acc = check.sum().item()/len(check)
    return acc


####################
# By Hannah Martiny
####################
#import torch 
#import numpy as np
from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_tSNE(idx2vec, word2idx, words, filename):    
    """
    idx2vec: embeddings
    word2idx: convert words to their indices
    words: unique words / amino acids
    filename: destination + name of file
    """

    # initialize tSNE model
    model = TSNE(n_components=2, perplexity=10, n_iter=5000, method='exact', verbose=1, learning_rate=5.0, random_state=1)

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
                        'padding': 'white'}

    custom_legend = [plt.Line2D([0], [0], color='green', marker='o', linestyle=''), 
                    plt.Line2D([0], [0], color='red', marker='o', linestyle=''), 
                    plt.Line2D([0], [0], color='blue', marker='o', linestyle=''), 
                    plt.Line2D([0], [0], color='orange', marker='o', linestyle=''), 
                    plt.Line2D([0], [0], color='purple', marker='o', linestyle='')]

    # plot
    plt.figure(figsize=(20, 20))
    plt.rcParams.update({'font.size': 50, 'axes.edgecolor': 'black'})
    #plt.xlim(-3.5,3.3)
    #plt.ylim(-2,3.5)
    for i, label in zip(target_ids, words):
        if label == '_' or label == 'padding': continue # skip
        plt.scatter(x[i], y[i], c=coloring_scheme[label], s=500)
        plt.annotate(label, (x[i]+0.05, y[i]+0.05))
    #plt.legend(custom_legend, ['positively charged', 'negatively charged', 'polar uncharged', 'special cases', 'hydrophobic'], loc='center left', bbox_to_anchor=(1, 0.5), title='Side chain properties')
    #plt.legend(custom_legend, ['positively charged', 'negatively charged', 'polar uncharged', 'special cases', 'hydrophobic'], loc='best', title='Side chain properties')
    plt.savefig(filename, dpi=1000)

