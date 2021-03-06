import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from model.skipgram import SkipGram
from model.cbow import cbow
import argparse
from Bio.SubsMat import MatrixInfo
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--idx2word', type=str, help="Path + name to idx2word file", default='data/sg_idx2word.txt')
    parser.add_argument('--word2idx', type=str, help="Path + name to word2idx file", default='data/sg_word2idx.txt')
    parser.add_argument('--file', type=str, help="Path + name to file containing analogies to test")
    parser.add_argument('--model_type', type=str, help = "Skip-gram (SG) or continouos bag of words (CBOW) model?", choices=['CBOW', 'SG'])
    parser.add_argument('--model', type=str, help="Path + name to saved model")
    parser.add_argument('--emb_dim', type=int, default=2, help="Embedding dimension")
    parser.add_argument('--vocab_size', type=int, default=22, help="size of vocabulary")
    parser.add_argument('--verbose', action='store_true', help="Print results")
    parser.add_argument('--blosum62', action='store_true', help="Compare with blosum62 scores (requires biopython)")

    return parser.parse_args()

def most_similar(idx2vec, idx2word, w1, k=1, verbose=True, output=False):
    """ Find the most similar word to w1 in the embedding space
    
    Parameters
    ----------
    idx2vec: array or matrix
        map word index to embedding vector
    idx2word: list
        map word index to word 
    w1: str
        input word
    k: integer, default: 1
        take the k most similar words
    verbose: boolean, default: True
        print output inside function
    output: boolean, default: False
        if true, return indexes and probabilities of the k most similar words

    Returns
    ----------
    idx: tensor 
        indexes for the most similar words to w1
    prob: tensor
        probability of word indexes

    """
    # get embeddings for w1
    w1_emb = idx2vec[w1]

    # find most similar word to w1
    dist = torch.matmul(idx2vec, w1_emb)
    prob, idx = torch.topk(dist, k, dim=-1)
    
    if verbose:
        if k == 1:
            print("Most similar word to {}: {} with probability {:.3f}".format(idx2word[w1], idx2word[idx.item()], prob.item()))
        else:
            print("Top {0} similar words to {1}".format(k, idx2word[w1]))

            for i in range(k):
                print(idx.data[i], "with probability", prob.data[i])

    if output:
        return prob, idx

def compare_blosum62(idx2vec, idx2word, w1, k=3, verbose=True, unk='_'):
    """ Find the k most similar words in two methods: word embedding and BLOSUM62 matrix [1]

    Parameters
    ----------
    idx2vec: vector or tensor
        map word index to embedding vector
    idx2word: list
        map word index to word 
    w1: str
        input word
    k: int, default: 3
        take the k most similar words
    verbose: boolean, default: True
        print output inside function if True
    unk: str, default: '_'
        unknown or padding symbol

    Returns
    ----------
    correct: int
        how many of the k most similar words were found in both methods
    ns: int
        number of total word pairs being compared

    References
    ----------
    1. Qi Y, Oja M, Weston J, Noble WS. A unified multitask architecture for predicting local protein properties. PLoS One. 2012;7(3). doi:10.1371/journal.pone.0032235.

    """
    blosum62 = MatrixInfo.blosum62
    
    _, pred_idxs = most_similar(idx2vec, idx2word, w1, k, verbose=False, output=True)

    if verbose:
        print("\nTop {0} closest words to {1}".format(k, idx2word[w1]))
    for pred_idx in pred_idxs:
        pred_idx = pred_idx.item()
        if unk not in [idx2word[w1], idx2word[pred_idx]]:
            # print("Predicted most similar to {0}: {1}".format(idx2word[w1], idx2word[pred_idx]))

            pair_pred = (idx2word[w1], idx2word[pred_idx])
            if pair_pred not in blosum62.keys():
                pair_pred = (idx2word[pred_idx], idx2word[w1])
            if verbose:
                print("Blosum62 score for {0}: {1}".format(pair_pred, blosum62[pair_pred]))

    scores = {pair: blosum62[pair] for pair in blosum62.keys() if idx2word[w1] in pair}
    highest = sorted(scores, key=scores.get, reverse=True)[:k]
    
    if verbose:
        print("Top {0} blosum62 scores with {1}".format(k, idx2word[w1]))
    correct, ns = 0, 0
    for a in highest:
        if verbose:
            print("{0}: {1}".format(a, blosum62[a]))

        if pair_pred in highest:
            correct += 1
        ns += 1
    
    return correct, ns

def eval_analogies(idx2vec, idx2word, w1, w2, w3, w4, verbose=False):
    """ Evaluate the analogical reasoning task: w1 is to w2, as w3 is to ? (w4) [1]

    Parameters
    ----------
    idx2vec: tensor or array
        map word index to embedding vector
    idx2word: list
        map word index to word 
    w1, w2, w3: str
        input words
    w4: str
        the true output word
    verbose: boolean, default: False
        print output inside function if True

    Returns
    ----------
    0 or 1 whether w4 was correctly predicted

    References
    ----------
    1. Le Q, Mikolov T. Distributed Representations of Sentences and Documents. http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=EEA503E2812074C19D96737C5EE51176?doi=10.1.1.646.3937&rep=rep1&type=pdf. Set december 27, 2018.

    """
    
    # get embeddings for w1, w2 and w3
    w1_emb = idx2vec[w1]
    w2_emb = idx2vec[w2]
    w3_emb = idx2vec[w3]

    # target
    target = w3_emb + (w2_emb-w1_emb)

    # find most similar word with cosine distance
    dist = torch.matmul(idx2vec, target)
    _, idx = torch.max(dist, dim=0)
    
    if verbose:
        print("Predicted analogy:")
        print("{0} is to {1}, as {2} is to {3}".format(idx2word[w1], idx2word[w2], idx2word[w3], idx2word[idx.item()]))

        print("Expected analogy:")
        print("{0} is to {1}, as {2} is to {3}".format(idx2word[w1], idx2word[w2], idx2word[w3], idx2word[w4]))
        print("")
    
    if idx.item() == w4: # correct
        return 1
    else:
        return 0

if __name__ == '__main__':
    args = parse_args()

    with open(args.idx2word, 'r') as f:
        idx2word = eval(f.readline())
    
    with open(args.word2idx, 'r') as f:
        word2idx = eval(f.readline())
    
    checkpoint = torch.load(args.model, map_location='cpu')

    if args.model_type == 'SG':
        net = SkipGram(embedding_dim=args.emb_dim, vocab_size=args.vocab_size)
        net.load_state_dict(checkpoint['model_state_dict'])
        idx2vec = net.in_embedding.weight.data.cpu()#.numpy()
    elif args.model_type == 'CBOW':    
        net = cbow(embedding_dim=args.emb_dim, vocab_size=args.vocab_size)
        net.load_state_dict(checkpoint['model_state_dict'])
        idx2vec = net.embeddings.weight.data.cpu()
    
    if args.blosum62:
        correct_blosum62, n = 0, 0
        for aa in range(len(idx2word)):
            score, ns = compare_blosum62(idx2vec, idx2word, aa, k=3, verbose=args.verbose)
            if isinstance(score, int):
                correct_blosum62 += score
                n += ns
        
        print("Blosum62 comparison with k nearest neighbors to each amino acid\nAccuracy:", correct_blosum62/n)
        print("")
    else:
        print("Similarities")
        for aa in range(len(idx2word)):
            most_similar(idx2vec, idx2word, aa)
    
    with open(args.file, 'r') as f:
        n = 0
        correct = 0
        for line in f.readlines():
            
            # skip comment lines
            if line[0] != '#':
                n += 1
                w1, w2, w3, w4 = line.strip('\n').split(' ')
                correct += eval_analogies(idx2vec, idx2word, word2idx[w1], word2idx[w2], word2idx[w3], word2idx[w4], verbose=args.verbose)
        
        print("Accuracy for analogical reasoning: {:.3f}".format(correct/n))