import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from model.skipgram import SkipGram
from model.cbow import cbow
from utils.data_process import Preprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--idx2word', type=str, help="Path + name to idx2word file", default='data/sg_idx2word.txt')
    parser.add_argument('--word2idx', type=str, help="Path + name to word2idx file", default='data/sg_word2idx.txt')
    parser.add_argument('--file', type=str, help="Path + name to file containing analogies to test")
    parser.add_argument('--model_type', type=str, help = "Skip-gram (SG) or continouos bag of words (CBOW) model?", choices=['CBOW', 'SG'])
    parser.add_argument('--model', type=str, help="Path + name to saved model")
    parser.add_argument('--emb_dim', type=int, default=2, help="Embedding dimension")
    parser.add_argument('--vocab_size', type=int, default=22, help="size of vocabulary")
    parser.add_argument('--verbose', action='store_true', help="Print analogies")

    return parser.parse_args()

def most_similar(idx2vec, idx2word, w1):
    # get embeddings for w1
    w1_emb = idx2vec[w1]

    # find most similar word to w1
    dist = torch.matmul(idx2vec, w1_emb)
    prob, idx = torch.max(dist, dim=-1)

    print("Most similar word to {}: {} with probability {:.3f}".format(idx2word[w1], idx2word[idx], prob))
    
def eval_analogies(idx2vec, idx2word, w1, w2, w3, w4, verbose=False):
    
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
        print("{0} is to {1}, as {2} is to {3}".format(idx2word[w1], idx2word[w2], idx2word[w3], idx2word[idx]))

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
    
    similar = {}
    for aa in range(len(idx2word)):
        most_similar(idx2vec, idx2word, aa)

    print("")
    with open(args.file, 'r') as f:
        n = 0
        correct = 0
        for line in f.readlines():
            
            # skip comment lines
            if line.startswith('#'):
                continue
            
            n += 1
            w1, w2, w3, w4 = line.strip('\n').split(' ')
            correct += eval_analogies(idx2vec, idx2word, word2idx[w1], word2idx[w2], word2idx[w3], word2idx[w4], verbose=args.verbose)
        
        print("Accuracy: {:.3f}".format(correct/n))