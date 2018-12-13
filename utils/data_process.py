import argparse
import os
import numpy as np
import pickle
import random

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--traindata', type=str, default='proteins.train.txt')
    parser.add_argument('--testdata', type=str, default='proteins.test.txt')
    parser.add_argument('--validdata', type=str, default='proteins.val.txt')
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--subsampling', type=bool, default=False)
    parser.add_argument('--direction', type=str, default='both', help="look forward, backward or both", choices=['forward', 'backward', 'both'])
    parser.add_argument('--save', type=bool, default=False)

    return parser.parse_args()

class Preprocess:
    def __init__(self, window_size, unk):
        self.window = window_size
        self.unk = unk
        
    def skipgram(self, sentence, i, direction):
        """Can implement directional skipgram"""
        center = sentence[i]

        left = sentence[max(0, i-self.window): i]
        right = sentence[i+1: i+self.window+1]

        padding = [self.unk] * self.window

        if len(left) < self.window:
            left = left + padding[:self.window-len(left)]
        if len(right) < self.window:
            right = padding[:self.window-len(right)] + right

        if direction == 'both':
            return center, left + right
        elif direction == 'backward':
            return center, left
        elif direction == 'forward':
            return center, right
                                
    def build(self, file, direction, subsampling=True, threshold=1e-5, word2idx=None):
        print("Creating vocab")
        
        self.sentences = []
        self.wc = {self.unk: 1}
        for i, line in enumerate(file.readlines()):
            sent = []
            for word in line.split():
                sent.append(word)
                    
                self.wc[word] = self.wc.get(word, 0) + 1
            self.sentences.append(sent)
                                
        self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)      
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        
        if word2idx is not None:
            self.word2idx = word2idx
        
        self.vocab = set([word for word in self.word2idx])
        self.vocab_size = len(self.vocab)
                
        print("Done with building vocab")
        
        if subsampling:
            self.subsampling(self.wc, threshold)
        
        self.convert(subsampling, direction)
        
    def subsampling(self, counts, threshold):
        
        N = sum(counts.values())
        
        freqs = {w: c/N for (w,c) in counts.items()}
        
        discard_table = {w:1-np.sqrt(threshold/f) for (w,f) in freqs.items()}
        
        self.discard_table = discard_table
        
    def discard(self, word_id):
        return random.random() > self.discard_table[word_id]
        
    def convert(self, subsampling, direction):

        print("Converting corpus..")
        data = []
                
        for sent in self.sentences:
            for i in range(len(sent)):
                center, contexts = self.skipgram(sent, i, direction)
                data.append((self.word2idx[center], np.array([self.word2idx[context] for context in contexts])))
            
        self.data = data

        print("Done")

if __name__ == 'main':
    args = parse_args()

    preprocessed_dir = '{2}/preprocessed/{0}_window_{1}'.format(args.window, args.direction, args.datadir)
    os.makedirs(preprocessed_dir, exist_ok=True)

    print("Train data")
    f = open(args.datadir + '/' + args.traindata, 'r')
    preprocess_train = Preprocess(window_size=args.window, unk='')
    preprocess_train.build(file=f, subsampling=args.subsampling, direction=args.direction)
    f.close()


    print("Validation data")
    f = open(args.datadir + '/' + args.validdata, 'r')
    preprocess_valid = Preprocess(window_size=args.window, unk='')
    preprocess_valid.build(file=f, subsampling=args.subsampling, direction=args.direction, word2idx=preprocess_train.word2idx)
    f.close()

    print("Test data")
    f = open(args.datadir + '/' + args.testdata, 'r')
    preprocess_test = Preprocess(window_size=args.window, unk='_')
    preprocess_test.build(file=f, subsampling=args.subsampling, direction=args.direction)#, word2idx=preprocess_train.word2idx)
    f.close()

    if args.save:
        pickle.dump(preprocess_train, open(preprocessed_dir + '/traindata.pkl', 'wb'))
        pickle.dump(preprocess_valid, open(preprocessed_dir + '/validdata.pkl', 'wb'))
        pickle.dump(preprocess_test, open(preprocessed_dir + '/testdata.pkl', 'wb'))
