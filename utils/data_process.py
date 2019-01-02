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
    """Data preprocessing class for building the data for Skip-gram model
    
    Attributes
    ----------
    window: window size for data
    unk: what character to use for unknown data / padding

    Methods
    ----------
    skipgram(sentence, i, direction)
        Generate center and context words
    build(file, direction, subsampling=True, threshold=1e-5, word2idx=None, convert=True)
        Creates the vocabulary and data for the Skip-gram model
    subsampling(counts, threshold)
        Implements subsampling method to remove word that are very frequent.
    discard(word_id)
        Subsample a word with a probability given from subsampling(counts, threshold)
    convert(direction)
        converts center and contexts into indices (numerical values)
    """

    def __init__(self, window_size, unk):
        self.window = window_size
        self.unk = unk
        
    def skipgram(self, sentence, i, direction):
        """Class method for creating directional skip-gram data.
        
        Parameters
        ----------
        sentence: list of words in a sentence
        i: position in current sentence
        direction: specifies which direction you are looking in at the data (backward, forward or both)
        
        Return
        ----------
        center: word at the center position i
        contexts: words around the center

        """
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
                                
    def build(self, file, direction, subsampling=True, threshold=1e-5, word2idx=None, convert=True):
        """Class method for building vocabulary and data of center and context words.
        
        Parameters
        ----------
        file: txt file with sentences
        direction: specifies which direction you are looking in at the data (backward, forward or both)
        subsampling: that specifies whether the data should be subsampled. Default: False
        threshold: threshold for subsampling. Default: 1e-5
        word2idx: use a specified word2idx or create new. Default: None (create a new word2idx based on input file)
        convert: convert data (boolean). Default: True
        """

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
        if convert:
            self.convert(subsampling, direction)
        
    def subsampling(self, counts, threshold):
        """Class method for implementing subsampling.
        
        Parameters
        ----------
        counts: word counts
        threshold: position in current sentence
        
        Return
        ----------
        discard_table: gives probability of word being discarded
        """
        
        N = sum(counts.values())
        
        freqs = {w: c/N for (w,c) in counts.items()}
        
        discard_table = {w:1-np.sqrt(threshold/f) for (w,f) in freqs.items()}
        
        self.discard_table = discard_table
        
    def discard(self, word_id):
        """Class method for randomly discarding a word.
        
        Parameters
        ----------
        word_id: word index

        Return
        ----------
        a boolean value, telling whether or not the word has randomly been subsampled
        """
        
        return random.random() > self.discard_table[word_id]
        
    def convert(self, direction):
        """Class method for converting amino acid into indexes.
        
        Parameters
        ----------
        direction: specifies which direction you are looking in at the data (backward, forward or both)
        """
        print("Converting corpus..")
        data = []
                
        for sent in self.sentences:
            for i in range(len(sent)):
                center, contexts = self.skipgram(sent, i, direction)
                data.append((self.word2idx[center], np.array([self.word2idx[context] for context in contexts])))
            
        self.data = data

        print("Done")

if __name__ == '__main__':
    args = parse_args()

    preprocessed_dir = '{2}/preprocessed/{0}_window_{1}'.format(args.window, args.direction, args.datadir)
    os.makedirs(preprocessed_dir, exist_ok=True)

    print("Train data")
    f = open(args.datadir + '/' + args.traindata, 'r')
    preprocess_train = Preprocess(window_size=args.window, unk='_')
    preprocess_train.build(file=f, subsampling=args.subsampling, direction=args.direction)
    f.close()


    print("Validation data")
    f = open(args.datadir + '/' + args.validdata, 'r')
    preprocess_valid = Preprocess(window_size=args.window, unk='_')
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
