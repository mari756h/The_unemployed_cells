import argparse
import pandas as pd
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Create data for baseline")

    # add arguments
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('--file', type=str, help="Path + name to data file with sentences", required=True)
    requiredNamed.add_argument('--window', type=int, help="Window size", required=True)
    requiredNamed.add_argument('--direction', choices=['forward', 'backward'], type=str, help="How to create the pairs. Forward: look at what comes after the center. Backward: look at what comes before the center.", required=True)
    parser.add_argument('--randomize', action='store_true', help="Randomizes the data if added")
    parser.add_argument('--export', action='store_true', help="Export the tables of probabilities to a file")
    parser.add_argument('--word2idx', action='store_true', help="Convert the amino acids to indices")
    parser.add_argument('--word2idxfile', type=str, help="Path + name to word2idx file")

    return parser.parse_args()

class BaseProbabilities:
    """Create data for creating the baseline network.

    The probabilities are calculated for the amino acid pairs as given by the parameters: direction, window, and unordered.
    The window size is an integer that specifies how large the subsequence is, e.g. if window=3, we look at a subsequence with 3 amino acids in it.
    Direction specifies whether for a given amino acid (aa_t) we look at what is part of the sequence before or what comes after. 
    If the direction ='forward', we look at the amino acids after aa_t, e.g. aa_(t+1, t+2, .., t+(window-1)). 
    For direction = 'backward', we look at aa_(t-1, t-2, .., t-(window-1))
    Unordered refers to the idea that the order might specify something specific, so we can test whether or not a sequence of (aa_1, aa_2, aa_3) has a different probability than (aa_1, aa_3, aa_2) or (aa_2, aa_3, aa_1).
    If unordered is true, we do not care whether the subsequence is (aa_1, aa_2, aa_3) or e.g. (aa_1, aa_3, aa_2).

    Example:
    For an amino acid at position t, the direction is backward and the window size is 3. That means we are looking at the following sequence: 
        w1 = aa_(t-2)
        w2 = aa_(t-1)
        w3 = aa_t
    
    The probability for a subsequence (w1, w2, w3) occuring given what is the context (w1, w2):
        p = p(w1, w2, w3)/p(w1, w2)

    
    Attributes
    ----------
    direction: specifies what part of the sequence we are looking at 
    window: size of subsequence 
    unordered: whether we care about the order of the amino acids

    Methods
    ----------
    create_pairs(filename) 
        creates both the amino acid subsequences (pairs) and counts occurences to convert it to the probabilities.
    export(name)
        function to export the probability dataframe and the dataset pairs.

    convert_word2idx(word2idx)
        convert amino acid letters into indices (numbers)

    """

    def __init__(self, direction, window, unordered=False):
        self.direction = direction
        self.unordered = unordered
        self.window = window
        self.x_size = self.window-1
        self.padding = ['_'] * self.x_size

        # prob_top: p(a, b, c) if we want to predict c
        # prob_bottom: p(a, b)
        self.prob_bottom = {}
        self.prob_top = {} 

        # calculating of p(a, b, c) / p(a, b) stored in 
        # format: [x][y], if x = (a, b) and y = (c) 
        self.prob_combined = {} 
    
    def create_pairs(self, filename):
        """Function to create and count amino acid pairs (subsequences) and converts the counts into probabilities.

        Attributes
        ----------
            filename: name of file containing protein sequences
        """
        self.pair_data = []

        with open(filename, 'r') as f:
            for sequence in f.readlines():
                seq = sequence.split()

                # add padding to ends
                seq = self.padding + seq + self.padding

                for i in range(self.x_size, len(seq)-self.x_size):
                    
                    # e.g if we are looking at c, we want p(a, b, c)
                    if self.direction == 'backward':
                        x = seq[i-self.x_size: i]
                    
                    # e.g. if we look at c, we want p(c, d, e)
                    elif self.direction == 'forward':
                        x = seq[i+1:i+1+self.x_size]
                    
                    y = seq[i]

                    if self.unordered:
                        x = sorted(x)
                    
                    self.pair_data.append([x, y])
                    
                    x = "".join(x)

                    self.prob_bottom[x] = self.prob_bottom.get(x, 0) + 1
                    if x in self.prob_top:
                        self.prob_top[x][y] = self.prob_top[x].get(y, 0) + 1
                    else:
                        self.prob_top[x] = {y: 1}
                

        # check that the sum of counts are the same in both dictionaries
        assert sum(self.prob_bottom.values()) == sum([x for pairs in self.prob_top.values() for x in pairs.values()])

        # convert counts to probabilities
        N = sum(self.prob_bottom.values())
        self.prob_bottom = {x: c/N for x, c in self.prob_bottom.items()}
        for x, pairs in self.prob_top.items():
            pairs = {y: c/N for y, c in pairs.items()}
            self.prob_top[x] = pairs
        
        # division part, e.g. p(a, b, c) / p(a, b)
        for x, xval in self.prob_bottom.items():
            self.prob_combined[x] = {}
            for y, yval in self.prob_top[x].items():
                self.prob_combined[x][y] = yval/xval
        
        self.df = pd.DataFrame.from_dict(self.prob_combined)
        self.df = self.df.fillna(0)

    def export(self, name):
        """Exports data"""
        self.df.to_excel(name)

        print("Table of probabilities exported to\n {name}".format(name=name))

        if hasattr(self, "word2idx") and hasattr(self, "new_word2idx"):
            name_word2idx = name.split('_')[0] + '_word2idx.json'
            with open(name_word2idx, 'w') as f:
                json.dump(self.word2idx, f)

            print("Word2idx file exported to \n {name}".format(name=name_word2idx))

        data_file = name.strip('.xlsx') + '_data.json'
        with open(data_file, 'w') as f:
            json.dump(self.pair_data, f)
        
        print("Data file for neural network exported to \n {name}".format(name=data_file))

    
    def convert_word2idx(self, word2idx=None):
        """Converts amino acid letters into word indices (numbers)
        
        Attributes
        ----------
        word2idx: None or dictionary
            If specified, uses the given object as word2idx.
            If not specified (None), create a new word2idx dictionary.

        """
        if word2idx is not None:
            with open(word2idx, 'r') as f:
                self.word2idx = json.load(f)
        else:
            # create new word2idx
            self.word2idx = {}
            self.new_word2idx = True
            amino_acids = sorted(self.df.columns)
            for i, aa in enumerate(amino_acids):
                self.word2idx[aa] = i
                    
        self.idx2word = {i: aa for aa, i in self.word2idx.items()}        
        self.df.index = [self.word2idx[aa] for aa in self.df.index]

        new_columns = []
        for pair in self.df.columns:
            pair = [self.word2idx[aa] for aa in list(pair)]
            new_columns += [str(pair)]
        
        self.df.columns = new_columns

        for i, (x, y) in enumerate(self.pair_data):
            x = [self.word2idx[aa] for aa in x]
            y = self.word2idx[y]

            self.pair_data[i] = (x, y)

if __name__ == '__main__':
    args = parse_args()

    print("File:", args.file)
    print("Window:", args.window)
    print("Direction:", args.direction)
    print("Randomized:", args.randomize)

    testset = BaseProbabilities(direction=args.direction, window=args.window, unordered=args.randomize)
    testset.create_pairs(filename=args.file)

    if args.word2idx:
        testset.convert_word2idx(word2idx=args.word2idxfile)
    
    if args.export:
        data_dir, input_file = args.file.split('/') 
        output_name = "{dir}/baseline/{file}_baseline_window{window}_{direction}_{order}{w2i}.xlsx".format(dir=data_dir, file=input_file, window=args.window, direction=args.direction, order='unordered' if args.randomize else 'ordered', w2i='_idx' if args.word2idx else '')
        testset.export(name=output_name)