import argparse
from collections import Counter
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate perplexity from word frequencies")

    # add arguments
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('--datafile', type=str, help="Path + name to formatted data file with sentences", required=True)
    parser.add_argument('--formatted', action='store_true', help="Add if the file is a table of probabilities")
    return parser.parse_args()

def probabilities_calc(file):

    with open(file, 'r') as f:
        text = f.read().splitlines()
    
    seqs = [line.split() for line in text]
    
    
    counts = Counter(x for xs in seqs for x in xs)
    N = sum(counts.values())
    counts = {aa: c/N for aa, c in counts.items()}

    freqs = np.array(list(counts.values()))

    return freqs

def parse_prob_file(file):
    df = pd.read_excel(file)

    freqs = np.array(df.values)
    return freqs

def perplexity(probabilities):    
    entropy = np.sum([pp*np.log2(1.0/pp) for pp in probabilities])

    perp = np.power(2, entropy)

    return entropy, perp

if __name__ == "__main__":    
    args = parse_args()
    if args.formatted:
        aa_freqs = parse_prob_file(file=args.datafile)
    else:
        aa_freqs = probabilities_calc(file=args.datafile)
    entropy, perp = perplexity(aa_freqs)

    print("Entropy: {:.3f}, Perplexity: {:.3f}".format(entropy, perp))


