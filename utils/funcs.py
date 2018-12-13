import torch 
import numpy as np

def accuracy(ys, ts):
    """Calculate accuracy of model given prediction and true values"""
    
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(torch.max(ys, 1)[1], ts)
    
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())

def create_negative_table(frequencies, table_length, alpha=0.75):
    """Function for negative sampling"""

    # sum frequencies
    sums = np.sum(np.power(frequencies.values(), alpha))
    
    # init table to put values in
    neg_table = np.zeros(table_length, dtype=np.int32)

    start_index = 0

    for idx, freq in frequencies.items():
        # calculate power of frequency (P(w_i)^alpha)
        power_freq = np.power(freq, alpha)
        
        # the number of times a word’s index appears in the table is given by 
        end_index = start_index + int(power_freq/sums * table_length) + 1

        # fill this table with the index of each word in the vocabulary multiple times
        neg_table[start_index:end_index] = idx

        start_index = end_index

    return neg_table
