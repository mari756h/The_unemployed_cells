#!/usr/bin/python3

import argparse
import sys
import os
import json
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import numpy as np

from model.cnn import TextCNN
from CBoW_scripts.functions import DataLoader, accuracy

def parse_args():
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required arguments')

    requiredNamed.add_argument('--datadir', type=str, required=True)
    requiredNamed.add_argument('--testdata', type=str, required=True)
    requiredNamed.add_argument('--model', type=str, help="Path + name to saved model", required=True)
    requiredNamed.add_argument('--in_channels', type=int, required=True)
    requiredNamed.add_argument('--out_channels', type=int, required=True)
    requiredNamed.add_argument('--kernel_sizes', nargs='+', type=int, required=True)
    requiredNamed.add_argument('--strides', type=int, required=True)
    requiredNamed.add_argument('--dim_embed', type=int, required=True)
    requiredNamed.add_argument('--window', type=int, help="Window size", required=True)
    requiredNamed.add_argument('--direction',  type=str, choices=['before', 'after', 'both'], help='Direction of context (input) window.', required=True)
    requiredNamed.add_argument('--word2idx',  type=str,  help='word2idx file', required=True)
    parser.add_argument('--p_dropout', type=float, default=0, help="dropout probability, default: 0.5")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate for optimizer")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")

    return parser.parse_args()

def test(test_iter, model, cuda=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss, running_acc, running_length = 0, 0, 0
    for inputs, labels in test_iter:
        n_samples = inputs.shape[0]
        if cuda:
            inputs=inputs.cuda()
            labels=labels.cuda()
        output = model(inputs)
        loss = criterion(output, labels)
        running_loss += loss.item() * n_samples
        running_acc += accuracy(y_true=labels, y_pred=output) * n_samples
        running_length += n_samples
    
    running_loss /= running_length
    running_acc /= running_length
    
    print(" Test loss: {:.3f}, perplexity: {:.3f}, accuracy: {:.3f}\n".format(running_loss, np.exp(running_loss), running_acc))

    return running_loss, running_acc

if __name__ == "__main__":
    args = parse_args()

    # load in word2idx file
    if '.json' in args.word2idx:
        with open(args.word2idx, 'r') as f:
            word2idx = json.load(f)
    else:
        with open(args.word2idx, 'r') as f:
            word2idx = eval(f.readline())

    print("Loading data")
    test_data = DataLoader()
    test_data.load_corpus(path=args.testdata)
    test_data.count_corpus()
    test_data.make_context_pairs(window_size=args.window, direction=args.direction)
    test_data.words_to_index(word2idx=word2idx)

    test_tensor = data_utils.TensorDataset(torch.from_numpy(test_data.context_array[0]), torch.from_numpy(test_data.context_array[1]))
    load_test = data_utils.DataLoader(test_tensor, batch_size=args.batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()

    vocab_size = len(test_data.word_to_idx)
    net = TextCNN(num_embed=vocab_size, 
                dim_embed=args.dim_embed, 
                num_class=vocab_size, 
                p_dropout=args.p_dropout, 
                in_channels=args.in_channels, 
                out_channels=args.out_channels, 
                kernel_sizes=args.kernel_sizes, 
                strides=args.strides)
    if use_cuda:
        net.cuda()
        checkpoint = torch.load(args.model)
    else:
        checkpoint = torch.load(args.model, map_location='cpu')

    net.load_state_dict(checkpoint['model_state_dict'])

    test(load_test, net, use_cuda)