#!/usr/bin/python3

import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn as nn

import numpy as np

import argparse

from utils.data_process import Preprocess
from model.skipgram import SkipGram
from utils.funcs import plot_tSNE

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--traindata', type=str, default='proteins.train.txt')
    parser.add_argument('--testdata', type=str, default='proteins.test.txt')
    parser.add_argument('--model', type=str, help="Path + name to saved model")
    parser.add_argument('--batchsize', type=int, default=128*5)
    parser.add_argument('--tSNE', action='store_true')

    return parser.parse_args()

def test_net(dataloader, net, use_cuda, direction):
    """ Evaluates the network on the test data set
    
    Parameters
    ----------
    dataloader: data iterator
    net: deep learning network
    use_cuda: can it run on a gpu?

    Returns
    ----------
    test_loss : network loss on test set

    """
    test_loss, test_length = 0, 0

    N = len(dataloader)

    for i, (center, contexts) in enumerate(dataloader):
        print("{0}/{1}".format(i+1, N), end='\r')
        center = center.long()
        contexts = contexts.long()
        
        if use_cuda:
            center = center.cuda()
            contexts = contexts.cuda()

        output = net(center, contexts)

        loss = -output.mean(1).mean()

        test_loss += loss.item() * center.shape[0]
        test_length += center.shape[0]

    test_loss /= test_length

    return test_loss

if __name__ == '__main__':
    args = parse_args()

    model_name = args.model
    model_split = model_name.split('_')

    # model_name = "{0}/{1}/window_{5}_epoch_{2}_{3}_{4}_lr{6}_emb{7}_model.pkl".format(args.datadir, date, epoch+1, args.direction, args.optimizer, args.window, args.lr, args.embeddingdim)
    window = int(model_split[3])
    direction = model_split[6]
    optim_func = model_split[7]
    lr = float(model_split[8].strip('lr'))
    embedding_dim = int(model_split[9].strip('emb'))

    print("MODEL DETAILS:\nWindow size: {0}\nDirection: {1}\nOptimizer: {2} with lr={3}\nEmbedding dimension: {4}".format(window, direction, optim_func, lr, embedding_dim))

    print("PREPROCESSING DATA")
    f = open(args.datadir + '/' + args.traindata, 'r')
    preprocess_train = Preprocess(window_size=window, unk='_')    
    preprocess_train.build(file=f, direction=direction, convert=False)
    f.close()

    print("Test data")
    f = open(args.datadir + '/' + args.testdata, 'r')
    preprocess_test = Preprocess(window_size=window, unk='_')
    preprocess_test.build(file=f, word2idx=preprocess_train.word2idx, direction=direction)
    f.close()

    test_loader = data_utils.DataLoader(preprocess_test.data, batch_size=args.batchsize, shuffle=True, num_workers=4)

    print("LOADING SAVED MODEL AND OPTIMIZER")
    use_cuda = torch.cuda.is_available()    
    if use_cuda:
        checkpoint = torch.load(model_name)
    else:
        checkpoint = torch.load(model_name, map_location='cpu')

    net = SkipGram(embedding_dim=embedding_dim, vocab_size=preprocess_train.vocab_size + 1)
    
    net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if use_cuda:
        net.cuda()

    print("Model was trained in {} epoch(s)".format(checkpoint['epoch']))

    print("EVALUATING ON TEST SET")
    test_loss = test_net(dataloader=test_loader, net=net, use_cuda=use_cuda)
    print("Test loss: {:.3f}, Test perplexity: {:.3f}".format(test_loss, np.exp(test_loss)))

    if args.tSNE:
        # get words / amino acids that are unique
        words = sorted(preprocess_test.wc, key=preprocess_test.wc.get, reverse=True)
        words_array = np.array(words)

        idx2vec = net.in_embedding.weight.data.cpu().numpy()

        plot_name = "_".join(model_split[:-1]) + '_tSNE.png'
        plot_tSNE(idx2vec=idx2vec, word2idx=preprocess_test.word2idx, words=words_array, filename=plot_name)
