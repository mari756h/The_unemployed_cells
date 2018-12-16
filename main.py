#!/usr/bin/python3

import numpy as np
from utils.data_process import Preprocess
from model.skipgram import SkipGram
from test_network import test_net
from utils.funcs import plot_tSNE

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.functional as F
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import show, export_png

import datetime
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--traindata', type=str, default='proteins.train.txt')
    parser.add_argument('--testdata', type=str, default='proteins.test.txt')
    parser.add_argument('--validdata', type=str, default='proteins.val.txt')
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--direction', type=str, default='both', help="look forward, backward or both", choices=['forward', 'backward', 'both'])
    parser.add_argument('--embeddingdim', type=int, default=50)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('--batchsize', type=int, default=128*5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--subsampling', type=bool, default=False)
    parser.add_argument('--logevery', type=int, default=10000)
    parser.add_argument('--negatives', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help="implemented optimizers")
    parser.add_argument('--momentum', type=float, default=0, help="used if optimizer is SGD")
    parser.add_argument('--testnetwork', action='store_true')
    parser.add_argument('--tSNE', action='store_true')
    parser.add_argument('--resume', action='store_true', help="Resume training if added and has been done before")

    return parser.parse_args()

args = parse_args()

# get preprocessed data
preprocesseddir = args.datadir + '/preprocessed/{0}_window_{1}/'.format(args.window, args.direction)
dir_exist = os.path.isdir(preprocesseddir)
unk = '_'

print("PROCESSING DATA")
print("Train data...")
f = open(args.datadir + '/' + args.traindata, 'r')
preprocess_train = Preprocess(window_size=args.window, unk=unk)    
preprocess_train.build(file=f, subsampling=args.subsampling, direction=args.direction)
f.close()

print("Valid data...")
f = open(args.datadir + '/' + args.validdata, 'r')
preprocess_valid = Preprocess(window_size=args.window, unk=unk)
preprocess_valid.build(file=f, subsampling=args.subsampling, word2idx=preprocess_train.word2idx, direction=args.direction)
f.close()

print("NETWORK SETUP")
net = SkipGram(embedding_dim=args.embeddingdim, vocab_size=preprocess_train.vocab_size + 1, n_negs=args.negatives)
print(net)

if args.optimizer == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

criterion = nn.BCEWithLogitsLoss()

train_loader = data_utils.DataLoader(preprocess_train.data, batch_size=args.batchsize, shuffle=True, num_workers=4)
valid_loader = data_utils.DataLoader(preprocess_valid.data, batch_size=args.batchsize, shuffle=True, num_workers=4)

date = datetime.datetime.now().strftime("%Y_%m_%d")
path = args.datadir + '/' + date
os.makedirs(path, exist_ok=True )
use_cuda = torch.cuda.is_available()

if args.resume:
    print("RESUMING TRAINING")
    model_match = "window_{4}_epoch_._{2}_{3}_lr{5}_emb{6}_model.pkl".format(args.datadir, date, args.direction, args.optimizer, args.window, args.lr, args.embeddingdim)

    import re
    regex = re.compile(model_match)
    files = os.listdir(path)
    matches = list(filter(regex.findall, files))
    if len(matches) > 0:
        max_epoch = max([int(x.split('_')[3]) for x in matches])
        print("Maximum epoch previously run:", max_epoch, "\n")

        model_name = model_match.split('.')[0] + str(max_epoch) +  ".".join(model_match.split('.')[1:])

        if use_cuda:
            checkpoint = torch.load(model_name)
        else:
            checkpoint = torch.load(model_name, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("No model of these specifications has been run before. Starting a new model instead.\n")

## training 
print("Num epochs:", args.epochs)
print("Direction:", args.direction)
print("Optimizer:", args.optimizer)
print("Learning rate:", args.lr)
print("Window size:", args.window)

if use_cuda:
    print("Cuda available")
    net.cuda()
print("")

valid_loss = []
train_loss = []
log_every = args.logevery
print("TRAINING")
for epoch in range(args.epochs):
    running_loss, running_length = 0, 0
    
    net.train()
    for i, (center, contexts) in enumerate(train_loader):
        
        center = center.long()
        contexts = contexts.long()
        
        if use_cuda:
            center = center.cuda()
            contexts = contexts.cuda()
        
        output = net(center, contexts)
        loss = criterion(output.float(), contexts.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * center.shape[0]
        running_length += center.shape[0]

        if i % log_every == 0:
            print("Epoch {0}, iteration {1}, train loss: {2}, train perplexity {3}".format(epoch+1, i+1, running_loss/running_length, np.exp(running_loss/running_length)))
    
    train_loss.append(running_loss/running_length)

    net.eval()
    running_loss, running_length = 0, 0 
    
    for center, context in valid_loader:
        center = center.long()
        context = context.long()
        
        if use_cuda:
            center = center.cuda()
            context = context.cuda()
        
        predictions = net(center, context)
        loss = criterion(predictions.float(), context.float())
        
        running_loss += loss.item() * center.shape[0]
        running_length += center.shape[0]
                
    valid_loss.append(running_loss/running_length)

    print("\nEpoch {0}, training loss: {1}, training perplexity: {2}".format(epoch+1, train_loss[-1], np.exp(train_loss[-1])))
    print("Epoch {0}, valid loss: {1}, valid perplexity: {2}\n".format(epoch+1, valid_loss[-1], np.exp(valid_loss[-1])))

    # save model
    model_name = "{0}/{1}/window_{5}_epoch_{2}_{3}_{4}_lr{6}_emb{7}_model.pkl".format(args.datadir, date, epoch+1, args.direction, args.optimizer, args.window, args.lr, args.embeddingdim)
    torch.save({'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch+1, 'train_loss': train_loss[-1], 'valid_loss': valid_loss[-1]}, model_name)

# plot
runned_epochs = list(range(0, len(train_loss)))

train_loss = ColumnDataSource(data=dict(epochs=runned_epochs, loss=train_loss))
valid_loss = ColumnDataSource(data=dict(epochs=runned_epochs, loss=valid_loss))

p = figure(plot_width=400, plot_height=400, title = "Losses")
p.line(x='epochs', y='loss', color='blue', alpha=0.7, source=train_loss, legend='Training')
p.line(x='epochs', y='loss', color='red', alpha=0.7, source=valid_loss, legend='Validation')

tooltips = [("epoch", "@epochs"), ("loss", "@loss")]

p.add_tools(HoverTool(tooltips=tooltips))

output_file(args.datadir + '/' + date + '/{0}_{1}_{2}_training_losses.html'.format(args.window, args.direction, args.optimizer))
save(p)

if args.testnetwork:
    print("EVALUATING ON TEST SET")
    print("Loading test data..")
    f = open(args.datadir + '/' + args.testdata, 'r')
    preprocess_test = Preprocess(window_size=args.window, unk=unk)
    preprocess_test.build(file=f, subsampling=args.subsampling, word2idx=preprocess_train.word2idx, direction=args.direction)
    f.close()

    test_loader = data_utils.DataLoader(preprocess_test.data, batch_size=args.batchsize, shuffle=True, num_workers=4)

    print("Evaluating..")
    test_loss = test_net(dataloader=test_loader, net=net, criterion=criterion, use_cuda=use_cuda)
    print("Test loss: {:.3f}, Test perplexity: {:.3f}".format(test_loss, np.exp(test_loss)))

    if args.tSNE:
        # get words / amino acids that are unique
        words = sorted(preprocess_test.wc, key=preprocess_test.wc.get, reverse=True)
        words_array = np.array(words)

        # get learned embeddings
        idx2vec = net.in_embedding.weight.data.cpu().numpy()

        # plot it!
        model_split = model_name.split('_')
        plot_name = "_".join(model_split[:-1]) + '_tSNE.png'
        plot_tSNE(idx2vec=idx2vec, word2idx=preprocess_test.word2idx, words=words_array, filename=plot_name)