import argparse
import sys
import json
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model.cnn import TextCNN
from CBoW_scripts.functions import DataLoader, accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Train a convolutional neural network for sentence classification")

    # required arguments
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('--train_file', type=str, required=True, help = "Path+name to file with training data")
    requiredNamed.add_argument('--valid_file', type=str, required=True, help="Path+name to file with validation data")
    # requiredNamed.add_argument('--num_classes', type=int, required=True)
    requiredNamed.add_argument('--in_channels', type=int, required=True)
    requiredNamed.add_argument('--out_channels', type=int, required=True)
    requiredNamed.add_argument('--kernel_sizes', nargs='+', type=int, required=True)
    requiredNamed.add_argument('--strides', type=int, required=True)
    requiredNamed.add_argument('--dim_embed', type=int, required=True)
    requiredNamed.add_argument('--window', type=int, help="Window size")
    requiredNamed.add_argument('--direction',  type=str, choices=['before', 'after', 'both'], help='Direction of context (input) window.', required=True)
    requiredNamed.add_argument('--word2idx',  type=str,  help='word2idx file')

    # optional arguments
    parser.add_argument('--embedding_matrix', type=str)
    parser.add_argument('--p_dropout', type=float, default=0.5, help="dropout probability, default: 0.5")
    parser.add_argument('--emb_train', action='store_true', help="If given, embeddings of the network are also trained")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs to train network in")
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help="Optmizer to use in training")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate for optimizer")

    return parser.parse_args()

def train(train_iter, val_iter, model, epochs, optim_func, lr=0.1, cuda=False):
    
    if optim_func == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = lr)
    elif optim_func == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Optimizer: {} with lr={}".format(optim_func, lr))
    
    criterion = nn.CrossEntropyLoss()
    
    if cuda:
        print("Running on GPU")
        model.cuda()
        optimizer.cuda()
    
    print("")

    train_accs, valid_accs = [], []
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc, running_length = 0, 0, 0

        for inputs, labels in train_iter:
            n_samples = inputs.shape[0]

            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * n_samples
            running_acc += accuracy(y_true=labels, y_pred=output) * n_samples
            running_length += n_samples
        
        train_losses.append(running_loss/running_length)
        train_accs.append(running_acc/running_length)

        model.eval()
        running_loss, running_acc, running_length = 0, 0, 0
        for inputs, labels in val_iter:
            n_samples = inputs.shape[0]

            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            output = model(inputs)
            loss = criterion(output, labels)
            running_loss += loss.item() * n_samples
            running_acc += accuracy(y_true=labels, y_pred=output) * n_samples
            running_length += n_samples
        
        valid_losses.append(running_loss/running_length)
        valid_accs.append(running_acc/running_length)

        
        print("Epoch: {}".format(epoch+1))
        print(" Training loss: {:.3f}, perplexity: {:.3f}, accuracy: {:.3f}".format(train_losses[-1], np.exp(train_losses[-1]), train_accs[-1]))
        print(" Validation loss: {:.3f}, perplexity: {:.3f}, accuracy: {:.3f}".format(valid_losses[-1], np.exp(valid_losses[-1]), valid_accs[-1]))

    return train_losses, train_accs, valid_losses, valid_accs

if __name__ == "__main__":
    args = parse_args()

    # load in word2idx file
    if '.json' in args.word2idx:
        with open(args.word2idx, 'r') as f:
            word2idx = json.load(f)
    else:
        with open(args.word2idx, 'r') as f:
            word2idx = eval(f.readline())
    
    print("Window size: {0}".format(args.window))
            
    # preprocess data
    print("Loading and converting train data")
    train_data = DataLoader()
    train_data.load_corpus(path=args.train_file)
    train_data.count_corpus()
    train_data.make_context_pairs(window_size=args.window, direction=args.direction)
    train_data.words_to_index(word2idx=word2idx)

    print("Loading and converting valid data")
    valid_data = DataLoader()
    valid_data.load_corpus(path=args.valid_file)
    valid_data.count_corpus()
    valid_data.make_context_pairs(window_size=args.window, direction=args.direction)
    valid_data.words_to_index(word2idx=word2idx)

    # convert to pytorch batch loader thingy
    train_tensor = data_utils.TensorDataset(torch.from_numpy(train_data.context_array[0]), torch.from_numpy(train_data.context_array[1]))
    load_train = data_utils.DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)

    valid_tensor = data_utils.TensorDataset(torch.from_numpy(valid_data.context_array[0]), torch.from_numpy(valid_data.context_array[1]))
    load_valid = data_utils.DataLoader(valid_tensor, batch_size=args.batch_size, shuffle=True)

    # check if cuda is available
    use_cuda = torch.cuda.is_available()

    # setup network
    vocab_size = len(train_data.word_to_idx)
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

    print("CNN for sentence classification:\n", net)
    # load in embedding matrix if given
    if args.embedding_matrix is not None:
        if use_cuda:
            checkpoint = torch.load(args.embedding_matrix)
        else:
            checkpoint = torch.load(args.embedding_matrix, map_location='cpu')
        print(checkpoint['model_state_dict']['in_embedding.weight'].shape)
        print(checkpoint['model_state_dict']['in_embedding.weight'])
        net.load_embeddings(matrix=checkpoint['model_state_dict']['in_embedding.weight'], trainable=args.emb_train)
        print("Pretrained embeddings loaded")

    train_losses, train_accs, valid_losses, valid_accs = train(load_train, load_valid, net, optim_func=args.optimizer, epochs=args.epochs, lr=args.learning_rate)
