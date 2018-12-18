#!/usr/bin/python3

# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils

import argparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from functions import *


#########################
#       Parser
#########################
def get_args():
    parser = argparse.ArgumentParser(description='Module will run continuous bag-of-words on amino acid sequences.')
    parser.add_argument('-d', required=False, dest='window_direction', type=str, help='Direction of window, can be both, before or after.')
    parser.add_argument('-pad', required=False, dest='padding', action='store_true', help='Whether to use padding on not.')
    parser.add_argument('-ws', required=False, dest='window_size', type=int, help='Size of the sliding window.')
    parser.add_argument('-b', required=False, dest='batch_size', type=int, help='Size of neural network batches.')
    parser.add_argument('-f', required=True, dest='post_fix', type=str, help='Post_fix for file names.')
    parser.add_argument('-e', required=False, dest='epochs', type=int, help='Number of epochs.')
    parser.add_argument('-embed', required=False, dest='embedding_dim', type=int, help='Number of embedding dimensions.')
    parser.add_argument('-r', required=False, dest='resume', type=str, help='Filename for saved checkpoint.')
    parser.add_argument('-test', required=False, dest='test', action='store_true', help='If this is to test model.')
    parser.add_argument('-tsne', required=False, dest='tsne', action='store_true', help='If the test makes t-SNE plot.')

    
    # Set defaults
    parser.set_defaults(window_direction='both', padding=True, window_size=3, batch_size=128, epochs=10, resume=False, embedding_dim=100, test=False)
    return parser


########################################################################
#       MAIN FUNCTION
########################################################################
def main(args): 

    # General parameters 
    d_ws = args.window_direction    # Direction of window
    pad = args.padding              # Padding
    ws = args.window_size           # Window_size
    bs = args.batch_size            # Batch size

    ##########################
    # ## Training data
    ##########################

    # Get training data
    train_data = DataLoader()
    train_data.load_corpus(path='data/proteins.train.txt')
    train_data.count_corpus(padding=pad)

    # Make context pairs for training data
    train_data.make_context_pairs(window_size=ws, padding=pad, direction=d_ws)

    # Check word contexts
    word_sum = len(train_data.word_data)
    print(word_sum)
    for context, word in train_data.word_data[:10]: 
        print(context, word)

    # Convert to numpy
    train_data.words_to_index(word2idx=train_data.word_to_idx)

    # Save word2idx
    torch.save(train_data.word_to_idx, 'word2idx.pt')


    ############################
    # ## Validation data
    ############################
    # Get validation data
    valid_data = DataLoader()
    valid_data.load_corpus(path='data/proteins.val.txt')

    # Make context pairs for validation data
    valid_data.make_context_pairs(window_size=ws, padding=pad, direction=d_ws)

    # Convert to numpy
    valid_data.words_to_index(word2idx=train_data.word_to_idx)

    # After data has been loaded it is good to check what is looks like. 
    print('Number of training samples:\t', train_data.context_array[0].shape)
    print('Number of validation samples:\t', valid_data.context_array[0].shape)

    # Set up to use GPU if available
    use_cuda = torch.cuda.is_available()
    use_cuda


    #########################
    # # Mini batches
    #########################
    # Pytorch batch_loader
    train = data_utils.TensorDataset(torch.from_numpy(train_data.context_array[0]), torch.from_numpy(train_data.context_array[1]))
    load_train = data_utils.DataLoader(train, batch_size=bs, shuffle=True)

    valid = data_utils.TensorDataset(torch.from_numpy(valid_data.context_array[0]), torch.from_numpy(valid_data.context_array[1]))
    load_valid = data_utils.DataLoader(valid, batch_size=bs, shuffle=True)


    ###########################
    # # Model parameters
    ###########################
    # Set loss, model and optimizer
    criterion = nn.CrossEntropyLoss()
    net = cbow(vocab_size=len(train_data.word_to_idx), embedding_dim=args.embedding_dim, padding=pad)
    #optimizer = optim.SGD(net.parameters(), lr=0.01)
    #optimizer = optim.Adam(net.parameters(), lr=0.01)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)

    # If GPU is available
    if use_cuda:
        print('# Converting network to cuda-enabled')
        net.cuda()
    print(net)

    # Log file
    if os.path.exists('results/log_{}.txt'.format(args.post_fix)): 
        log_file = open('results/log_{}.txt'.format(args.post_fix), 'a')
    else: 
        log_file = open('results/log_{}.txt'.format(args.post_fix), 'w')
        log_file.write('epoch\tset\tloss\tperp\tacc\n')

    # Resume training from checkpoint
    if args.resume or args.test: 
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['train_loss']
        begin = epoch+1
    else: 
        begin = 0



    #########################
    # # Model test
    #########################
    if args.test: 
        # Get test data
        test_data = DataLoader()
        test_data.load_corpus(path='data/proteins.test.txt')

        # Make context pairs for validation data
        test_data.make_context_pairs(window_size=ws, padding=pad, direction=d_ws)

        # Convert to numpy
        test_data.words_to_index(word2idx=train_data.word_to_idx)

        # After data has been loaded it is good to check what is looks like. 
        print('Number of test samples:\t', test_data.context_array[0].shape)


        # Make batches
        test = data_utils.TensorDataset(torch.from_numpy(test_data.context_array[0]), torch.from_numpy(test_data.context_array[1]))
        load_test = data_utils.DataLoader(test, batch_size=bs, shuffle=True)

        # Run model on test set
        test_acc, test_loss = [], []
        
        ### Evaluation ###
        net.eval()
        
        test_preds, test_targs = [], []
        test_losses, test_accs, test_lengths = 0, 0, 0
        examples, n_examples = [], 5

        for i, (inputs, labels) in enumerate(load_test):
            n_samples = inputs.shape[0]
          
            # Convert targets and input to cuda if available
            if use_cuda: 
              inputs = inputs.cuda()
              labels = labels.cuda()
            
            # Get predictions
            output = net(inputs)
            preds = torch.max(input=output, dim=1)[1]
            
            if use_cuda: 
               preds = preds.data.cpu().numpy()
            else: 
               preds = preds.data.numpy()
            
            # Calculate validation loss
            test_losses += criterion(output, labels).item() * n_samples
            test_accs += accuracy(y_true=labels, y_pred=output) * n_samples
            test_lengths += n_samples
            
            # Save predictions and labels
            test_preds += preds.tolist()
            test_targs += labels.tolist()
            
            # Save example inputs
            if len(examples) < n_examples: 
                for n in range(n_examples):
                  examples.append([inputs[n], labels[n].item(), preds[n].item()])
        
        # Show results of evaluation
        print('# Epoch %2i, TEST: loss=%f, perp=%f, acc=%f\n' % (epoch+1, test_losses/test_lengths, np.exp(test_losses/test_lengths), test_accs/test_lengths))

        # Write to log file
        log_file.write(str(epoch+1)+'\ttest\t'+str(test_losses/test_lengths)+'\t'+str(np.exp(test_losses/test_lengths))+'\t'+str(test_accs/test_lengths)+'\n')

        # Show top N validation samples and their results
        print('# Prediction examples: prediction | target | input')
        for items in examples: 
            i, l, p = items

            # Transform indices to words
            input2word = [train_data.idx_to_word[e.item()] for e in i]
            pred2word = train_data.idx_to_word[p]
            targ2word = train_data.idx_to_word[l]
            print('\t', pred2word, '|', targ2word, '|', input2word)
            #print(pred2word + ' | ' + input2word + ' | ' + str(input2word))
        print('\n')

        log_file.close()

        ### t-SNE plot
        if args.tsne:  # Made by Hannah Martiny
            # get words / amino acids that are unique
            words = sorted(train_data.corpus_counts.wc, key=train_data.corpus_counts.wc.get, reverse=True)
            words_array = np.array(words)

            # get learned embeddings
            idx2vec = net.embeddings.weight.data.cpu().numpy()

            plot_name = 'results/' + args.post_fix + '_tSNE.png'
            plot_tSNE(idx2vec=idx2vec, word2idx=train_data.word_to_idx, words=words_array, filename=plot_name, use_cuda=use_cuda)



    #########################
    # # Model training
    #########################
    else: 
        # Lists and parameters
        examples, n_examples = [], 5
        losses = []
        train_acc, train_loss = [], []
        valid_acc, valid_loss = [], []
        iter_count = [0]
        verbose = False

        # Run through epochs
        max_epochs = args.epochs + begin
        for epoch in range(begin, max_epochs):
            ### Train ###
            current_loss = 0
            net.train()
            train_lengths, train_accs = 0, 0
            for i, (inputs, labels) in enumerate(load_train):
                n_samples = inputs.shape[0]
                
                # Convert targets and input to cuda if available
                if use_cuda: 
                  inputs = inputs.cuda()
                  labels = labels.cuda()

                # Zero gradient
                net.zero_grad()

                # Run the forward pass, getting probabilities over next words
                probs = net(inputs)

                # Compute loss
                loss = criterion(probs, labels)

                # Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                current_loss += loss.item() * n_samples
                
                # Accuracy
                train_accs += accuracy(y_true=labels, y_pred=probs) * n_samples 
                train_lengths += n_samples
            
            train_loss.append(current_loss/train_lengths)
            train_acc.append(train_accs/train_lengths)

            
            ### Evaluation ###
            net.eval()
            
            val_preds, val_targs = [], []
            val_losses, val_accs, val_lengths = 0, 0, 0

            for i, (inputs, labels) in enumerate(load_valid):
                n_samples = inputs.shape[0]
              
                # Convert targets and input to cuda if available
                if use_cuda: 
                  inputs = inputs.cuda()
                  labels = labels.cuda()
                
                # Get predictions
                output = net(inputs)
                preds = torch.max(input=output, dim=1)[1]
                
                if use_cuda: 
                   preds = preds.data.cpu().numpy()
                else: 
                   preds = preds.data.numpy()
                
                # Calculate validation loss
                val_losses += criterion(output, labels).item() * n_samples
                val_accs += accuracy(y_true=labels, y_pred=output) * n_samples
                val_lengths += n_samples
                
                # Save predictions and labels
                val_preds += preds.tolist()
                val_targs += labels.tolist()
                
                # Save example inputs
                if len(examples) < n_examples: 
                    for n in range(n_examples):
                      examples.append([inputs[n], labels[n].item(), preds[n].item()])

            # Calculate accuracy and loss
            valid_loss.append(val_losses/val_lengths)
            valid_acc.append(val_accs/val_lengths)
            
            # Show results of evaluation
            print('# Epoch %2i, TRAIN: loss=%f, perp=%f, acc=%f\n' % (epoch+1, current_loss/train_lengths, np.exp(current_loss/train_lengths), train_accs/train_lengths))
            print('# Epoch %2i, VALID: loss=%f, perp=%f, acc=%f\n' % (epoch+1, val_losses/val_lengths, np.exp(val_losses/val_lengths), val_accs/val_lengths))

            # Write to log file
            log_file.write(str(epoch+1)+'\ttrain\t'+str(current_loss/train_lengths)+'\t'+str(np.exp(current_loss/train_lengths))+'\t'+str(train_accs/train_lengths)+'\n')
            log_file.write(str(epoch+1)+'\tvalid\t'+str(val_losses/val_lengths)+'\t'+str(np.exp(val_losses/val_lengths))+'\t'+str(val_accs/val_lengths)+'\n')

            #if epoch % 1 == 0:
            #print("### Epoch %2i:\tTrain loss %f, Train perplexity %f, Train acc %f\n\t\tValid loss %f, Valid perplexity %f, Valid acc %f\n" % (
            #        epoch+1, train_loss[-1], train_perp, train_acc_cur, valid_loss[-1], val_perp, valid_acc_cur))

            # Show top N validation samples and their results
            #print('# Predition examples: prediction | target | input')
            #for items in examples: 
            #  i, l, p = items

              # Transform indices to words
            #  input2word = [train_data.idx_to_word[e.item()] for e in i]
            #  pred2word = train_data.idx_to_word[p]
            #  targ2word = train_data.idx_to_word[l]
            #  print('\t', pred2word, '|', targ2word, '|', input2word)
            #  #print(pred2word + ' | ' + input2word + ' | ' + str(input2word))
            #print('\n')

            # Save model 
            #torch.save(net.state_dict(), 'models/model_state_dict_epoch{0}_{1}.pt'.format(epoch+1, args.post_fix))

            # Save checkpoint
            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': current_loss/train_lengths, 'valid_loss': val_losses/val_lengths}, 
                'checkpoints/check{0}_{1}.pt'.format(epoch+1, args.post_fix))


        # Plot performances 
        epoch = np.arange(begin, max_epochs)
        fig, axes = plt.subplots(figsize=(2*7,7), ncols=2, nrows=1)
        ylabels = ['Loss', 'Accuracy']
        legends = [['Train loss', 'Valid loss'], ['Train Acc', 'Val Acc']]
        ydata = [[[np.exp(l) for l in train_loss], [np.exp(l) for l in valid_loss]], [train_acc, valid_acc]]

        for i, ax in enumerate(fig.axes): 
            ax.plot(epoch, ydata[i][0], 'r', epoch, ydata[i][1], 'b')
            ax.legend(legends[i])
            ax.set_xlabel('Epochs')
            ax.set_ylabel(ylabels[i])
            #if ylabels[i] == 'Accuracy': 
            #    ax.set_ylim(0, 0.3)
            #else: 
            #    ax.set_ylim(15, 20)
        plt.savefig('results/performances_{}.pdf'.format(args.post_fix), dpi=1000)

        log_file.close()


#######################
#   Run the program 
#######################
if __name__ == "__main__":
    parser = get_args()
    try:
        args = parser.parse_args()
    except IOError as msg: 
        parser.error(str(msg))

    main(args)      # Main program 

