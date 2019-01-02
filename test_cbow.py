#!/usr/bin/python3

# Import packages
import CBoW_scripts.functions as f
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import sys
import os
from tqdm import tqdm
import numpy as np
import argparse


#########################
#       Parser
#########################
def get_args():
    parser = argparse.ArgumentParser(description='Module will run continuous bag-of-words on amino acid sequences.')
    parser.add_argument('-test_data', required=True, dest='test_data', type=str, help='Path to test data.')
    parser.add_argument('-model', required=True, dest='model', type=str, help='Path to test data.')
    parser.add_argument('-word2idx', required=True, dest='word2idx', type=str, help='Path to word2idx (Output file from training).')
    parser.add_argument('-direction', required=False, dest='window_direction', type=str, choices=['before', 'after', 'both'], help='Direction of context window.')
    parser.add_argument('-padding', required=False, dest='padding', action='store_true', help='Whether to use padding on not.')
    parser.add_argument('-window_size', required=False, dest='window_size', type=int, help='Size of the context window.')
    parser.add_argument('-batch_size', required=False, dest='batch_size', type=int, help='Size of neural network batches.')
    parser.add_argument('-embed_dim', required=False, dest='embedding_dim', type=int, help='Number of embedding dimensions.')
    
    # Set defaults
    parser.set_defaults(window_direction='both', padding=True, window_size=3, batch_size=128, embedding_dim=100)
    return parser


########################################################################
#       MAIN FUNCTION
########################################################################
def main(args): 
    # Load word2idx (was created during training)
    word2idx = torch.load(args.word2idx)
    print(word2idx)


    # Load test data
    test_data = f.DataLoader()
    test_data.load_corpus(path=args.test_data)

    # Make context pairs for validation data
    test_data.make_context_pairs(window_size=args.window_size, padding=args.padding, direction=args.window_direction)

    # Convert to numpy
    test_data.words_to_index(word2idx=word2idx)

    # After data has been loaded it is good to check what is looks like. 
    print('Number of test samples:\t', test_data.context_array[0].shape)

    # Make batches
    test = data_utils.TensorDataset(torch.from_numpy(test_data.context_array[0]), 
                                    torch.from_numpy(test_data.context_array[1]))
    load_test = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=True)

    # Set up to use GPU if available
    use_cuda = torch.cuda.is_available()


    # Load net class
    net = f.cbow(vocab_size=len(word2idx), embedding_dim=args.embedding_dim, padding=True)

    # If GPU is available
    if use_cuda:
        print('# Converting network to cuda-enabled')
        net.cuda()
        loc_map = None
    else: 
        loc_map='cpu'
    print(net)


    # Set up neural net
    check = torch.load(args.model, map_location=loc_map)
    net.load_state_dict(check['model_state_dict'])
    epoch = check['epoch']

    # Set criterion 
    criterion = nn.CrossEntropyLoss()

    # Run model on test set
    test_acc, test_loss = [], []

    ### Evaluation ###
    net.eval()

    test_preds, test_targs = [], []
    test_losses, test_accs, test_lengths = 0, 0, 0
    examples, n_examples = [], 5

    # Print running 
    pbar_test = tqdm(load_test, position=0)
    pbar_test.set_description("[Epoch {}, test]".format(epoch+1))

    for i, (inputs, labels) in enumerate(pbar_test):
        #print('Batch {0}/{1}'.format(i+1, len(load_test)))
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
        test_accs += f.accuracy(y_true=labels, y_pred=output) * n_samples
        test_lengths += n_samples

        # Save predictions and labels
        test_preds += preds.tolist()
        test_targs += labels.tolist()

        # Save example inputs
        if len(examples) < n_examples: 
            for n in range(n_examples):
                examples.append([inputs[n], labels[n].item(), preds[n].item()])
        
        # Print percentage run
        pbar_test.set_postfix(loss=test_losses/test_lengths, perp=np.exp(test_losses/test_lengths), acc=test_accs/test_lengths)
    print('\n### Test completed!')


    # Show results of evaluation
    print('# Epoch %2i, TEST: loss=%f, perp=%f, acc=%f\n' % (epoch+1, test_losses/test_lengths, 
                                                             np.exp(test_losses/test_lengths), 
                                                             test_accs/test_lengths))


    # Create idx2word
    idx2word = {value: key for key, value in word2idx.items()}
    print(idx2word)

    # Show examples of predictions
    print('# Input | Label | Prediction\n')
    for ex in examples: 
        i, l, p = ex
        i = [idx2word[idx.item()] for idx in i]
        print(''.join(i) + ' | ' + idx2word[l] + ' | ' + idx2word[p])


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
