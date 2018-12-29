
# coding: utf-8

# In[1]:


# Import packages
import functions as f
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import sys
import os
from tqdm import tqdm
import numpy as np

# Working directory
wkdir = '/Users/Marianne/Dropbox (Personlig)/DTU/9. semester/02456_Deep_learning/project/results/'

# Parameters
ws = 20
d_ws = 'after'
bs = 256
emb = 2
check = 20
post_fix = '_{0}_{1}_lr001_em{2}'.format(d_ws, ws, emb)


# # Load data

# In[2]:


# Load word2idx
word2idx = torch.load(wkdir+'checkpoints/word2idx.pt')
print(word2idx)


# In[3]:


# Load test data
test_data = f.DataLoader()
test_data.load_corpus(path='data/proteins.test.txt')

# Make context pairs for validation data
test_data.make_context_pairs(window_size=ws, padding=True, direction=d_ws)

# Convert to numpy
test_data.words_to_index(word2idx=word2idx)

# After data has been loaded it is good to check what is looks like. 
print('Number of test samples:\t', test_data.context_array[0].shape)


# In[4]:


# Make batches
test = data_utils.TensorDataset(torch.from_numpy(test_data.context_array[0]), 
                                torch.from_numpy(test_data.context_array[1]))
load_test = data_utils.DataLoader(test, batch_size=bs, shuffle=True)


# # Load model to test

# In[5]:


# Set up neural net
check = torch.load(wkdir+'checkpoints/check' + str(check) + post_fix + '.pt', map_location='cpu')
net = f.cbow(vocab_size=len(word2idx), embedding_dim=emb, padding=True)
net.load_state_dict(check['model_state_dict'])
epoch = check['epoch']

# Set criterion 
criterion = nn.CrossEntropyLoss()


# # Run test

# In[6]:


# Set up to use GPU if available
use_cuda = torch.cuda.is_available()
use_cuda


# In[7]:


# If GPU is available
if use_cuda:
    print('# Converting network to cuda-enabled')
    net.cuda()
print(net)


# In[8]:


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



# In[ ]:


# Show results of evaluation
print('# Epoch %2i, TEST: loss=%f, perp=%f, acc=%f\n' % (epoch+1, test_losses/test_lengths, 
                                                         np.exp(test_losses/test_lengths), 
                                                         test_accs/test_lengths))


# In[ ]:




