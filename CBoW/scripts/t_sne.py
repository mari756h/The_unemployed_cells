from functions import *
import pandas as pd
import matplotlib.pyplot as plt
import torch


wkdir = '/Users/Marianne/Dropbox (Personlig)/DTU/9. semester/02456_Deep_learning/project/results/'



########################
#   Load logs
########################
label = 'before_ws1_em2'
both1 = pd.read_table(wkdir+'logs/log_before_1_lr001_em2.txt')
both_train = both1[both1.set == 'train']
both_val = both1[both1.set == 'valid']
epochs = list(range(1, both_train.shape[0]+1))

print(both_train.head())
print(both_val.head())


###########################
#	Performance plots
###########################
plt.plot(epochs, both_train.perp, 'r', epochs, both_val.perp, 'b')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend(['Train', 'Valid'])
plt.grid(which='major')
plt.title(label)
plt.savefig(wkdir+label+'.pdf', dpi=1000)


# Print epoch of minimum validation perplexity
print('\n# Minimum perplexity:', both_val.loc[both_val.perp.idxmin(axis=0)].perp)

# Epoch of minimum value
epoch_min = both_val.loc[both_val.perp.idxmin(axis=0)].epoch
print('# Minimum epoch:', epoch_min)


########################
#   Make t-SNE plot
########################
# Load word2idx
"""
word2idx = torch.load(wkdir+'checkpoints/word2idx.pt')
print(word2idx)

# Set up neural net
check = torch.load(wkdir+'checkpoints/check20_both_20_lr001_em2.pt', map_location='cpu')
net = cbow(vocab_size=len(word2idx), embedding_dim=2, padding=True)
net.load_state_dict(check['model_state_dict'])

# get words / amino acids that are unique
words = sorted(word2idx, key=word2idx.get, reverse=True)
words_array = np.array(words)

# get learned embeddings
idx2vec = net.embeddings.weight.data.cpu().numpy()

# Plot t-SNE
plot_name = wkdir + label + '_tSNE.pdf'
plot_tSNE(idx2vec=idx2vec, word2idx=word2idx, words=words_array, filename=plot_name)

"""