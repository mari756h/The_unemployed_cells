import torch
import torch.nn as nn
import torch.nn.functional as F

class cbow(nn.Module):

    def __init__(self, vocab_size, embedding_dim=20, padding=True):
        super(cbow, self).__init__()
        # num_embeddings is the number of words in your train, val and test set
        # embedding_dim is the dimension of the word vectors you are using
        if padding: 
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, 
                                          padding_idx=0)
        else: 
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, 
                                          padding_idx=None)
        
        self.linear_out = nn.Linear(in_features=embedding_dim, out_features=vocab_size, bias=False)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        
        # To not care about the order of the words we take the mean of the time dimension
        means = torch.mean(embeds, dim=1)
        
        # Softmax on output
        #probs = F.log_softmax(out, dim=1)
        probs = F.log_softmax(self.linear_out(means), dim=1)
        
        return probs