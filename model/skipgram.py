import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    """Skip-gram architecture in Pytorch

    Attributes
    ----------
    embedding_dim: dimension of embeddings
    vocab_size: vocabulary size
    padding_idx: index for padding

    Methods
    ----------
    forward(center, context)
        Sends the data through the embedding dimension and returns log_softmax probabilities
    """
    
    def __init__(self, embedding_dim, vocab_size, padding_idx=0):
        super(SkipGram, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        self.in_embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embedding_dim, padding_idx=padding_idx)
        self.out_embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim =self.embedding_dim, padding_idx=padding_idx)
        
        self.in_embedding.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_dim), torch.FloatTensor(self.vocab_size - 1, self.embedding_dim).uniform_(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim)]))
        self.out_embedding.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embedding_dim), torch.FloatTensor(self.vocab_size - 1, self.embedding_dim).uniform_(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim)]))
        
        self.in_embedding.weight.requires_grad = True
        self.out_embedding.weight.requires_grad = True
        
    def forward(self, center, context):
        # batch_size = center.size()[0]
        try:
            context_size = context.shape[1]
        except IndexError:
            context_size = 1

        center_vectors = self.in_embedding(center).unsqueeze(2)
        context_vectors = self.out_embedding(context)
        
        predictions = torch.bmm(context_vectors, center_vectors)
        predictions = predictions.squeeze()
        try:
                predictions = predictions.softmax(dim=1)
        except:
                predictions = predictions.softmax(dim=-1)
        predictions = predictions.log().view(-1, context_size)
        
        return predictions
