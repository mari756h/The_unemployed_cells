import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    
    def __init__(self, embedding_dim, vocab_size, n_negs=5, padding_idx=0):
        super(SkipGram, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_negs =5
        
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
        # nwords = torch.FloatTensor(batch_size, context_size*self.n_negs).uniform_(0, self.vocab_size-1).long()

        center_vectors = self.in_embedding(center).unsqueeze(2)
        context_vectors = self.out_embedding(context)
        # negative_vectors = self.out_embedding(nwords).neg()
        
        # pos_loss = torch.bmm(context_vectors, center_vectors).squeeze().sigmoid().log().mean(-1)
        # neg_loss = torch.bmm(negative_vectors, center_vectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)

        
        predictions = torch.bmm(context_vectors, center_vectors)
        predictions = predictions.squeeze()
        try:
                predictions = predictions.softmax(dim=1)
        except:
                predictions = predictions.softmax(dim=-1)
        predictions = predictions.log().view(-1, context_size)
        
        return predictions
