import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class TextCNN(nn.Module):
    """ PyTorch implementation of a convolutional neural network for sentence classification [1].

    Implementation is adapted from the following repository https://github.com/Shawn1993/cnn-text-classification-pytorch

    Attributes
    ----------
    num_embed: int
        number of embeddings
    dim_embed: int
        dimension of embedding layer
    num_class: int
        number of classes
    p_dropout: float
        probability of dropout
    in_channels: list
        ..
    out_channels: list
        ..
    kernel_sizes: list
        ..
    strides:
        ..

    Methods
    ----------
    load_embeddings(matrix, non_trainable)
        aa
    forward(x)
        feed data through network


    References
    ----------
    1. Kim Y. Convolutional neural networks for sentence classification. arXiv Prepr arXiv14085882. 2014:1746–1751. https://arxiv.org/pdf/1408.5882.pdf
    """

    def __init__(self, num_embed, dim_embed, num_class, p_dropout, in_channels, out_channels, kernel_sizes, strides):
        super(TextCNN, self).__init__()

        self.num_embed = num_embed #Vocab size, V
        self.dim_embed = dim_embed # Emb dim, D
        self.num_class = num_class # C
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # embedding layer from skip-gram or cbow
        self.embed = nn.Embedding(
            num_embeddings=self.num_embed, embedding_dim=self.dim_embed)

        # convolutional part of the neural network
        # allows for the possibility of multiple filters
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=(self.kernel_sizes[i], self.dim_embed),
                      stride=self.strides) 
            for i in range(len(self.kernel_sizes))
        ])


        # regularization
        self.dropout = nn.Dropout(p=p_dropout)

        # linear output layer
        self.fc1 = nn.Linear(in_features=len(self.kernel_sizes)*self.out_channels, out_features=self.num_class)

    def load_embeddings(self, matrix, trainable=False):
        """Load pretrained word embeddings to the embedding module

        Inspired by the following blog post: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

        Attributes
        ----------
        Matrix: array or tensor
            pretrained word embedding matrix
        non_trainable: boolean, default: False
            do not train the embeddings if true

        """
        self.embed.weight.data.copy_(matrix)

        if not trainable:
            self.embed.weight.requires_grad = False
    
    def forward(self, x):
        """Run data through network.
        
        Attributes
        ----------
        x: tensor
            input to network

        Returns
        ----------
        out: tensor
            output of network
        """

        # get embeddings
        x = self.embed(x)
        x = torch.unsqueeze(x, 1)

        # # run x through the different filters
        # xs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # # max-over-time pooling
        # xs = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in xs]
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        
        # concatenate
        out = torch.cat(xs, 2)
        out = out.view(out.size(0), -1)

        # dropout
        out = self.dropout(out)

        # and output linear layer
        out = self.fc1(out)

        return out
