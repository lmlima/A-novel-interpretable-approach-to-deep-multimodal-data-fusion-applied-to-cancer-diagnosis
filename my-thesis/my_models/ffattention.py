#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Leandro Lima
"""

#
# ff-attention in Pytorch package, created by
# Dylan Bourgeois (@dtsbourg)
#
# License: MIT (see LICENSE.md)
#
# Original work by Colin Raffel and Daniel P. Ellis
# "Feed-Forward Networks with Attention Can
# Solve Some Long-Term Memory Problems"
# https://arxiv.org/abs/1512.08756
# (Licensed under CC-BY)

import torch
import torch.nn.functional as F


class FFAttention(torch.nn.Module):
    """
    `FFAttention` is the Base Class for the Feed-Forward Attention Network.
    It is implemented as an abstract subclass of a PyTorch Module. You can
    then subclass this to create an architecture adapted to your problem.
    The FeedForward mecanism is implemented in five steps, three of
    which have to be implemented in your custom subclass:
    1. `embedding` (NotImplementedError)
    2. `activation` (NotImplementedError)
    3. `attention` (Already implemented)
    4. `context` (Already implemented)
    5. `out` (NotImplementedError)
    Attributes:
        batch_size (int): The batch size, used for resizing the tensors.
        T (int): The length of the sequence.
        D_in (int): The dimension of each element of the sequence.
        D_out (int): The dimension of the desired predicted quantity.
        hidden (int): The dimension of the hidden state.
    """
    def __init__(self, batch_size=30, T=1, D_in=87, D_out=6, hidden=100):
        super(FFAttention, self).__init__()
        # Net Config
        self.T = T
        self.batch_size = batch_size
        self.n_features = D_in
        self.out_dim = D_out
        self.hidden = hidden

    def embedding(self, x_t):
        """
        Step 1:
        Compute embeddings h_t for element of sequence x_t
        In : torch.Size([batch_size, sequence_length, sequence_dim])
        Out: torch.Size([batch_size, sequence_length, hidden_dimensions])
        """
        raise NotImplementedError

    def activation(self, h_t):
        """
        Step 2:
        Compute the embedding activations e_t
        In : torch.Size([batch_size, sequence_length, hidden_dimensions])
        Out: torch.Size([batch_size, sequence_length, 1])
        """
        raise NotImplementedError

    def attention(self, e_t):
        """
        Step 3:
        Compute the probabilities alpha_t
        In : torch.Size([batch_size, sequence_length, 1])
        Out: torch.Size([batch_size, sequence_length, 1])
        """
        softmax = torch.nn.Softmax(dim=1)
        alphas = softmax(e_t)
        return alphas

    def context(self, alpha_t, x_t):
        """
        Step 4:
        Compute the context vector c
        In : torch.Size([batch_size, sequence_length, 1]), torch.Size([batch_size, sequence_length, sequence_dim])
        Out: torch.Size([batch_size, 1, hidden_dimensions])
        """
        # return torch.bmm(alpha_t.view(self.batch_size, self.out_dim, self.T), x_t)
        return torch.bmm(alpha_t.view(-1, self.T, 1), x_t)

    def out(self, c):
        """
        Step 5:
        Feed-forward prediction based on c
        In : torch.Size([batch_size, 1, hidden_dimensions])
        Out: torch.Size([batch_size, 1, 1])
        """
        raise NotImplementedError

    def forward(self, x, training=True):
        """
        Forward pass for the Feed Forward Attention network.
        """
        self.training = training
        x_e = self.embedding(x)
        x_a = self.activation(x_e)
        alpha = self.attention(x_a)
        x_c = self.context(alpha, x_e)
        x_o = self.out(x_c)
        return x_o, alpha


class FFAttentionClassifier(FFAttention):
    """
    `FFAttentionClassifier` is a subclass of `FFAttention` that
    implements it as part of a classifier.
    """
    def __init__(self, batch_size=30, T=1, D_in=87, D_out=6, hidden=128):
        super(FFAttentionClassifier, self).__init__(batch_size=batch_size, T=T, D_in=D_in, D_out=D_out, hidden=hidden)
        # self.linear = torch.nn.Linear(self.hidden, self.out_dim)
        self.layer0 = torch.nn.Linear(self.n_features, self.hidden)
        self.layer1 = torch.nn.Linear(self.hidden, 1)
        self.layer2 = torch.nn.Linear(self.hidden, self.hidden)
        self.out_layer = torch.nn.Linear(self.hidden, self.out_dim)

    def embedding(self, x_t):
        # x_t: [batch_size, sequence_dim] -> [batch_size, sequence_length, sequence_dim]
        x_t = x_t.unsqueeze(1)
        x_t = self.layer0(x_t)
        return F.leaky_relu(x_t)

    def activation(self, h_t):
        return torch.tanh(self.layer1(h_t))

    def out(self, c):
        # c: [batch_size, sequence_dim] <- [batch_size, sequence_length, sequence_dim]
        c = c.squeeze(1)
        x = F.leaky_relu(self.layer2(c))
        x = F.leaky_relu(self.out_layer(x))
        return x

