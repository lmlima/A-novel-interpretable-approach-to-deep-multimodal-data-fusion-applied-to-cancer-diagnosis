#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco, Leandro Lima
E-mail: pacheco.comp@gmail.com

This file implements the Metadata Processing Block (MetaBlock)

If you find any bug or have some suggestion, please, email me.
"""

import torch.nn as nn
import torch
import math


class MetaBlock(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U):
        super(MetaBlock, self).__init__()
        self.T1 = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.T2 = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        if isinstance(U, list):
            t1 = self.T1(U[0])
            t2 = self.T2(U[1])
        else:
            t1 = self.T1(U)
            t2 = self.T2(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V


class MetaBlockKron(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    torch.kron(U[0], torch.eye(U.shape[-1], device="cuda")).shape
    """

    def __init__(self, V, U):
        super(MetaBlock, self).__init__()
        self.T1 = nn.Sequential(nn.Linear(U, U * V), nn.BatchNorm1d(U * V))
        # self.T2 = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        # t1 = self.T1(U)
        t2 = self.T2(U)
        #        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        V = torch.sigmoid(torch.tanh(torch.kron(V, U)) + t2.unsqueeze(-1))
        return V


class GCell(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U):
        super(GCell, self).__init__()
        self.T1 = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))
        self.T2 = nn.Sequential(nn.Linear(U, V), nn.BatchNorm1d(V))

    def forward(self, V, U):
        t1 = self.T1(U)
        t2 = self.T2(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V


class MetaBlockSN(nn.Module):
#class MetaBlockSNv3(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U, num_groups=32):
        super(MetaBlockSN, self).__init__()
        self.T1 = nn.Sequential(nn.Linear(U, V), nn.GroupNorm(num_groups, V))
        self.T2 = nn.Sequential(nn.Linear(U, V), nn.GroupNorm(num_groups, V))

        # Initialize weights with Kaiming initialization
        for module in [self.T1[0], self.T2[0]]:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, V, U):
        t1 = self.T1(U)
        t2 = self.T2(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V

class MetaBlockSNv2(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U, num_groups=32):
        super(MetaBlockSNv2, self).__init__()
        self.T1 = nn.Sequential(nn.Linear(U, V), nn.GroupNorm(num_groups, V), nn.GELU())
        self.T2 = nn.Sequential(nn.Linear(U, V), nn.GroupNorm(num_groups, V), nn.GELU())

        # Initialize weights with Kaiming initialization
        for module in [self.T1[0], self.T2[0]]:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, V, U):
        t1 = self.T1(U)
        t2 = self.T2(U)
        V = torch.sigmoid(torch.tanh(V * t1.unsqueeze(-1)) + t2.unsqueeze(-1))
        return V


class MetaBlockSNv1(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, V, U, num_groups=32):
        super(MetaBlockSNv1, self).__init__()
        self.T1 = nn.Sequential(nn.Linear(U, V), nn.GroupNorm(num_groups, V), nn.GELU(), nn.Softplus())
        self.T2 = nn.Sequential(nn.Linear(U, V), nn.GroupNorm(num_groups, V), nn.GELU(), nn.Softplus())

        # Initialize weights with Kaiming initialization
        for module in [self.T1[0], self.T2[0]]:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, V, U):
        t1 = self.T1(U)
        t2 = self.T2(U)
        V = torch.mul(torch.sigmoid(V), torch.exp(torch.mul(V, t1.unsqueeze(-1))) + t2.unsqueeze(-1))
        return V
