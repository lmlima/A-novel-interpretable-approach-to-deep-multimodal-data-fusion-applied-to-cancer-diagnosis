#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Leandro M. de Lima

This file implements the Metadata CARU Processing Block (MetaCARU)

If you find any bug or have some suggestion, please, email me.
"""

import torch.nn as nn
import torch
from MetaCARU import MetaCARU, MetaCARUsimplified
from metablock import MetaBlock


class MutualAttention(nn.Module):
    """
    Implementing the Mutual Attention fusion
    https://doi.org/10.1007/s00371-022-02492-4
    """

    def __init__(self, img, meta, num_heads=3):
        super(MutualAttention, self).__init__()
        self.num_heads = num_heads

        self.MLPMeta = MAMLP(meta, meta, meta)
        self.MLPImage = MAMLP(img, img, img)

        # nn.MultiheadAttention()
        self.MAttMeta = nn.MultiheadAttention(self.num_heads * meta, self.num_heads)
        self.MAttImage = nn.MultiheadAttention(self.num_heads * img, self.num_heads)

        self.MAttMeta_linear = nn.Linear(self.num_heads * meta, meta)
        self.MAttImage_linear = nn.Linear(self.num_heads * img, img)

        meta_size = self.num_heads * meta
        img_size = self.num_heads * img
        # meta_size = meta
        # img_size = img
        # Meta k,v,q
        self.k_meta = nn.Linear(meta_size, meta_size)
        self.v_meta = nn.Linear(meta_size, meta_size)
        self.q_meta = nn.Linear(meta_size, img_size)

        # Image k,v,q
        self.k_img = nn.Linear(img_size, img_size)
        self.v_img = nn.Linear(img_size, img_size)
        self.q_img = nn.Linear(img_size, meta_size)

    def forward(self, img, meta):
        meta = SLE(meta)

        meta = self.MLPMeta(meta)
        img = self.MLPImage(img)

        MAimg = img.repeat(1, self.num_heads)
        MAmeta = meta.repeat(1, self.num_heads)

        q_img_ma = self.q_img(MAimg).unsqueeze(0)
        k_meta_ma = self.k_meta(MAmeta).unsqueeze(0)
        v_meta_ma = self.v_meta(MAmeta).unsqueeze(0)

        q_meta_ma = self.q_meta(MAmeta).unsqueeze(0)
        k_img_ma = self.k_img(MAimg).unsqueeze(0)
        v_img_ma = self.v_img(MAimg).unsqueeze(0)

        meta_out = self.MAttMeta_linear(self.MAttMeta(q_img_ma, k_meta_ma, v_meta_ma)[0])
        img_out = self.MAttImage_linear(self.MAttImage(q_meta_ma, k_img_ma, v_img_ma)[0])

        output = torch.cat([meta_out + meta.unsqueeze(0), img_out + img.unsqueeze(0)], -1)

        return output.squeeze(0)

class MAMCARU(nn.Module):
    """
    Implementing the Mutual Attention boosting with MetaCARU fusion.
    https://doi.org/10.1007/s00371-022-02492-4
    """

    def __init__(self, img, meta, num_heads=3):
        super(MAMCARU, self).__init__()
        self.num_heads = num_heads

        # nn.MultiheadAttention()
        self.MAttMeta = nn.MultiheadAttention(self.num_heads * meta, self.num_heads)
        self.MAttImage = nn.MultiheadAttention(self.num_heads * img, self.num_heads)

        self.MAttMeta_linear = nn.Linear(self.num_heads * meta, meta)
        self.MAttImage_linear = nn.Linear(self.num_heads * img, img)

        meta_size = self.num_heads * meta
        img_size = self.num_heads * img
        # meta_size = meta
        # img_size = img

        self.MetaCARU = MetaCARU(img, meta)

        # Meta k,v,q
        self.k_meta = nn.Linear(meta_size, meta_size)
        self.v_meta = nn.Linear(meta_size, meta_size)
        self.q_meta = nn.Linear(meta_size, img_size)

        # Image k,v,q
        self.k_img = nn.Linear(img_size, img_size)
        self.v_img = nn.Linear(img_size, img_size)
        self.q_img = nn.Linear(img_size, meta_size)

    def forward(self, img, meta):
        MAimg = img.repeat(1, self.num_heads)
        MAmeta = meta.repeat(1, self.num_heads)

        q_img_ma = self.q_img(MAimg).unsqueeze(0)
        k_meta_ma = self.k_meta(MAmeta).unsqueeze(0)
        v_meta_ma = self.v_meta(MAmeta).unsqueeze(0)

        q_meta_ma = self.q_meta(MAmeta).unsqueeze(0)
        k_img_ma = self.k_img(MAimg).unsqueeze(0)
        v_img_ma = self.v_img(MAimg).unsqueeze(0)

        meta_out = self.MAttMeta_linear(self.MAttMeta(q_img_ma, k_meta_ma, v_meta_ma)[0]).squeeze(0)
        img_out = self.MAttImage_linear(self.MAttImage(q_meta_ma, k_img_ma, v_img_ma)[0]).squeeze(0)

        # output = torch.cat([meta_out + meta.unsqueeze(0), img_out + img.unsqueeze(0)], -1)
        output = self.MetaCARU(img_out + img, meta_out + meta)

        return output

class MAMCARUS(nn.Module):
    """
    Implementing the Mutual Attention boosting with MetaCARU Simplified fusion.
    https://doi.org/10.1007/s00371-022-02492-4
    """

    def __init__(self, img, meta, num_heads=3):
        super(MAMCARUS, self).__init__()
        self.num_heads = num_heads

        # nn.MultiheadAttention()
        self.MAttMeta = nn.MultiheadAttention(self.num_heads * meta, self.num_heads)
        self.MAttImage = nn.MultiheadAttention(self.num_heads * img, self.num_heads)

        self.MAttMeta_linear = nn.Linear(self.num_heads * meta, meta)
        self.MAttImage_linear = nn.Linear(self.num_heads * img, img)

        meta_size = self.num_heads * meta
        img_size = self.num_heads * img
        # meta_size = meta
        # img_size = img

        self.MetaCARUsimplified = MetaCARUsimplified(img, meta)

        # Meta k,v,q
        self.k_meta = nn.Linear(meta_size, meta_size)
        self.v_meta = nn.Linear(meta_size, meta_size)
        self.q_meta = nn.Linear(meta_size, img_size)

        # Image k,v,q
        self.k_img = nn.Linear(img_size, img_size)
        self.v_img = nn.Linear(img_size, img_size)
        self.q_img = nn.Linear(img_size, meta_size)

    def forward(self, img, meta):
        MAimg = img.repeat(1, self.num_heads)
        MAmeta = meta.repeat(1, self.num_heads)

        q_img_ma = self.q_img(MAimg).unsqueeze(0)
        k_meta_ma = self.k_meta(MAmeta).unsqueeze(0)
        v_meta_ma = self.v_meta(MAmeta).unsqueeze(0)

        q_meta_ma = self.q_meta(MAmeta).unsqueeze(0)
        k_img_ma = self.k_img(MAimg).unsqueeze(0)
        v_img_ma = self.v_img(MAimg).unsqueeze(0)

        meta_out = self.MAttMeta_linear(self.MAttMeta(q_img_ma, k_meta_ma, v_meta_ma)[0]).squeeze(0)
        img_out = self.MAttImage_linear(self.MAttImage(q_meta_ma, k_img_ma, v_img_ma)[0]).squeeze(0)

        # output = torch.cat([meta_out + meta.unsqueeze(0), img_out + img.unsqueeze(0)], -1)
        output = self.MetaCARUsimplified(img_out + img, meta_out + meta)

        return output

class MAMBlock(nn.Module):
    """
    Implementing the Mutual Attention boosting with MetaBlock fusion.
    https://doi.org/10.1007/s00371-022-02492-4
    """

    def __init__(self, img, meta, num_heads=3):
        super(MAMBlock, self).__init__()
        self.num_heads = num_heads

        # nn.MultiheadAttention()
        self.MAttMeta = nn.MultiheadAttention(self.num_heads * meta, self.num_heads)
        self.MAttImage = nn.MultiheadAttention(self.num_heads * img, self.num_heads)

        self.MAttMeta_linear = nn.Linear(self.num_heads * meta, meta)
        self.MAttImage_linear = nn.Linear(self.num_heads * img, img)

        meta_size = self.num_heads * meta
        img_size = self.num_heads * img
        # meta_size = meta
        # img_size = img

        self.MetaBlock = MetaBlock(img, meta)

        # Meta k,v,q
        self.k_meta = nn.Linear(meta_size, meta_size)
        self.v_meta = nn.Linear(meta_size, meta_size)
        self.q_meta = nn.Linear(meta_size, img_size)

        # Image k,v,q
        self.k_img = nn.Linear(img_size, img_size)
        self.v_img = nn.Linear(img_size, img_size)
        self.q_img = nn.Linear(img_size, meta_size)

        self.flatten = nn.Flatten()

    def forward(self, img, meta):
        MAimg = img.repeat(1, self.num_heads)
        MAmeta = meta.repeat(1, self.num_heads)

        q_img_ma = self.q_img(MAimg).unsqueeze(0)
        k_meta_ma = self.k_meta(MAmeta).unsqueeze(0)
        v_meta_ma = self.v_meta(MAmeta).unsqueeze(0)

        q_meta_ma = self.q_meta(MAmeta).unsqueeze(0)
        k_img_ma = self.k_img(MAimg).unsqueeze(0)
        v_img_ma = self.v_img(MAimg).unsqueeze(0)

        meta_out = self.MAttMeta_linear(self.MAttMeta(q_img_ma, k_meta_ma, v_meta_ma)[0]).squeeze(0)
        img_out = self.MAttImage_linear(self.MAttImage(q_meta_ma, k_img_ma, v_img_ma)[0]).squeeze(0)

        # output = torch.cat([meta_out + meta.unsqueeze(0), img_out + img.unsqueeze(0)], -1)

        sum_img = img_out + img
        # Make sure there is at least 3 dimensions
        if len(sum_img.shape) < 3:
            sum_img = sum_img.unsqueeze(2)
        output = self.MetaBlock(sum_img, meta_out + meta)

        output = self.flatten(output)

        return output


def SLE(x):
    """
    Soft Label Encoder
    :param x:
    :return:
    """
    x[x == 0] = 0.1
    return x

class MAMLP(nn.Module):
    """
    Multi Layer Perceptron. It is 3 fully connected layers, each full-connection layer is connected to the ReLU6 activation function.
    """

    def __init__(self, input_size, output_size, hidden_size):
        super(MAMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x
