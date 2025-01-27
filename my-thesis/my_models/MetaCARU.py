#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Leandro M. de Lima

This file implements the Metadata CARU Processing Block (MetaCARU)

If you find any bug or have some suggestion, please, email me.
"""

import torch.nn as nn
import torch

from densesparse import DenseSparse

class MetaCARU(nn.Module):
    """
    Implementing the Metadata CARU Processing Block (MetaCARU)
    """
    def __init__(self, img, meta, densesparse=False):
        super(MetaCARU, self).__init__()
        self.ds = densesparse

        emb_dim = 25
        emb_out = 15

        col_name = ["smoke", "drink", "background_father", "background_mother", "pesticide", "gender",
                    "skin_cancer_history",
                    "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", "region", "itch", "grew",
                    "hurt",
                    "changed", "bleed", "elevation"]
        col_name_dense = [
            "age",
            "diameter_1",
            "diameter_2",
        ]

        self.Fi = nn.Sequential(nn.Linear(img, img), nn.BatchNorm1d(img))
        self.Gi = nn.Sequential(nn.Linear(img, img), nn.BatchNorm1d(img))

        if self.ds:
            self.Fm = nn.Sequential(DenseSparse(col_name_dense, col_name, emb_dim, img), nn.BatchNorm1d(img))
            self.Gm = nn.Sequential(DenseSparse(col_name_dense, col_name, emb_dim, img), nn.BatchNorm1d(img))
        else:
            self.Fm = nn.Sequential(nn.Linear(meta, img), nn.BatchNorm1d(img))
            self.Gm = nn.Sequential(nn.Linear(meta, img), nn.BatchNorm1d(img))

    def forward(self, img, meta):
        fi = self.Fi(img)
        gi = self.Gi(img)

        fm = self.Fm(meta)
        gm = self.Gm(meta)

        p = fm
        n = torch.tanh(fi + p)
        z = torch.sigmoid(gm + gi)
        l = torch.sigmoid(p) * z
        h = ((1-l)*img) + (l*n)

        return h

class MetaCARUBn(nn.Module):
    """
    Implementing the Metadata CARU Processing Block (MetaCARU)
    """
    def __init__(self, img, meta):
        super(MetaCARUBn, self).__init__()
        self.Fi = nn.Sequential(nn.Linear(img, img), nn.BatchNorm1d(img))
        self.Gi = nn.Sequential(nn.Linear(img, img), nn.BatchNorm1d(img))

        self.Fm = nn.Sequential(nn.Linear(meta, img), nn.BatchNorm1d(img))
        self.Gm = nn.Sequential(nn.Linear(meta, img), nn.BatchNorm1d(img))

        self.bn = nn.BatchNorm1d(img)

    def forward(self, img, meta):
        fi = self.Fi(img)
        gi = self.Gi(img)

        fm = self.Fm(meta)
        gm = self.Gm(meta)

        p = fm
        n = torch.tanh(fi + p)
        z = torch.sigmoid(gm + gi)
        l = torch.sigmoid(p) * z
        h = ((1-l)*img) + (l*n)

        return self.bn(h)

class MetaCARUsimplified(nn.Module):
    """
    Implementing the Metadata CARU Processing Block (MetaCARU)
    """
    def __init__(self, img, meta):
        super(MetaCARUsimplified, self).__init__()

        self.Fm = nn.Sequential(nn.Linear(meta, img), nn.BatchNorm1d(img))
        self.Gm = nn.Sequential(nn.Linear(meta, img), nn.BatchNorm1d(img))

    def forward(self, img, meta):
        fi = img
        gi = img

        fm = self.Fm(meta)
        gm = self.Gm(meta)

        p = fm
        n = torch.tanh(fi + p)
        z = torch.sigmoid(gm + gi)
        l = torch.sigmoid(p) * z
        h = ((1-l)*img) + (l*n)

        return h

class MetaCARUsimplifiedBn(nn.Module):
    """
    Implementing the Metadata CARU Processing Block (MetaCARU) with extra batch normalization before output
    """
    def __init__(self, img, meta):
        super(MetaCARUsimplifiedBn, self).__init__()

        self.Fm = nn.Sequential(nn.Linear(meta, img), nn.BatchNorm1d(img))
        self.Gm = nn.Sequential(nn.Linear(meta, img), nn.BatchNorm1d(img))

        self.bn = nn.BatchNorm1d(img)

    def forward(self, img, meta):
        fi = img
        gi = img

        fm = self.Fm(meta)
        gm = self.Gm(meta)

        p = fm
        n = torch.tanh(fi + p)
        z = torch.sigmoid(gm + gi)
        l = torch.sigmoid(p) * z
        h = ((1-l)*img) + (l*n)

        return self.bn(h)
