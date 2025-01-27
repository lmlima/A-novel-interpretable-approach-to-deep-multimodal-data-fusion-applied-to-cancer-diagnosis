"""
Author: Leandro Lima
"""


import torch.nn as nn
import torch
import numpy as np
import hyptorch.nn as hypnn

class DenseSparse(nn.Module):
    """
    Implementing the Feature Interaction Convolutional Neural Network for a single field based on the paper:
    https://ieeexplore.ieee.org/document/9378849
    """

    def __init__(self, dense_len, sparse_len, emb_dim=25, emb_out=15, categorical_max_size=15, poincare_dict=None,
                 use_SNN=False
                 ):
        super(DenseSparse, self).__init__()

        valid_poincare_types = ["feature_conv_concat", "feature_concat"]
        self.dense_len = dense_len
        self.sparse_len = sparse_len

        if use_SNN:
            self.activation = nn.SELU()
        else:
            self.activation = nn.ReLU()

        self.sparse_emb = nn.Embedding(categorical_max_size, emb_dim)

        # 30 kernels of 3 different size 2x1 , 3x1 and 5x1 with stride 1 are applied in convolutional operations
        self.sparse_conv1 = nn.Sequential(
            nn.Conv1d(emb_dim, emb_out, 2, stride=1),
            self.activation,
            nn.MaxPool1d(2, 2),
            nn.Flatten()
        )
        self.sparse_conv2 = nn.Sequential(
            nn.Conv1d(emb_dim, emb_out, 3, stride=1),
            self.activation,
            nn.MaxPool1d(2, 2),
            nn.Flatten()
        )
        self.sparse_conv3 = nn.Sequential(
            nn.Conv1d(emb_dim, emb_out, 5, stride=1),
            self.activation,
            nn.MaxPool1d(2, 2),
            nn.Flatten()
        )

        if use_SNN:
            networks = [self.sparse_conv1, self.sparse_conv2, self.sparse_conv3]
            for net in networks:
                for param in net.parameters():
                    # biases zero
                    if len(param.shape) == 1:
                        nn.init.constant_(param, 0)
                    # others using lecun-normal initialization
                    else:
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

        if poincare_dict is not None and poincare_dict["use_Poincare"]:
            assert poincare_dict["type"] in valid_poincare_types, "Invalid poincare type"

            self.toPC = hypnn.ToPoincare(poincare_dict['c'])
            self.poincare_type = poincare_dict["type"]
            self.poincare_r = torch.tensor(poincare_dict["r"])
        else:
            self.poincare_type = None

        self.output_shape = calculateOutputSize(emb_out, sparse_len, dense_len, self.poincare_type)

    def forward(self, meta):
        dense_meta, sparse_meta = self.separateDenseSparse(meta, self.dense_len)

        sparse_meta = sparse_meta.int() + 1
        sparse_meta_emb = self.sparse_emb(sparse_meta)
        sparse_meta_emb = sparse_meta_emb.transpose(1, 2)

        # Add clip as in GUO, Yunhui et al. Clipped hyperbolic classifiers are super-hyperbolic classifiers.
        # CLIP(x^{E}, r) = min{1, \frac{r}{||x^{E}||}} * x^{E}
        if self.poincare_r > 0:
            sparse_meta_clip = torch.min(torch.tensor(1.), self.poincare_r / torch.norm(sparse_meta.float())) * sparse_meta.float()
            sparse_meta_emb_clip = torch.min(torch.tensor(1.), self.poincare_r / torch.norm(sparse_meta_emb.float())) * sparse_meta_emb.float()
        else:
            sparse_meta_clip = sparse_meta.float()
            sparse_meta_emb_clip = sparse_meta_emb.float()

        if self.poincare_type == "feature_conv_concat":
            sparse_meta_pc = self.toPC(sparse_meta_clip)
            sparse_meta_emb_pc = self.toPC(sparse_meta_emb_clip)
            sparse_meta_emb = torch.cat([sparse_meta_emb, sparse_meta_emb_pc], dim=2)
        elif self.poincare_type == "feature_concat":
            sparse_meta_pc = self.toPC(sparse_meta_clip)

        sparse_meta_conv = torch.cat([
            self.sparse_conv1(sparse_meta_emb),
            self.sparse_conv2(sparse_meta_emb),
            self.sparse_conv3(sparse_meta_emb)

        ], dim=1)

        if self.poincare_type is not None:
            sparse_meta = torch.cat([sparse_meta, sparse_meta_conv, sparse_meta_pc], dim=1)
        else:
            sparse_meta = torch.cat([sparse_meta, sparse_meta_conv], dim=1)
        # sparse_meta = self.metablock1(sparse_meta_conv.unsqueeze(-1), sparse_meta.float()).squeeze()

        features = torch.cat([dense_meta, sparse_meta], dim=1)
        # features = self.metablock2(sparse_meta.unsqueeze(-1), dense_meta).squeeze()

        return features

    def separateDenseSparse(self, meta, split_location):
        split = split_location
        dense_meta = meta[:, -split:]
        sparse_meta = meta[:, :-split]
        return dense_meta, sparse_meta

def calculateOutputSize(emb_out, sparse_len, dense_len, poincare_type):
    def calculateConv1dOutputSize(input_size, kernel_size, stride, padding, dilation=1):
        out_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        return np.floor(out_size).astype(int)

    def calculateMaxPool1dOutputSize(input_size, kernel_size, stride, padding, dilation=1):
        out_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        return np.floor(out_size).astype(int)

    if poincare_type == "feature_conv_concat":
        sparse_len = 2*sparse_len

    conv1d_2_conv_out = calculateConv1dOutputSize(sparse_len, 2, 1, 0)
    conv1d_2_max_out = calculateMaxPool1dOutputSize(conv1d_2_conv_out, 2, 2, 0)
    conv1d_2_out = conv1d_2_max_out * emb_out

    conv1d_3_conv_out = calculateConv1dOutputSize(sparse_len, 3, 1, 0)
    conv1d_3_max_out = calculateMaxPool1dOutputSize(conv1d_3_conv_out, 2, 2, 0)
    conv1d_3_out = conv1d_3_max_out * emb_out

    conv1d_5_conv_out = calculateConv1dOutputSize(sparse_len, 5, 1, 0)
    conv1d_5_max_out = calculateMaxPool1dOutputSize(conv1d_5_conv_out, 2, 2, 0)
    conv1d_5_out = conv1d_5_max_out * emb_out

    # conv1d_7_conv_out = calculateConv1dOutputSize(emb_dim, 7, 1, 1)
    # conv1d_7_max_out = calculateMaxPool1dOutputSize(conv1d_7_conv_out, 7, 2, 1)
    # conv1d_7_out = conv1d_7_max_out * emb_out

    all_conv1d_size = conv1d_2_out + conv1d_3_out + conv1d_5_out

    if poincare_type == "feature_concat":
        sparse_len = 2*sparse_len

    sparse_size = sparse_len + all_conv1d_size
    dense_size = dense_len

    out_size = sparse_size + dense_size
    return int(out_size)
