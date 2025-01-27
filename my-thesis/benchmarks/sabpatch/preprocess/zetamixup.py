"""
Author: Leandro Lima
"""

import torch
import torch.nn.functional as F
import sklearn.preprocessing
import numpy as np
from numpy.random import default_rng


class ZetaMixup():
    def __init__(self, n_cls, b_size, gamma, p=0.3):
        self.rng = default_rng()

        # Specify number of classes and training batch size
        self.n_cls, self.b_size = n_cls, b_size  # 10, 32
        self.gamma = gamma
        self.p = p

        N = b_size
        self.c = sum([x ** (-gamma) for x in list(range(1, N + 1))])


    def __call__(self, batch):
        x = batch[0]
        y = batch[1]

        aug_x, aug_y = self.generate_samples(x, y)
        # Sampled instances to be replaced by self.p percentage
        idx = torch.randperm(self.b_size)[:int(self.b_size * self.p)]

        x[idx, :] = aug_x[idx, :]
        y[idx] = torch.argmax(aug_y, axis=1)[idx]

        return [x, y, batch[2], batch[3]]

    def generate_samples(self, X, Y):
        def zeta_mixup_weights(batch_size, gamma, c):
            """
                Sample weights from the terms of a p-series

                Given N samples (where 2 ≤ N ≤ m and thus, theoretically, the entire dataset),
                an N × N random permutation matrix π, and the resulting randomized ordering of samples
                s = π[1, 2, . . . , N]^T, the weights are defined as:
                w_i = ((s_i)^(−γ)) / C , i ∈ [1, N]
            :param batch_size:
            :param gamma: hyperparameter
            :param c: normalization constant

            :return:
            """
            sequence = list(range(1, batch_size + 1))
            w = [((x) ** (-gamma)) / c for x in sequence]
            w = np.tile(np.array(w), (batch_size, 1))
            # s = map(np.random.shuffle, w)
            # w = np.apply_along_axis(np.random.permutation, 1, w)
            w = np.apply_along_axis(self.rng.permutation, 1, w)
            # w = rng.permutation(w, axis=1)

            w = sklearn.preprocessing.normalize(w, norm='l1')
            return torch.tensor(w, dtype=torch.float)

        def zeta_mixup(X, Y, n_classes, weights):
            """
            X -> input feature tensor ([N, C, H, W])
            Y -> label tensor ([N, 1])
            weights -> weights tensor ([W, W])
            N: batch size; C: channels; H: height; W: width
            """
            # compute weighted average of all samples
            X_new = torch.einsum("ijkl,pi->pjkl", X, weights)
            # encode original labels to one-hot vectors
            Y_onehot = F.one_hot(Y, n_classes).type(torch.float)
            # compute weighted average of all labels
            Y_new = torch.einsum("pq,qj->pj", weights, Y_onehot)
            # return synthesized samples and labels
            return X_new, Y_new

        # Generate weights using normalized p-series
        weights = zeta_mixup_weights(batch_size=self.b_size, gamma=self.gamma, c=self.c)

        return zeta_mixup(X, Y, self.n_cls, weights)

    def generate_label(self, x, y, label, n):
        """
        Generate n samples of specific label
        :param x: dataset features
        :param y: dataset label
        :param label: generate only synthetic samples of that specific labal
        :param n: number of generated synthetic samples
        :return:
        """
        count_aug = 0
        new_x_list = []
        new_y_list = []

        while count_aug < n:
            aug_x, aug_y = self.generate_samples(x, y)
            aug_y = torch.argmax(aug_y, dim=1)

            mask = (aug_y == label)
            aug_y = aug_y[mask]
            aug_x = aug_x[mask]
            count_aug = count_aug + len(aug_y)

            # append new samples
            new_x_list.append(aug_x)
            new_y_list.append(aug_y)

        if new_x_list and new_y_list:
            new_x = torch.cat(new_x_list)[:n]
            new_y = torch.cat(new_y_list)[:n]
        else:
            new_x, new_y = [], []
        # Check if new_x, new_y == n
        return new_x, new_y





# Use
# sys.path.insert(0, '/home/leandro/PycharmProjects/SkinLesion/SkinLesion/my-thesis/benchmarks/sab/preprocess')
# import torch
# from zetamixup import ZetaMixup
# x = torch.randn(15, 3, 224, 224)
# y = torch.randint(0, (3 - 1), (15,))
# a = ZetaMixup(3, 15, 2.8)
# a.generate_samples(x, y)