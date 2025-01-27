"""
Author: Leandro Lima
"""


"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""

import torch
from torch import Tensor
import numpy as np

# from easyfsl.methods import FewShotClassifier
from easyfsl.utils import compute_prototypes
from easyfsl.samplers import TaskSampler

import copy

import wrn_model
import torch.backends.cudnn as cudnn
import torch.nn as nn

from abc import abstractmethod
from typing import List, Tuple
from pathlib import Path

import os
import pickle


class FewShotClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(self, backbone: nn.Module, use_softmax: bool = False):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: the feature extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method)
            use_softmax: whether to return predictions as soft probabilities
        """
        super().__init__()

        self.backbone = backbone
        # self.backbone_output_shape = compute_backbone_output_shape(backbone)
        # self.feature_dimension = self.backbone_output_shape[0]

        self.use_softmax = use_softmax

        self.prototypes = None
        self.support_features = None
        self.support_labels = None

    @abstractmethod
    def forward(
            self,
            query_images: Tensor,
    ) -> Tensor:
        """
        Predict classification labels.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    @abstractmethod
    def process_support_set(
            self,
            support_images: Tensor,
            support_labels: Tensor,
    ):
        """
        Harness information from the support set, so that query labels can later be predicted using
        a forward call

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a process_support_set method."
        )

    @staticmethod
    def is_transductive() -> bool:
        raise NotImplementedError(
            "All few-shot algorithms must implement a is_transductive method."
        )

    def softmax_if_specified(self, output: Tensor) -> Tensor:
        """
        If the option is chosen when the classifier is initialized, we perform a softmax on the
        output in order to return soft probabilities.
        Args:
            output: output of the forward method

        Returns:
            output as it was, or output as soft probabilities
        """
        return output.softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:
        """
        Compute prediction logits from their euclidean distance to support set prototypes.
        Args:
            samples: features of the items to classify

        Returns:
            prediction logits
        """
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) -> Tensor:
        """
        Compute prediction logits from their cosine distance to support set prototypes.
        Args:
            samples: features of the items to classify

        Returns:
            prediction logits
        """
        return (
                nn.functional.normalize(samples, dim=1)
                @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def store_support_set_data(
            self,
            support_images: Tensor,
            support_labels: Tensor,
    ):
        """
        Extract support features, compute prototypes,
            and store support labels, features, and prototypes
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        self.support_labels = support_labels
        self.support_features = self.backbone(support_images)
        self.prototypes = compute_prototypes(self.support_features, support_labels)


class PrototypicalNetworks(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def __init__(self, *args, **kwargs):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        super().__init__(*args, **kwargs)

        # if len(self.backbone_output_shape) != 1:
        #     raise ValueError(
        #         "Illegal backbone for Prototypical Networks. "
        #         "Expected output for an image is a 1-dim tensor."
        #     )

    def process_support_set(
            self,
            support_images: Tensor,
            support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        # Extract the features of support images
        if isinstance(support_images, list):
            # Multimodal input
            support_features = self.backbone.forward(*support_images)
        else:
            support_features = self.backbone.forward(support_images)

        # Compute the prototypes of the support set
        self.prototypes = compute_prototypes(support_features, support_labels).float()

    def forward(
            self,
            query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        # Extract the features of query images
        if isinstance(query_images, list):
            # Multimodal input
            query_features = self.backbone.forward(*query_images)
        else:
            query_features = self.backbone.forward(query_images)

        z_query = query_features.float()

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query.float(), self.prototypes.float())

        # Use it to compute classification scores
        scores = -dists

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False


class PrototypicalNetworksDC(PrototypicalNetworks):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Free lunch for few-shot learning: distribution calibration
    https://arxiv.org/pdf/2101.06395.pdf

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    With distribution calibration.
    """

    def __init__(self, backbone, beta=0.5, alpha=0.21, k=2, n_sampled=750, base_info_path=".", save_features=False, *args, **kwargs, ):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
            :param beta: Tukey’s Ladder of Powers and Yeo-Johnson Transformation beta value.
            :param alpha:  Hyper-parameter that determines the degree of dispersion of features sampled from the
             calibrated distribution.
            :param k: Number of base classes to consider in weighted average of the distributions.
            :param n_sampled: Number of sampled points.
        """
        super().__init__(backbone, *args, **kwargs)

        # if len(self.backbone_output_shape) != 1:
        #     raise ValueError(
        #         "Illegal backbone for Prototypical Networks. "
        #         "Expected output for an image is a 1-dim tensor."
        #     )

        self.k = k
        self.beta = beta
        self.alpha = alpha

        self.num_sampled = n_sampled

        self.base_means, self.base_cov = calculate_base_info(base_info_path=base_info_path)

        self.save_features = save_features

    def process_support_set(
            self,
            support_images: Tensor,
            support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        support_features = self.backbone.forward(support_images)
        if self.save_features:
            save_features(support_features, support_labels, "/tmp/support_orig.pkl")

        support_features = gaussianization(support_features, self.beta)
        if self.save_features:
            save_features(support_features, support_labels, "/tmp/support_gauss.pkl")

        n_lsamples = support_features.shape[0]
        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = self.num_sampled
        for i in range(n_lsamples):
            mean, cov = distribution_calibration(
                support_features[i].cpu().numpy(),
                self.base_means, self.base_cov, k=self.k, alpha=self.alpha
            )
            sampled_data.append(torch.from_numpy(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled)))
            # sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
            sampled_label.extend([support_labels[i]] * num_sampled)

        sampled_data = torch.cat(sampled_data).reshape(n_lsamples * num_sampled, -1).to(support_features.device)
        sampled_label = torch.stack(sampled_label)

        # Add augmented samples to support set
        support_features = torch.cat([support_features, sampled_data])
        support_labels = torch.cat([support_labels, sampled_label])

        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(
            self,
            query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        # Extract the features of support and query images
        z_query = self.backbone.forward(query_images).float()
        z_query = gaussianization(z_query, self.beta)

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query.float(), self.prototypes.float())

        # Use it to compute classification scores
        scores = -dists

        return self.softmax_if_specified(scores)

    def set_save_features(self, save_features):
        self.save_features = save_features

def calculate_base_info(dataset="miniImagenet", base_info_path="."):
    # ---- Base class statistics
    base_means = []
    base_cov = []

    base_features_path = Path(base_info_path, "distribution_calibration/checkpoints/%s/base_features.plk" % dataset)
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            base_means.append(mean)
            base_cov.append(cov)

    # self.base_means = base_means
    # self.base_cov = base_cov
    return base_means, base_cov


def load_paper_backbone():
    num_classes = 3
    dataset = "miniImagenet"

    # model = wrn_model.wrn28_10(num_classes=200 , loss_type = 'dist')
    checkpoint_dir = "/home/leandro/Documentos/doutorado/dados/distribution_calibration/checkpoints-20221017T155939Z-001/checkpoints/%s/pretrained_model.tar" % dataset
    modelfile = checkpoint_dir

    model = wrn_model.wrn28_10(num_classes=num_classes)

    # model = model.cuda()
    cudnn.benchmark = True

    checkpoint = torch.load(modelfile)
    state = checkpoint['state']
    state_keys = list(state.keys())

    callwrap = False
    if 'module' in state_keys[0]:
        callwrap = True
    if callwrap:
        model = WrappedModel(model)
    model_dict_load = model.state_dict()
    model_dict_load.update(state)
    model.load_state_dict(model_dict_load)
    model.linear.L.weight = model.linear.L.weight.detach()

    # tmp = torch.load(checkpoint_dir)
    # start_epoch = tmp['epoch'] + 1
    # print("restored epoch is", tmp['epoch'])
    # state = tmp['state']
    # state_keys = list(state.keys())
    #
    # for i, key in enumerate(state_keys):
    #     if "feature." in key:
    #         newkey = key.replace("feature.",
    #                              "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
    #         state[newkey] = state.pop(key)
    #     else:
    #         state[key.replace("classifier.", "linear.")] = state[key]
    #         state.pop(key)
    #
    # model.load_state_dict(state)

    return model


def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query - base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0) + alpha

    return calibrated_mean, calibrated_cov


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def gaussianization(x, beta=0.5):
    """
        Based on GDC- Generalized Distribution Calibration for Few-Shot Learning (https://arxiv.org/abs/2204.05230)

        x_ = tukey(x, beta), if x >= 0 always
            yeo_johnson(x, beta), otherwise
    :param x:
    :param beta:
    :return:
    """

    def tukey(x, beta):
        if beta != 0:
            return torch.pow(x, beta)
        else:
            return torch.log(x)

    def yeojohnson_negative(x, beta):
        """
            y =
            -((-x + 1)**(2 - beta) - 1) / (2 - beta),  for x < 0, beta != 2
            -log(-x + 1),                                for x < 0, beta = 2
        """
        if beta != 2:
            return -((-x + 1) ** (2 - beta) - 1) / (2 - beta)
        else:
            return -torch.log(-x + 1)

    positive_index = (x >= 0)
    # Calculate Tukey’s Ladder of Powers for x>=0
    x[positive_index] = tukey(x[positive_index], beta)
    # Calculate Yeo-Johnson Transformation for x<0
    x[~positive_index] = yeojohnson_negative(x[~positive_index], beta)

    return x


def save_features(data, data_label, filename):
    data = data.cpu().numpy()
    data_label = data_label.cpu().numpy()

    filename_data = Path(filename)
    filename_label = Path(filename_data.parent, 'labels_' + filename_data.stem + filename_data.suffix)

    # Check if file exists
    if os.path.isfile(filename_data):
        with open(filename_data, 'rb') as handle:
            stored_data = pickle.load(handle)
            # Concatenate arrays
            data = np.concatenate((stored_data, data), axis=0)
        with open(filename_label, 'rb') as handle:
            stored_data_label = pickle.load(handle)
            # Concatenate arrays
            data_label = np.concatenate((stored_data_label, data_label), axis=0)


    with open(filename_data, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filename_label, 'wb') as handle:
        pickle.dump(data_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
