#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Leandro Lima


This file implements the methods and functions to load data as a PyTorch dataset

If you find any bug or have some suggestion, please, email me.
"""
import torch
from torch import Tensor
from typing import List, Tuple

from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
from easyfsl.samplers import TaskSampler
from easyfsl.datasets import FewShotDataset


class MyFSLDataset(FewShotDataset):
    """
    This is the standard way to implement a dataset pipeline in PyTorch. We need to extend the torch.utils.data.Dataset
    class and implement the following methods: __len__, __getitem__ and the constructor __init__
    """

    def __init__(self, imgs_path, labels, meta_data=None, transform=None):
        """
        The constructor gets the images path and their respectively labels and meta-data (if applicable).
        In addition, you can specify some transform operation to be carry out on the images.

        It's important to note the images must match with the labels (and meta-data if applicable). For example, the
        imgs_path[x]'s label must take place on labels[x].

        Parameters:
        :param imgs_path (list): a list of string containing the image paths
        :param labels (list) a list of labels for each image
        :param meta_data (list): a list of meta-data regarding each image. If None, there is no information.
        Defaul is None.
        :param transform (torchvision.transforms.Compose): transform operations to be carry out on the images
        """

        super().__init__()
        self.imgs_path = imgs_path
        self.labels = labels
        self.meta_data = meta_data

        # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll get
        # an exception
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.imgs_path)

    def get_labels(self):
        return self.labels

    def __getitem__(self, item):
        """
        It gets the image, labels and meta-data (if applicable) according to the index informed in `item`.
        It also performs the transform on the image.

        :param item (int): an index in the interval [0, ..., len(img_paths)-1]
        :return (tuple): a tuple containing the image, its label and meta-data (if applicable)
        """

        image = Image.open(self.imgs_path[item]).convert("RGB")

        # Applying the transformations
        image = self.transform(image)

        img_id = self.imgs_path[item].split('/')[-1].split('.')[0]

        if self.meta_data is None:
            meta_data = []
        else:
            meta_data = self.meta_data[item]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        # return image, labels, meta_data, img_id
        return image, labels, meta_data


def get_data_loader(imgs_path, labels, meta_data=None, transform=None, batch_transform=None,
                    num_workers=4,
                    pin_memory=True, fsl_params=None, returnDataset=False):
    """
    This function gets a list og images path, their labels and meta-data (if applicable) and returns a DataLoader
    for these files. You also can set some transformations using torchvision.transforms in order to perform data
    augmentation. Lastly, params is a dictionary that you can set the following parameters:
    batch_size (int): the batch size for the dataset. If it's not informed the default is 30
    shuf (bool): set it true if wanna shuffe the dataset. If it's not informed the default is True
    num_workers (int): the number thread in CPU to load the dataset. If it's not informed the default is 0 (which


    :param returnDataset: return dataset. Default is False.
    :param imgs_path (list): a list of string containing the images path
    :param labels (list): a list of labels for each image
    :param meta_data (list, optional): a list of meta-data regarding each image. If it's None, it means there's
    no meta-data. Default is None
    :param transform (torchvision.transforms, optional): use the torchvision.transforms.compose to perform the data
    augmentation for the dataset. Alternatively, you can use the jedy.pytorch.utils.augmentation to perform the
    augmentation. If it's None, none augmentation will be perform. Default is None
    :param batch_transform (transforms, optional): use the transforms to perform the data
    augmentation for the dataset as a batch. Default is None
    :param batch_size (int): the batch size. If the key is not informed or params = None, the default value will be 30
    :param shuf (bool): if you'd like to shuffle the dataset. If the key is not informed or params = None, the default
    value will be True
    :param num_workers (int): the number of threads to be used in CPU. If the key is not informed or params = None, the
    default value will be  4
    :param pin_memory (bool): set it to True to Pytorch preload the images on GPU. If the key is not informed or
    params = None, the default value will be True
    :return (torch.utils.data.DataLoader): a dataloader with the dataset and the chose params
    """

    dt = MyFSLDataset(imgs_path, labels, meta_data, transform)

    collate_fn = None
    batch_sampler = None

    if fsl_params is not None:
        n_way, n_shot, n_query, n_tasks = fsl_params
        sampler = MetadataTaskSampler(
            dt, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks
        )
        batch_sampler = sampler
        collate_fn = sampler.episodic_collate_fn

    dl = data.DataLoader(dataset=dt, num_workers=num_workers,
                         pin_memory=pin_memory,
                         collate_fn=get_collate(collate_fn, batch_transform),
                         batch_sampler=batch_sampler,
                         )
    if returnDataset:
        return dl, dt
    else:
        return dl


def get_collate(collate_fn=None, batch_transform=None):
    def mycollate(batch):
        if collate_fn is not None:
            collated = collate_fn(batch)
        else:
            collated = torch.utils.data.dataloader.default_collate(batch)

        if batch_transform is not None:
            collated = batch_transform(collated)
        return collated

    return mycollate


class MetadataTaskSampler(TaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        true_class_ids = list({x[1] for x in input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        all_meta = torch.tensor(
            [x[2] for x in input_data]
        )
        all_meta = all_meta.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_meta.shape[1:])
        )

        if all_meta.shape[2] != 0:
            support_images_meta = all_meta[:, : self.n_shot].reshape(
                    (-1, *all_meta.shape[2:])
                )
            query_images_meta = all_meta[:, self.n_shot:].reshape((-1, *all_meta.shape[2:]))
        else:
            support_images_meta = []
            query_images_meta = []

        support_images = (
            all_images[:, : self.n_shot].reshape(
                (-1, *all_images.shape[2:])
            ),
            support_images_meta
        )

        query_images = (
            all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:])),
            query_images_meta
        )
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )
