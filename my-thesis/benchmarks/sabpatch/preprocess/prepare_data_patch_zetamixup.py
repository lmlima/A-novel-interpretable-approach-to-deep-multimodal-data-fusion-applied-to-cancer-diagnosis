# -*- coding: utf-8 -*-
"""
Author: Leandro Lima

Script to generate augmented data to train, validate and test NDB-22 patch dataset
"""

import sys

import torchvision

sys.path.insert(0, '../../..')  # including the path to deep-tasks folder
from constants import RAUG_PATH

sys.path.insert(0, RAUG_PATH)
import pandas as pd
import os
from raug.utils.loader import split_k_folder_csv, label_categorical_to_number
import pandas as pd
from pathlib import Path
from PIL import Image
from zetamixup import ZetaMixup
import torchvision.transforms as transforms
import torch
from torchvision.utils import save_image

BASE_PATH = "/home/leandro/Documentos/doutorado/dados/sab-patch/data_train_test/data"
data_file = os.path.join(BASE_PATH, "sabpatch_parsed_folders.csv")
label = "diagnostico"
gamma = 2.8
add_aug = 1000

path_train = Path(BASE_PATH, "train")
path_test = Path(BASE_PATH, "test")

aug_prefix = "aug"
path_aug = Path(BASE_PATH, aug_prefix)
path_aug.mkdir(parents=True, exist_ok=True)

train_dl = torchvision.datasets.ImageFolder(path_train)
test_dl = torchvision.datasets.ImageFolder(path_test)

data_df = pd.read_csv(data_file)
convert_tensor = transforms.Compose([
    transforms.PILToTensor()
])
convert_tensor2 = transforms.Compose([
    transforms.ToTensor()
])

# Set as not synthetic samples to all samples loaded from dataset
data_df["synthetic"] = False

label_dict = pd.Series(data_df[label].values, index=data_df["label_number"]).to_dict()
aug_data_list = []


def save_aug_images(x, fold, path_aug):
    images = [x_ for x_ in x]
    path_list = []

    for i, img in enumerate(images):
        file_path = Path(path_aug, F"aug_fold_{fold}_img_{i}.png")
        # Save file in relative_path
        save_image(img, file_path)
        path_list.append(file_path)

    return path_list


for fold in range(1, 6):
    data_fold_df = data_df[data_df["folder"] == fold]
    samples_per_label = data_fold_df[label].value_counts()

    # Warning: ZetaMixup can generate only n! different images, where n is number of all images.
    aug_needed = samples_per_label.max() - samples_per_label + add_aug
    aug_needed_dict = aug_needed.to_dict()

    print(F"Fold {fold}: {aug_needed}")
    print(F"Current samples: {samples_per_label}")

    # Load all images and its label in folder 'fold'
    samples = [
        (convert_tensor2(Image.open(Path(BASE_PATH, file["path"])).convert("RGB")), torch.tensor(file["label_number"]))
        for index, file in data_fold_df.iterrows()]
    x, y = map(list, zip(*samples))
    x = torch.stack(x).type(torch.float)
    y = torch.stack(y)

    # Calculate aug_need samples
    n_cls = len(data_fold_df["label_number"].value_counts().index)
    n_samples = len(data_fold_df)

    # Use ZetaMixup for data augmentation
    aug = ZetaMixup(n_cls, n_samples, gamma)

    # Generate n samples needed for each label
    new_x_list = []
    new_y_list = []
    for label_idx, label_name in label_dict.items():
        aug_needed_label_size = aug_needed_dict[label_name]
        print(F"Generating {aug_needed_label_size} for label {label_name} - {label_idx}")
        new_x, new_y = aug.generate_label(x, y, label_idx, aug_needed_label_size)
        if len(new_x) and len(new_y):
            new_x_list.append(new_x)
            new_y_list.append(new_y)

    # Augmented balanced dataset
    aug_x = torch.cat(new_x_list)
    aug_y = torch.cat(new_y_list)
    del new_x_list
    del new_y_list

    aug_y_label = [label_dict[i] for i in aug_y.tolist()]

    aug_img_path = save_aug_images(aug_x, fold, path_aug)
    aug_img_shortpath = [path.relative_to(BASE_PATH) for path in aug_img_path]
    del aug_x

    aug_size = len(aug_img_shortpath)
    data = {
        "path": aug_img_shortpath,
        label: aug_y_label,
        "label_number": aug_y,
        "folder": [fold] * aug_size,
        "synthetic": [True] * aug_size,
    }
    df = pd.DataFrame(data=data)
    aug_data_list.append(df)

aug_data_list.append(data_df)
aug_data = pd.concat(aug_data_list, axis=0, ignore_index=True)

# Save augmented dataset in sabpatch_parsed_folders_aug.csv
save_path = Path(BASE_PATH, "sabpatch_parsed_folders_aug.csv")
aug_data.to_csv(save_path, index=False)
