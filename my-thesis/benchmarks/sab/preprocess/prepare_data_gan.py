"""
Author: Leandro Lima
"""

# Parear imagens cirurgicas com as histopatologicas baseado no csv

import pandas as pd
import shutil
import os
from pathlib import Path
import itertools
import numpy as np
from PIL import Image



folds = [1, 2, 3, 4, 5]

mixup_aug = True

# csv_filename = '/home/leandro/Documentos/doutorado/dados/sab/sab_parsed_folders.csv'
csv_filename = '/tmp/teste/sab_parsed_folders.csv'

dst_dir = '/tmp/teste/resultado'
#
base_path = '/tmp/teste'
orig_img_histo = Path(base_path, 'imgs')
# label = "diagnostico_clinico"
label = "displasia"


# label_number

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def copy_files(orig, dest):
    shutil.copy(Path(orig_img_histo, orig), dest)

def load_img(img_filename):
    file_path = Path(orig_img_histo, img_filename)
    image = Image.open(file_path).convert("RGB")
    return image

def mixup_data(all_imgs, dst_path, n=6, alpha=1.0):
    ''' Returns mixed inputs '''

    batch_size = all_imgs.size

    for k in range(n):
        index = np.random.permutation(batch_size)
        for i1, i2 in enumerate(index):
            x1 = all_imgs.iloc[i1]
            x2 = all_imgs.iloc[i2]

            mixed_x = mixup(x1, x2, alpha)
            mixup_filename = Path(Path(x1).stem + F"_mixup_{k}_{i1}_{i2}" + Path(x1).suffix)
            mixed_x.save(Path(dst_path, mixup_filename))

    # return mixed_x


def mixup(x1, x2, alpha=1.0):
    ''' Returns mixed inputs '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    x1_img = load_img(x1)
    x2_img = load_img(x2)

    # mixed_x = lam * x1_img + (1 - lam) * x2_img
    mixed_x = Image.blend(x2_img, x1_img, lam)

    # We always use samples of same class
    # y_a, y_b = y, y[index]
    return mixed_x

def dataset_AB(df, folds, class_A, class_B):
    comb_str = F"{class_A}__{class_B}"

    for folder in folds:

        val_folder = folds[folder-2]
        val_csv_folder = df[(df['folder'] == val_folder)]

        # Exclude classifier validation folder and GAN validation folder
        train_csv_folder = df[~df['folder'].isin([folder, val_folder])]


        # path	diagnostico_clinico	folder	label_number
        for index, row in train_csv_folder.iterrows():
            if row[label] == class_A:
                dst_path = Path(dst_dir, comb_str, str(folder), "trainA")
                os.makedirs(dst_path, exist_ok=True)
                copy_files(row["path"], dst_path)
            elif row[label] == class_B:
                dst_path = Path(dst_dir, comb_str, str(folder), "trainB")
                os.makedirs(dst_path, exist_ok=True)
                copy_files(row["path"], dst_path)

        for index, row in val_csv_folder.iterrows():
            if row[label] == class_A:
                dst_path = Path(dst_dir, comb_str, str(folder), "testA")
                os.makedirs(dst_path, exist_ok=True)
                copy_files(row["path"], dst_path)

            elif row[label] == class_B:
                dst_path = Path(dst_dir, comb_str, str(folder), "testB")
                os.makedirs(dst_path, exist_ok=True)
                copy_files(row["path"], dst_path)

        if mixup_aug:
            mixup_data(val_csv_folder[val_csv_folder[label] == class_A]["path"], Path(dst_dir, comb_str, str(folder), "testA"))
            mixup_data(val_csv_folder[val_csv_folder[label] == class_B]["path"], Path(dst_dir, comb_str, str(folder), "testB"))



df = pd.read_csv(csv_filename)
# df_label = df[['diagnostico_clinico', "label_number"]].value_counts()
# # df1.to_frame("group").astype(str)
#
# idx = pd.IndexSlice
# si = df_label.index.get_level_values(1).to_series().apply(lambda x: chr(ord('a') + x)).str.upper()
#
# df_label.loc[idx[:, si.index]] = si.values
#
# df_label = df_label.to_frame("group").reset_index()

class_combinations = list(itertools.combinations(df[label].unique(), 2))

for class_A, class_B in class_combinations:
    dataset_AB(df, folds, class_A, class_B)
