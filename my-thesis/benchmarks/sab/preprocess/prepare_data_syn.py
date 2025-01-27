"""
Author: Leandro Lima
"""

# Gerar csv

import pandas as pd
import os
import glob
from pathlib import Path
import shutil

folder = [1, 2, 3, 4, 5]
csv_filename = '/home/leandro/Documentos/doutorado/dados/sab/extra/displasia_cancer/sab_parsed_folders.csv'
base_path = "/tmp/gan/results"
dst_dir = '/tmp/gan/saida'

# Carcinoma_Leucoplasia
# labels = {
#     'BtoA': {
#         "label_number": 0,
#         "label": "CARCINOMA DE CÉLULAS ESCAMOSAS"
#     },
#     'AtoB': {
#             "label_number": 1,
#             "label": "LEUCOPLASIA"
#         },
# }
# label_col = "diagnostico_clinico"

# Displasia
# label_col = "ausencia_de_displasia"
# labels = {
#     'BtoA': {
#         "label_number": 0,
#         "label": "não"
#     },
#     'AtoB': {
#             "label_number": 1,
#             "label": "sim"
#         },
# }

# displasia_cancer
label_col = "displasia"
# Cancer - 1
# Presente - 2
# Ausente - 0

df = pd.read_csv(csv_filename)
df["synthetic"] = False

initial_n, _ = df.shape

gan_imgs = Path(dst_dir, "gan_imgs")
gan_imgs.mkdir(parents=True, exist_ok=True)

for current_path in glob.iglob(F"{base_path}/**/", recursive=False):
    print(F"Preparing {current_path}")
    current_AB = current_path.split("/")[-2]
    classA, classB = current_AB.split("__")

    df_label = df[[label_col, "label_number"]].value_counts().reset_index()

    labels = {
        'BtoA': {
            "label_number": df_label[df_label[label_col] == classA]["label_number"].values[0],
            "label": classA
        },
        'AtoB': {
            "label_number": df_label[df_label[label_col] == classB]["label_number"].values[0],
            "label": classB
        },
    }

    for i in folder:
        gan_img_path = F'{current_path}/{str(i)}/test'

        filesB = glob.glob(F'{gan_img_path}/AtoB*.png')
        filesA = glob.glob(F'{gan_img_path}/BtoA*.png')

        file_listA = list(map(os.path.split, filesA))
        file_listB = list(map(os.path.split, filesB))
        syn_folder = folder[i-2]

        for file in file_listA:
            new_row = {
                "path": F"{current_AB}_{i}_{file[-1]}",
                "folder": syn_folder,
                "label_number": labels["AtoB"]["label_number"],
                label_col: labels["AtoB"]["label"],
                "synthetic": True,
            }
            df = df.append(new_row, ignore_index=True)

            shutil.copy(Path(*file), Path(gan_imgs, new_row["path"]))

        for file in file_listB:
            new_row = {
                "path": F"{current_AB}_{i}_{file[-1]}",
                "folder": syn_folder,
                "label_number": labels["BtoA"]["label_number"],
                label_col: labels["BtoA"]["label"],
                "synthetic": True,
            }
            df = df.append(new_row, ignore_index=True)

            shutil.copy(Path(*file), Path(gan_imgs, new_row["path"]))

total_n, _ = df.shape
print(F"Initial instances: {initial_n}\nSynthetic instances: {total_n-initial_n}\nTotal instances: {total_n}")
df.to_csv(F"{dst_dir}/sab_parsed_folders.syn.csv", index=False)
