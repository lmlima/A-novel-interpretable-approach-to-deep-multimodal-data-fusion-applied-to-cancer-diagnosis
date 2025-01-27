"""
Author: Leandro Lima
"""

import pandas as pd
import shutil
import os
from pathlib import Path
from dateutil.relativedelta import relativedelta
import numpy as np


DATA_PATH = '/tmp/sab'
DATA_NAME = "sab_data_full.xlsx"
# IMG_PATH = '/tmp/sab/images/'
IMG_PATH = "/home/leandro/Documentos/doutorado/dados/sab/histopatologico/completa/imagensHistopatologica_completa/imagensHistopatologica"

DEST_PATH = '/tmp/sab/output'

display_stats = True


def copy_images(images_list):
    #create directory if not existent
    image_dest_path = Path(DEST_PATH, "images")
    image_dest_path.mkdir(parents=True, exist_ok=True)

    # copy a list of images filenames to a new folder
    for path_orig, new_path in images_list:
        # file = Path(image)
        # file_dest = Path(F"{idx:04}{file.suffix}")
        shutil.copy(
            Path(IMG_PATH, path_orig),
            Path(image_dest_path, new_path),
        )


def rename_filename(s):
    # rename filename to 4 digits
    file = Path(s.path_orig)
    return F"{s.name:04}{file.suffix}"


def calc_age(s):
    # calculate age from date s.data_nascimento to date s.data_atendimento_ini
    # print(F"{s.data_nascimento} -> {s.data_procedimento}")
    try:
        date_init = pd.to_datetime(s.data_procedimento)
        date_end = pd.to_datetime(s.data_nascimento)
        return relativedelta(date_init, date_end).years
    except:
        return None



#####
# read dataframe from xlsx
df = pd.read_excel(
    Path(DATA_PATH, DATA_NAME),
    keep_default_na=True,
)

# rename column "path" to "path_orig"
df.rename(columns={"path": "path_orig"}, inplace=True)
df.rename(columns={"grau de displasia": "dysplasia_severity"}, inplace=True)
df.rename(columns={"displasia": "diagnosis"}, inplace=True)
df.rename(columns={"sexo": "gender"}, inplace=True)
# df.rename(columns={"exp_sol": "sun_exposure"}, inplace=True)
df.rename(columns={"cor_pele": "skin_color"}, inplace=True)
df.rename(columns={"localizacao": "localization"}, inplace=True)
df.rename(columns={"tamanho_maior": "larger_size"}, inplace=True)

df.rename(columns={"uso_cigarro": "tobacco_use"}, inplace=True)
df.rename(columns={"uso_bebida": "alcohol_consumption"}, inplace=True)

# fix diagnosis
date_diagnosis = {
    "na": "OSCC",
    "sim": "Leukoplakia with dysplasia",
    "não": "Leukoplakia without dysplasia"
}
df.diagnosis.replace(date_diagnosis, inplace=True)

# Fix localization
df.localization.replace(
    {
        "LINGUA": "LÍNGUA",
        "PALATO ": "PALATO",
    },
    inplace=True
)
#  Grouping
df.localization.replace(
    {
        "PALATO MOLE": "PALATO",
        "PALATO DURO": "PALATO",
        "BORDA DE LÍNGUA": "LÍNGUA",
        "ASSOALHO DE LINGUA": "LÍNGUA",
        "GENGIVA SUPERIOR": "GENGIVA",
        "FUNDO DE VESTÍBULO": "GENGIVA",
    },
    inplace=True
)

# create new filename and copy it
df["path"] = df.apply(rename_filename, axis=1)
copy_images(list(zip(df["path_orig"], df["path"])))

# Fix data_nascimento
df_nasc = pd.read_csv(F"{DATA_PATH}/nasc.csv")
df = pd.merge(df, df_nasc, on="paciente_id", how="left").drop(columns="data_nascimento_x").rename(columns={"data_nascimento_y": 'data_nascimento'})

# create a column with age in years
df["age"] = df.apply(calc_age, axis=1)
bins = [0, 40, 60, np.inf]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=False)

# Translate columns
dic_rename_uso = {
    "X": "Not informed",
    "S": "Yes",
    "N": "No",
    "P": "Former"
}
df["tobacco_use"].replace(dic_rename_uso, inplace=True)
df["alcohol_consumption"].replace(dic_rename_uso, inplace=True)

dic_rename_skin = {
    "B": "White",
    "P": "Brown",
    "N": "Black",
    "O": "Not informed",
}
df["skin_color"].replace(dic_rename_skin, inplace=True)

dic_rename_localization = {
    "LÍNGUA": "Tongue",
    "GENGIVA": "Gingiva",
    "ASSOALHO DE BOCA": "Floor of mouth",
    # "FUNDO DE VESTÍBULO": "Buccal sulculs",
    "LABIO": "Lip",
    "MUCOSA JUGAL": "Buccal mucosa",
    "PALATO": "Palate",
    "MUCOSA LABIAL": "Labial mucosa",
}
df["localization"].replace(dic_rename_localization, inplace=True)

dic_rename_severity = {
    "leve": "Mild",
    "moderada": "Moderate",
    "severa": "Severe",
}
df["dysplasia_severity"].replace(dic_rename_severity, inplace=True)

# sun_exposure transform
# create a list of our conditions
conditions = [
    (df['exp_sol'] == -1),
    (df['exp_sol'] == 0),
    (df['exp_sol'] > 0),

    ]

# create a list of the values we want to assign for each condition
values = ['Not informed', 'No', 'Yes']

# create a new column and use np.select to assign values to it using our lists as arguments
df["sun_exposure"] = np.select(conditions, values)


#
# Task II
# create a list of our conditions
conditions = [
    (df['diagnosis'] != "OSCC"),
    (df['diagnosis'] == "OSCC")
    ]

# create a list of the values we want to assign for each condition
values = ['Leukoplakia', 'OSCC']

# create a new column and use np.select to assign values to it using our lists as arguments
df["TaskII"] = np.select(conditions, values)

# Task III
# create a list of our conditions
conditions = [
    ((df['diagnosis'] != "OSCC") & (df["dysplasia_severity"].isna())),
    ((df['diagnosis'] == "OSCC") | (df["dysplasia_severity"].notna()))
    ]

# create a list of the values we want to assign for each condition
values = ['Absence', 'Presence']

# create a new column and use np.select to assign values to it using our lists as arguments
df["TaskIII"] = np.select(conditions, values)

# Task IV
df["TaskIV"] = df["diagnosis"]

df = df.reset_index().rename(columns={"index": 'public_id'})

# Sun_exposure in hours, -1 is not informed.
# Filter only public columns
public_columns = ["public_id", "path", "localization", "larger_size", "tobacco_use", "alcohol_consumption", "sun_exposure", "gender", "skin_color", "age_group", "diagnosis", "dysplasia_severity", "TaskII", "TaskIII", "TaskIV"]

df_public = df[public_columns].copy(deep=True)

df_public.to_csv(Path(DEST_PATH, "ndb-ufes.csv"), index=False)
df_public.to_excel(Path(DEST_PATH, "ndb-ufes.xlsx"), index=False)

####
# Display stats
def print_stats(cur_df):
    print(F"Count: {len(cur_df)}")

    print("Gender")
    print(cur_df["gender"].value_counts().sort_index())

    print("Age group")
    print(cur_df["age_group"].value_counts().sort_index())

    print("Lesion location")
    print(cur_df["localization"].value_counts().sort_index())

    print("Cigarette use")
    print(cur_df["tobacco_use"].value_counts().sort_index())

    print("Alcohol use")
    print(cur_df["alcohol_consumption"].value_counts().sort_index())

    print("Sun exposure")
    print(cur_df["sun_exposure"].value_counts().sort_index())

    print("Lesion size")
    print(cur_df["larger_size"].describe().T[["mean", "std", "min", "max"]])

    print("Skin Color")
    print(cur_df["skin_color"].value_counts().sort_index())

    print("")


def print_segmented(dataframe, label):
    print(label)
    for i in dataframe[label].unique():
        cur_df = dataframe[dataframe[label] == i]

        print(i.replace("\n", ""))
        print_stats(cur_df)


if display_stats:
    print("All")
    print_stats(df)
    print("-" * 30)

    print_segmented(df_public, "TaskII")
    print("-" * 30)

    print_segmented(df_public, "TaskIII")
    print("-" * 30)

    print_segmented(df_public, "TaskIV")
    print("-" * 30)


print("Fim")