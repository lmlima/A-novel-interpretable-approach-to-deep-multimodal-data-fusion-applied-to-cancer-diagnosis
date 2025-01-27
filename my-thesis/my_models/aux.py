# -*- coding: utf-8 -*-
"""
Autor: Leandro Lima
"""
import pandas as pd
import re


def revert_ohe(x, col_sparse, col_dense):
    # x = train_csv_folder
    col_feat = x.columns

    for col in col_sparse:
        # feat_names = [k.replace(f"{col}_", "") for k in col_feat if col in k]
        feat_names = [k for k in col_feat if k.startswith(col)]

        x_ohe = x[feat_names]

        def get_label(row):
            for c in x_ohe.columns:
                if row[c] == 1:
                    # return c.replace(f"{col}_", "")
                    return re.sub(r"^" + col + "_", "", c)

        # Add
        x_new = x_ohe.apply(get_label, axis=1).to_frame(name=col)
        x = pd.concat([x, x_new], axis="columns")

    # Replace string labels with numbers
    for col in col_sparse:
        x[col] = x[col].astype("category").cat.codes
    x = x[col_sparse + col_dense]

    return x
