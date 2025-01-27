"""
Author: Leandro Lima
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate result table in latex",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("src", help="Source location")
args = parser.parse_args()
config = vars(args)

# file = "/home/leandro/Documentos/doutorado/resultados/SkinCancer/sab/pretrained/Dispasia_Cancer/batch30_subpatch_18f/sab/result_desc_small.csv"
#file = "/home/leandro/Documentos/doutorado/docs/resultados/sab/IM2023/TaskII/result_desc_small.csv"
file = config["src"]

def calc_PercIncrease(start, final):
    return (final - start) / abs(start) * 100

def calc_PercDiff(start, final):
    return (final - start) * 100

df = pd.read_csv(file, header=[0, 1, 2], index_col=[0, 1], skipinitialspace=True)

# Get level1 index names
level1 = df.index.get_level_values(1).unique()

# Get level1 index names as index in a dictionary and dictionary value as the index name alphabetic order. Add 'None' as index 0 .
custom_dict = {level1[i]: i+1 for i in range(len(level1))}
if 'None' in level1:
    custom_dict['None'] = 0
    
# custom_dict = {'None': 0, 'concat': 1, 'mat': 2, 'metablock': 3, 'metanet': 4}
metrics = ['baccuracy', 'precision', 'recall', 'auc']

df_small = df.loc(axis=1)[pd.IndexSlice[metrics, ['mean', 'std'], :]].sort_index(level=1, sort_remaining=True).sort_index(level=1, key=lambda x: x.map(custom_dict), sort_remaining=True).swaplevel(0, 1)

for metric in metrics:
    print(F"Max: {metric}")
    print(df_small[df_small[metric]['mean'].values == df_small[metric]['mean'].values.max()][metric]['mean'])
    print("-"*10)

    max_All = df_small[metric]['mean'].values.max()
    max_Image = (df_small[df_small.index.get_level_values(0).isin(['None'])])[metric]['mean'].max().values[0]

    print(F"Max All: {max_All}, Max Image: {max_Image}")
    print(F"Percentage Increase: {calc_PercIncrease(max_Image, max_All):.2f} %")
    print(F"Percentage Diff: {calc_PercDiff(max_Image, max_All):.2f} pp")

    print("-"*20)

    df_small['str_'+metric] = df_small.apply(lambda row: F"{float(row[metric]['mean'].values):.4f} ({float(row[metric]['std'].values):.4f})", axis=1)

df_small = df_small.drop(columns=metrics)

print(df_small.to_latex(index=True))

