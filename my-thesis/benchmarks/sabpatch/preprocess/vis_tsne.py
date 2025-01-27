"""
Author: Leandro Lima
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE

perplexity = 30

label_map = {
    "1":"OSCC",
    "0":"Absence of dysplasia",
    "2":"Presence of dysplasia",
}

# Load dataframe from pickle
df_test = pd.read_pickle('/home/leandro/Documentos/doutorado/docs/2022-2/artigo_patches/test_features.pkl')
df_train = pd.read_pickle('/home/leandro/Documentos/doutorado/docs/2022-2/artigo_patches/train_features.pkl')
df_train.columns = df_train.columns.astype(str)
df_test.columns = df_test.columns.astype(str)


df = pd.concat([df_train, df_test])

# X = df.drop(columns=["target"])
X = df.drop(columns=["predicted", "target"])


tsne = TSNE(n_components=2, perplexity=perplexity)
# tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=perplexity)

z = tsne.fit_transform(X)

df_tsne = pd.DataFrame()
df_tsne["Target"] = df["target"].astype(int).astype(str).map(label_map)
df_tsne["Component 1"] = z[:,0]
df_tsne["Component 2"] = z[:,1]

tsne_plot = sns.scatterplot(y="Component 1", x="Component 2", hue=df_tsne["Target"].tolist(),
                palette=sns.color_palette("hls", 3),
                data=df_tsne)
tsne_plot.set(title="Dataset data T-SNE projection after fine-tuning")

fig = tsne_plot.get_figure()
fig.savefig(f"/home/leandro/Documentos/doutorado/docs/2022-2/artigo_patches/tsne_perp{perplexity}.png", dpi = 600)