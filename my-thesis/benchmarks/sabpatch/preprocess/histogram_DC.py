"""
Author: Leandro Lima
"""

import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def show_hist(fn, remove_zeros=False, title=''):
    fn = Path(fn)
    with open(fn, 'rb') as handle:
        stored_data = pickle.load(handle)
    with open(Path(fn.parent, "labels_" + fn.name), 'rb') as handle:
        stored_data_labels = pickle.load(handle)

    # df = pd.DataFrame(stored_data.flatten())
    df = pd.DataFrame(stored_data)
    if remove_zeros:
        df = df[df != 0]

    df['label'] = stored_data_labels

    df = pd.melt(df, id_vars=['label'])

    print(df.shape)

    g = sns.displot(data=df, x="value", col="label", kde=True, bins=20, line_kws={"linewidth": 3})
    g.fig.suptitle(title)
    g.fig.tight_layout()
    plt.show()

    g = sns.displot(data=df, x="value", hue="label", kind="kde", linewidth=1)
    g.fig.suptitle(title)
    g.fig.tight_layout()
    plt.show()

show_hist("/tmp/features/640/support_orig.pkl", remove_zeros=False, title="Original Support Data")
show_hist("/tmp/features/640/support_gauss.pkl", remove_zeros=False, title="Gaussianizated Support Data")

show_hist("/tmp/features/640/support_orig.pkl", remove_zeros=True, title="Original Support Data")
show_hist("/tmp/features/640/support_gauss.pkl", remove_zeros=True, title="Gaussianizated Support Data")

