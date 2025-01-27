"""
Author: Leandro Lima
"""

# Load a pytroch pretreined model from a result directory, get the features from the fusion layer, apply t-sne and plot result.

import sys

sys.path.insert(0, '../')  # including the path to deep-tasks folder
sys.path.insert(0, '../my_models')  # including the path to my_models folder
# sys.path.insert(0, 'sab/')
sys.path.insert(0, 'pad/')

from constants import RAUG_PATH

sys.path.insert(0, RAUG_PATH)
from raug.loader import get_data_loader
from raug.eval import test_model
from my_model import set_model, get_norm_and_size
from raug.utils.loader import get_labels_frequency
from aug_pad import ImgEvalTransform

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
# from SkinLesion.my_thesis.utils import load_model, get_features

# Define the result directory and model path
model_dir = Path('/mnt/SSD-dados/LeandroLima/resultados/tese/model-base/poincare_feature_conv_concat/metablock_GenericTimm_pit_s_distilled_224_reducer_90_fold_4_17297960406985123')
_checkpoint_path = Path(model_dir, 'best-checkpoint/best-checkpoint.pth')

# Load the pretrained model
# model = load_model(model_path)
transform_param = get_norm_and_size(_model_name)

test_data_loader = get_data_loader (test_imgs_path, test_labels, test_meta_data, transform=ImgEvalTransform(*transform_param),
                                   batch_size=30, shuf=False, num_workers=16, pin_memory=True)

model = set_model(_model_name, len(_labels_name), neurons_reducer_block=_neurons_reducer_block,
                  comb_method=_comb_method, comb_config=_comb_config, pretrained=False)


# Get the features from the fusion layer
features, labels = get_features(model, model_dir)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features)

# Create a DataFrame for plotting
df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
df_tsne['label'] = labels

# Plot the t-SNE result
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], c=df_tsne['label'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Label')
plt.title('t-SNE of Fusion Layer Features')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()