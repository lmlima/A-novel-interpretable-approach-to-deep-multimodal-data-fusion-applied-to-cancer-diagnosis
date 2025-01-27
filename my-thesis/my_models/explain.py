"""
Author: Leandro Lima
"""


import sys
from typing import Generic

from constants import RAUG_PATH

sys.path.insert(0, RAUG_PATH)

import numpy as np
from auto_shap.auto_shap import produce_shap_output_with_agnostic_explainer, generate_shap_summary_plot
import pandas as pd
from pathlib import Path

from MultiClassifier import MultiClassifier
from GenericTimm import GenericTimm


def select_shap_model(model_wrap):
    if isinstance(model_wrap, MultiClassifier):
        return model_wrap.model,"MultiClassifier"
    elif isinstance(model_wrap, GenericTimm):
        if model_wrap.outter_classifier is None:
            raise Exception("No outter classifier defined.")
        return model_wrap.outter_classifier, "GenericTimm"
    else:
        return NotImplemented

def prepare_data(model_wrap, data_loader, device):
    if isinstance(model_wrap, MultiClassifier):
        return model_wrap.prepare_data(data_loader)
    elif isinstance(model_wrap, GenericTimm):
        return prepare_data_local(model_wrap, data_loader, device)
    else:
        return NotImplemented

def select_fusion_type(model_wrap):
    if model_wrap.late_fusion:
        return "late"
    elif model_wrap.use_outter_classifier and (not model_wrap.late_fusion) :
        return "mixed"
    elif (not model_wrap.use_outter_classifier) and (not model_wrap.late_fusion):
        return "joint"
    else:
        return NotImplemented

def prepare_data_local(model_wrap, data_loader, device):
    # TODO: Load image and extract features on backbone model.
    model_wrap.eval()
    fusion_type = select_fusion_type(model_wrap)

    for batch_idx, (img, target, meta_data, sample_name) in enumerate(data_loader):
        img = img.to(device)
        # meta_data = meta_data.to(self.device)
        if fusion_type == "late":
            feat_out = model_wrap.backbone_features(img).detach().cpu().numpy()
            feat_out = feat_out.squeeze((-1, -2))
            try:
                concat_features = np.concatenate((feat_out, meta_data), axis=1)
                concat_samples = np.concatenate((concat_samples, concat_features), axis=0)
                concat_targets = np.concatenate((concat_targets, target), axis=0)
                concat_name = np.concatenate((concat_name, sample_name), axis=0)
            except:
                concat_features = np.concatenate((feat_out, meta_data), axis=1)
                concat_samples = concat_features
                concat_targets = target
                concat_name = sample_name
        elif fusion_type == "mixed":
            feat_out = model_wrap.backbone_features(img)
            feat_out = model_wrap.reducer_block(feat_out.squeeze(-1).squeeze(-1))
            feat_out = model_wrap.classifier(feat_out).detach().cpu().numpy()
            # feat_out (batch_size, num_classes)
            try:
                concat_features = np.concatenate((feat_out, meta_data), axis=1)
                concat_samples = np.concatenate((concat_samples, concat_features), axis=0)
                concat_targets = np.concatenate((concat_targets, target), axis=0)
                concat_name = np.concatenate((concat_name, sample_name), axis=0)
            except:
                concat_features = np.concatenate((feat_out, meta_data), axis=1)
                concat_samples = concat_features
                concat_targets = target
                concat_name = sample_name
        elif fusion_type == "joint":
            meta_data = meta_data.to(device)
            feat_out = model_wrap.backbone_fusion(img, meta_data).detach().cpu().numpy()
            feat_out = feat_out.squeeze((-1, -2))
            try:
                concat_samples = np.concatenate((concat_samples, feat_out), axis=0)
                concat_targets = np.concatenate((concat_targets, target), axis=0)
                concat_name = np.concatenate((concat_name, sample_name), axis=0)
            except:
                concat_samples = feat_out
                concat_targets = target
                concat_name = sample_name

    return concat_samples, concat_targets, concat_name


def shap_explain(model_wrap, test_data_loader, columns=None, labels=None, device=None, save_path="/tmp/shap_output"):
    """
        Generate a summary of shap analysis.
    Args:
        model_wrap: Model wrap.
        test_data_loader: Dataloader of analysed data.
        columns: List of columns as input of model analysed by shap
        labels: List of model output labels.
        device: Device where model_wrap is allocated. Must be set when using a GenericTimm model.
        save_path (object): Path to save shap results.


    """
    Path(save_path, "plots").mkdir(parents=True, exist_ok=True)

    model, model_type = select_shap_model(model_wrap)
    test_concat_features, y_true, test_samples_name = prepare_data(model_wrap, test_data_loader, device)

    df_test = pd.DataFrame(test_concat_features)
    if (columns is not None) and (len(df_test.columns) == len(columns)):
        df_test.columns = columns
    else:
        df_test.columns = [f'feature_{i}' for i in range(df_test.shape[1])]
    # y_pred = self.model.predict_proba(test_concat_features)
    # produce_shap_values_and_summary_plots(model=self.model, x_df=df_test, save_path=save_path, use_agnostic=True)

    # import shap
    # explainer = shap.TreeExplainer(self.model)
    # shap_values = explainer.shap_values(df_test)
    # shap.summary_plot(shap_values, df_test.values, plot_type="bar", features_names=df_test.columns)
    ####
    if model_type == "MultiClassifier":
        shap_values, shap_expected_value, global_shap_df = produce_shap_output_with_agnostic_explainer(model, df_test,
                                                                                                   boosting_model=True,
                                                                                                   regression_model=False,
                                                                                                   linear_model=False,
                                                                                                   return_df=False)
    elif model_type == "GenericTimm":
        shap_values, shap_expected_value, global_shap_df = produce_shap_output_with_agnostic_explainer(model, df_test,
                                                                                                   n_jobs=1,
                                                                                                   boosting_model=False,
                                                                                                   regression_model=False,
                                                                                                   linear_model=False,
                                                                                                   return_df=False)
    else:
        raise NotImplemented

    rows, col = shap_values.shape
    classes = len(labels)
    col = col // classes
    shap_values_reshaped = shap_values.reshape(rows, col, classes)

    # Split the reshaped array along the second axis (axis=1) into 6 parts
    split_shap_values = np.split(shap_values_reshaped, classes, axis=-1)

    # Save shap information
    global_shap_df.to_csv(Path(save_path, 'global_shap_values.csv'), index=False)
    with open(Path(save_path, 'shap_expected_value.txt'), 'w') as f:
        f.write(str(shap_expected_value))

    with open(Path(save_path, 'columns_name.txt'), 'w') as f:
        f.write(str(columns))

    with open(Path(save_path, 'labels_name.txt'), 'w') as f:
        f.write(str(labels))

    # Create a list to hold the DataFrames
    dfs = []

    # For each split array, convert it into a DataFrame
    for i, split in enumerate(split_shap_values):
        # Flatten the array from shape (383, 1, 87) to (383, 87)
        flattened_split = split.reshape(rows, -1)

        # Create a DataFrame and assign column names like 'feature_x'
        df = pd.DataFrame(flattened_split)
        df.columns = [f'feature_{i * col + j}' for j in range(col)]

        # Save shap values
        global_shap_df.to_csv(Path(save_path, f"class_{labels[i]}_local_shap_values.csv"), index=False)

        # Generate shap plots
        generate_shap_summary_plot(df, x_df=df_test, plot_type="dot", save_path=save_path,
                                   file_prefix=f"class_{labels[i]}")
        generate_shap_summary_plot(df, x_df=df_test, plot_type="bar", save_path=save_path,
                                   file_prefix=f"class_{labels[i]}")

        # Append the DataFrame to the list
        dfs.append(df)

