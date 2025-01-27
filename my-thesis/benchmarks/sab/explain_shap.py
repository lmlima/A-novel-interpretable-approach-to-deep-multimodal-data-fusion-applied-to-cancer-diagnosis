"""
Author: Leandro Lima
"""


import sys

my_models_path = "/home/leandro/PycharmProjects/SkinLesion/SkinLesion/my-thesis/my_models"
sys.path.insert(0, '../../')  # including the path to deep-tasks folder
sys.path.insert(0, my_models_path)  # including the path to my_models folder
from constants import RAUG_PATH

sys.path.insert(0, RAUG_PATH)
from raug.loader import get_data_loader
from raug.eval import test_model
from my_model import set_model, get_norm_and_size
from raug.utils.loader import get_labels_frequency
from aug_sab import ImgEvalTransform, ImgEvalReverseTransform
import torch

import pandas as pd
from pathlib import Path
import re
import json

import shap
import numpy as np
import matplotlib.pyplot as plt

import copy
from aux import revert_ohe
import pickle


# path_dir = "/home/leandro/Documentos/doutorado/tmp/shap/models/without_data"
path_dir = "/home/leandro/Documentos/doutorado/tmp/shap/models/sab"
path_dir = "/home/leandro/Documentos/doutorado/tmp/shap/models/sab/dev_only"

new_ds_path = "/home/leandro/Documentos/doutorado/dados/sab/NDB-UFES"
# new_ds_path = "/home/leandro/Documentos/doutorado/dados/sab/diag_definitivo/extra/displasia_cancer/18_features"

new_img_path = "/home/leandro/Documentos/doutorado/dados/sab/NDB-UFES/images"
# new_img_path = "/home/leandro/Documentos/doutorado/dados/sab/histopatologico/imagensHistopatologica"

output_path = "/tmp/pad/results/shap"

# Search existing models
path = Path(path_dir)
p_list = [i for i in path.glob('*') if i.is_dir()]

regex = r'(?P<fusion>[A-Za-z0-9-]+)_(?P<model>[A-Za-z0-9-_]+)_reducer_(?P<reducer>\d+)_fold_(?P<fold>\d+)_\d+'

def class_labels(preds, class_names):
    def is_pred(i, preds_values):
        return i == np.argmax(preds_values)

    return [f'{class_names[i]} ({preds[i].round(2):.2f})' if not is_pred(i, preds) else fr'{class_names[i]} ($\bf{{{preds[i].round(2):.2f}}}$)' for i in range(len(class_names))]
    # return [f'{class_names[i]} ({preds[i].round(2):.2f})' if is_pred(i, preds) else fr'{class_names[i]} ' + fr'\bf{(' + '{:.2f}'.format(preds[i].round(2)) + fr')}' for i in range(len(class_names))]


# TODO: Grouping categorical features as in https://github.com/slundberg/shap/issues/397

for item in p_list:
    _check_base_path = item
    _checkpoint_path = Path(_check_base_path, "best-checkpoint/best-checkpoint.pth")

    curr_dir = item.stem
    curr_output_path = Path(output_path, curr_dir)

    # Create output directory
    Path(curr_output_path).mkdir(parents=True, exist_ok=True)

    match = re.compile(regex).search(curr_dir)
    p_info = match.groupdict()

    # Opening JSON file
    f = open(Path(item, "1/config.json"))

    data = json.load(f)
    data.pop('SACRED_OBSERVER')

    _task = data['_task']
    _task_dict = data['_task_dict'][_task]
    _base_path = new_ds_path if new_ds_path else _task_dict['path']
    _csv_path_train = Path(_base_path, Path(_task_dict['train_filename']).name)
    _csv_path_test = Path(_base_path, Path(_task_dict['test_filename']).name)
    _imgs_folder_train = new_img_path if new_img_path else Path(_task_dict["img_path"])
    _fusion = str(data['_comb_method']).capitalize()
    _classifier = data['_classifier']
    _experimental_cfg = data['_experimental_cfg']
    experimental_cfg = copy.deepcopy(_experimental_cfg)
    experimental_cfg["embedding"]["col_sparse"] = _task_dict.get("col_sparse", None)
    experimental_cfg["embedding"]["col_dense"] = _task_dict.get("col_dense", None)

    _meta_data_columns = _task_dict["features"]
    _comb_config = len(experimental_cfg["embedding"]["col_sparse"] + experimental_cfg["embedding"]["col_dense"]) if \
        experimental_cfg["embedding"]["use_DS"] else len(_meta_data_columns)

    _label_name = _task_dict["label"]
    _img_path_col = _task_dict['img_col']
    label_number_col = F"label_number"

    transform_param = get_norm_and_size(data['_model_name'])

    print("- Loading test data...")
    csv_test = pd.read_csv(_csv_path_test)
    test_imgs_id = csv_test[_img_path_col].values
    test_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in test_imgs_id]
    test_labels = csv_test[label_number_col].values
    _labels_name = csv_test[_label_name].unique()   # TODO: Check if need to be ordered
    if data['_use_meta_data']:
        test_meta_data = csv_test[_meta_data_columns]
        meta_data_columns = _meta_data_columns

        if experimental_cfg["embedding"]["use_DS"]:
            col_sparse = experimental_cfg["embedding"]["col_sparse"]
            col_dense = experimental_cfg["embedding"]["col_dense"]
            test_meta_data = revert_ohe(test_meta_data, col_sparse, col_dense)[col_sparse + col_dense]
            meta_data_columns = col_sparse + col_dense

        test_meta_data = test_meta_data.values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        test_meta_data = None
        print("-- No metadata")

    model = set_model(data['_model_name'], len(_labels_name),
                      neurons_reducer_block=data['_neurons_reducer_block'],
                      comb_method=data['_comb_method'], comb_config=_comb_config,
                      pretrained=False, classifier=_classifier, experimental_cfg=experimental_cfg)
    
    # Load weights in model
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    checkpoint = torch.load(_checkpoint_path, map_location=device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load(_checkpoint_path, map_location=device))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # from surgeon_pytorch import Inspect, get_layers
    #
    # print(get_layers(model))
    # model_wrapped = Inspect(model, layer='features')

    print("Init SHAP...")
    multiinput = False if _fusion == "None" else True

    ## Deep Explainer
    n_background = 5 # For production: 100 or 1000
    #n_background = None # To use all train data
    explain_background = True
    n_eval_per_class = 1
    num_workers = 8
    n = n_background

    # To explain
    test_data_loader = get_data_loader(test_imgs_path, test_labels, test_meta_data,
                                       transform=ImgEvalTransform(*transform_param),
                                       batch_size=100, shuf=True, num_workers=num_workers, pin_memory=True)

    x_explain = torch.Tensor()
    y_explain = torch.Tensor()
    metadata_explain = torch.Tensor()

    for s in range(n_eval_per_class):
        data_explain = next(iter(test_data_loader))[:-1]
        explain_cls, explain_idx = np.unique(data_explain[1], return_index=True)
        data_explain = [i[explain_idx] if len(i) != 0 else [] for i in data_explain]
        x_explain_item, y_explain_item, metadata_explain_item = data_explain

        x_explain = torch.cat([x_explain, x_explain_item])
        y_explain = torch.cat([y_explain, y_explain_item]).int()
        metadata_explain = torch.cat([metadata_explain, metadata_explain_item])

    # Background
    print("- Loading training data...")
    csv_train = pd.read_csv(_csv_path_train)
    train_imgs_id = csv_train[_img_path_col].values
    train_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = csv_train[label_number_col].values
    if data['_use_meta_data']:
        train_meta_data = csv_train[_meta_data_columns]
        meta_data_columns = _meta_data_columns

        if experimental_cfg["embedding"]["use_DS"]:
            col_sparse = experimental_cfg["embedding"]["col_sparse"]
            col_dense = experimental_cfg["embedding"]["col_dense"]
            train_meta_data = revert_ohe(train_meta_data, col_sparse, col_dense)[col_sparse + col_dense]
            meta_data_columns = col_sparse + col_dense

        train_meta_data = train_meta_data.values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        train_meta_data = None

    train_batch_size = n_background if n_background is not None else len(train_labels)
    train_data_loader = get_data_loader(train_imgs_path, train_labels, train_meta_data,
                                       transform=ImgEvalTransform(*transform_param),
                                       batch_size=train_batch_size, shuf=True, num_workers=num_workers, pin_memory=True)

    data_background = next(iter(train_data_loader))[:-1]
    x_background, y_background, metadata_background = data_background

    # Explain all background data
    if explain_background:
        x_explain, y_explain, metadata_explain = x_background, y_background, metadata_background


    if multiinput:
        # x_test, y_test, metadata_test = x_test[:n], y_test[:n], metadata_test[:n]
        # # test_images, test_labels_values = x_test[-2:], y_test[-2:]
        #
        # to_explain = {
        #     "img": x_test[-2:],
        #     "metadata": metadata_test[-2:],
        #     "label": y_test[-2:],
        # }

        # Deep Explainer
        explainer = shap.DeepExplainer(model, [x_background, metadata_background])
        shap_values = explainer.shap_values([x_explain, metadata_explain])

        # Explainer
        # explainer = shap.Explainer(model, [x_background, metadata_background])
        # shap_values = explainer.shap_values([x_explain, metadata_explain])

        # KernelExplainer
        # explainer = shap.KernelExplainer(model, [x_background, metadata_background])
        # shap_values = explainer.shap_values([x_explain, metadata_explain], nsamples=100)

        # ## Gradient Explainer
        # explainer = shap.GradientExplainer(model, [x_background, metadata_background])
        # shap_values = explainer.shap_values([x_explain, metadata_explain])
        # explainer.expected_value = np.zeros(len(_labels_name),)  # GradientExplainer does not have expected_value
        # # TODO: Calculate expected_value for GradientExplainer

        # Open a file and use dump()
        with open(Path(curr_output_path, "save_shap_values.pkl"), 'wb') as file:
            # A new file will be created
            pickle.dump(shap_values, file)

        output_len = len(shap_values)
        shap_img = [shap_values[i][0] for i in range(output_len)]
        shap_metadata = [shap_values[i][1] for i in range(output_len)]

        # Shap Image
        # We need to reshape it to (H, W, C):
        shap_bhwc = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_img]
        # shap_bhwc = torch.Tensor(shap_img).permute(0, 2, 3, 1).numpy()
        # test_numpy = np.swapaxes(np.swapaxes(to_explain["img"].numpy(), 1, -1), 1, 2)
        test_bhwc = ImgEvalReverseTransform(*transform_param)(x_explain).permute(0, 2, 3, 1).numpy()

        labels_vect = []
        n_to_explain = x_explain.shape[0]

        shap_img_arr = np.array(shap_img)
        y_img_predictions = np.apply_over_axes(np.sum, shap_img_arr, range(len(shap_img_arr.shape))[2:]).squeeze().transpose() + [explainer.expected_value] * n_to_explain

        for sample in range(n_to_explain):
            labels_vect.append(class_labels(y_img_predictions[sample], _labels_name))
        labels_vect = np.array(labels_vect)

        shap.image_plot(shap_bhwc, test_bhwc, labels=labels_vect, true_labels=list(_labels_name[y_explain]), show=False)

        title_str = F"Image explanation"

        ax = plt.gcf()
        ax.suptitle(title_str, fontsize='xx-large')
        plt.text(0.16, 0.91, 'True label', fontweight='bold', fontsize='x-large', transform=ax.transFigure)
        # plt.show()
        img_savefig = Path(curr_output_path, "image-explanation")
        plt.savefig(img_savefig.with_suffix(".png"))
        plt.savefig(img_savefig.with_suffix(".pdf"))
        plt.close()

        # Shap metadata
        ##
        # summarize the effects of all the features
        # shap.plots.beeswarm(shap_metadata, show=False)
        for label_idx, label_name in enumerate(_labels_name):
            title_str = F"Metadata explanation of class {label_name}"
            shap.summary_plot(
                shap_metadata[label_idx],
                metadata_explain,
                feature_names=meta_data_columns,
                show=False,
                plot_type="dot",
                title=title_str,
                plot_size=(30, 5)
            )
            img_savefig = Path(curr_output_path, F"decision-explanation_summary-{label_name}")
            plt.savefig(img_savefig.with_suffix(".png"))
            plt.savefig(img_savefig.with_suffix(".pdf"))
            plt.close()

        ##
        shap_meta_arr = np.array(shap_metadata)
        y_meta_predictions = np.apply_over_axes(np.sum, shap_meta_arr,  range(len(shap_meta_arr.shape))[2:]).squeeze().transpose() + [explainer.expected_value] * n_to_explain

        # Sum of img and meta shap values for each class
        y_img_meta_predictions = y_img_predictions + y_meta_predictions

        for sample in range(n_to_explain):
            # Our naive cutoff point is zero log odds (probability 0.5).
            # y_hat = [np.argmax(y_meta_predictions[sample])]
            # legends_str = class_labels(y_meta_predictions[sample], _labels_name)
            y_hat = [np.argmax(y_img_meta_predictions[sample])]
            legends_str = class_labels(y_img_meta_predictions[sample], _labels_name)
            title_str = F"Metadata explanation for sample of class {_labels_name[y_explain[sample]]}"
            shap.multioutput_decision_plot(
                explainer.expected_value.tolist(),
                shap_metadata,
                row_index=sample,
                legend_labels=legends_str,
                legend_location='lower right',
                features=metadata_explain.numpy(),
                feature_names=meta_data_columns,
                highlight=y_hat,
                title=title_str,
                show=False
            )
            img_savefig = Path(curr_output_path, F"decision-explanation_{sample}")
            plt.savefig(img_savefig.with_suffix(".png"))
            plt.savefig(img_savefig.with_suffix(".pdf"))
            plt.close()

    else:
        # Deep Explainer

        explainer = shap.DeepExplainer(model, x_background)
        shap_values = explainer.shap_values(x_explain)

        ## Gradient Explainer
        # explainer = shap.GradientExplainer(model, x_test[:-2])
        # shap_values = explainer.shap_values(x_test[-2:], nsamples=100)

        # Shap Image
        output_len = len(shap_values)
        shap_img = [shap_values[i] for i in range(output_len)]

        # We need to reshape it to (H, W, C):
        shap_bhwc = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_img]
        test_bhwc = ImgEvalReverseTransform(*transform_param)(x_explain).permute(0, 2, 3, 1).numpy()

        labels_vect = []
        n_to_explain = x_explain.shape[0]

        shap_img_arr = np.array(shap_img)
        y_img_predictions = np.apply_over_axes(np.sum, shap_img_arr, range(len(shap_img_arr.shape))[2:]).squeeze().transpose() + [explainer.expected_value] * n_to_explain

        for sample in range(n_to_explain):
            labels_vect.append(class_labels(y_img_predictions[sample], _labels_name))

        shap.image_plot(shap_bhwc, test_bhwc, labels=labels_vect, show=False)
        title_str = F"Image explanation"

        ax = plt.gcf()
        ax.suptitle(title_str, fontsize='xx-large')
        plt.text(0.05, 0.9, 'True label', fontweight='bold', fontsize='x-large', transform=ax.transFigure)
        last_img_pos = 220
        pos_inc = -470
        for y_label in y_explain.flip(0):
            plt.text(-2000, last_img_pos, _labels_name[y_label.int()], fontsize='large', fontweight='bold')
            last_img_pos += pos_inc
        img_savefig = Path(curr_output_path, "image-explanation")
        plt.savefig(img_savefig.with_suffix(".png"))
        plt.savefig(img_savefig.with_suffix(".pdf"))
        plt.close()
