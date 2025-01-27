# -*- coding: utf-8 -*-
"""
Author: Leandro Lima
"""



import sys

sys.path.insert(0, '../../')  # including the path to deep-tasks folder
sys.path.insert(0, '../../my_models')  # including the path to my_models folder
from constants import RAUG_PATH

sys.path.insert(0, RAUG_PATH)
from raug.loader import get_data_loader
from raug.train import fit_model
from raug.eval import test_model
from my_model import set_model, get_norm_and_size
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch
from aug_sab import ImgTrainTransform, ImgEvalTransform
import time
from sacred import Experiment
from sacred.observers import FileStorageObserver
from raug.utils.loader import get_labels_frequency
import copy
from aux import revert_ohe

from MultiClassifier import MultiClassifier

from explain import shap_explain
import numpy as np

# Dataset download: https://data.mendeley.com/datasets/zr7vgbcyr2/1

# Starting sacred experiment
ex = Experiment()

generate_shap = True

@ex.config
def cnfg():
    # Dataset variables

    # Task
    _task = "TaskII"
    _task_base_path = "/home/leandro/Documentos/doutorado/dados"
    _task_dict = {
        "TaskII": {
            "label": "TaskII",
            "features": [
                "localization_Tongue", "localization_Lip", "localization_Floor of mouth", "localization_Buccal mucosa",
                "localization_Palate", "localization_Gingiva", "larger_size", "tobacco_use_Yes", "tobacco_use_Former",
                "tobacco_use_No", "tobacco_use_Not informed", "alcohol_consumption_No", "alcohol_consumption_Former",
                "alcohol_consumption_Yes", "alcohol_consumption_Not informed", "sun_exposure_No", "sun_exposure_Yes",
                "sun_exposure_Not informed", "gender_M", "gender_F", "age_group_2", "age_group_1", "age_group_0"
            ],
            "col_sparse":
                [
                    "localization", "tobacco_use", "alcohol_consumption", "sun_exposure", "gender", "age_group"
                ],
            "col_dense":
                [
                    "larger_size",
                ],
            "path": F"{_task_base_path}/sab/NDB-UFES",
            "img_path": F"{_task_base_path}/sab/NDB-UFES/images",
            "img_col": "path",
            "train_filename": "ndbufes_TaskII_parsed_folders.csv",
            "test_filename": "ndbufes_TaskII_parsed_test.csv",
        },

        "TaskIII": {
            "label": "TaskIII",
            "features": [
                "localization_Tongue", "localization_Lip", "localization_Floor of mouth", "localization_Buccal mucosa",
                "localization_Palate", "localization_Gingiva", "larger_size", "tobacco_use_Yes", "tobacco_use_Former",
                "tobacco_use_No", "tobacco_use_Not informed", "alcohol_consumption_No", "alcohol_consumption_Former",
                "alcohol_consumption_Yes", "alcohol_consumption_Not informed", "sun_exposure_No", "sun_exposure_Yes",
                "sun_exposure_Not informed", "gender_M", "gender_F", "age_group_2", "age_group_1", "age_group_0"
            ],
            "col_sparse":
                [
                    "localization", "tobacco_use", "alcohol_consumption", "sun_exposure", "gender", "age_group"
                ],
            "col_dense":
                [
                    "larger_size",
                ],
            "path": F"{_task_base_path}/sab/NDB-UFES",
            "img_path": F"{_task_base_path}/sab/NDB-UFES/images",
            "img_col": "path",
            "train_filename": "ndbufes_TaskIII_parsed_folders.csv",
            "test_filename": "ndbufes_TaskIII_parsed_test.csv",
        },

        "TaskIV": {
            "label": "TaskIV",
            "features": [
                "localization_Tongue", "localization_Lip", "localization_Floor of mouth", "localization_Buccal mucosa",
                "localization_Palate", "localization_Gingiva", "larger_size", "tobacco_use_Yes", "tobacco_use_Former",
                "tobacco_use_No", "tobacco_use_Not informed", "alcohol_consumption_No", "alcohol_consumption_Former",
                "alcohol_consumption_Yes", "alcohol_consumption_Not informed", "sun_exposure_No", "sun_exposure_Yes",
                "sun_exposure_Not informed", "gender_M", "gender_F", "age_group_2", "age_group_1", "age_group_0"
            ],
            "col_sparse":
                [
                    "localization", "tobacco_use", "alcohol_consumption", "sun_exposure", "gender", "age_group"
                ],
            "col_dense":
                [
                    "larger_size",
                ],
            "path": F"{_task_base_path}/NDB-UFES",
            "img_path": F"{_task_base_path}/NDB-UFES/images",
            "img_col": "path",
            "train_filename": "ndbufes_TaskIV_parsed_folders.csv",
            "test_filename": "ndbufes_TaskIV_parsed_test.csv",
        },

        "Patch": {
            "label": "diagnostico",
            "features": None,
            "path": F"{_task_base_path}/sab-patch/data_train_test/data",
            "img_path": F"{_task_base_path}/sab-patch/data_train_test/data",
            "img_col": "path",
            "train_filename": "sab_parsed_folders.csv",
            "test_filename": "sab_parsed_test.csv",
        },

        "Patch_Displasia": {
            "label": "diagnostico",
            "features": None,
            "path": F"{_task_base_path}/sab-patch/sabpatch_displasia",
            "img_path": F"{_task_base_path}/sab-patch/sabpatch_displasia",
            "img_col": "path",
            "train_filename": "sab_parsed_folders.csv",
            "test_filename": "sab_parsed_test.csv",

        }
    }
    # BASE_PATH = F"{_task_base_path}/sab-patch/data_train_test/data"

    _experimental_cfg = {
        "embedding": {
            #"use_DS": False,
            "use_DS": True,
            "use_DDS": False,
            "emb_dim": 25,
            "emb_out": 15,
            "categorical_max_size": 15,
            "use_SNN": False,
        },
        "poincare": {
            #"use_Poincare": False,
            "use_Poincare": True,
            "c": 1.0,
            "r": 0.0,  # Set to r<=0 to disable poincare clip
            "type": "feature_conv_concat"  # "feature_conv_concat" or "feature_concat"
        },
        "late_fusion": {
            "use_outter_classifier": True,          # Usually always True
            "pretrained_outer_classifier": "transformer",  # "mlp" or "transformer"
            "freeze_backbone": False,
            "use_softmax": False,
            "late_fusion": False,
            "pre_fusion": False,
            "late_residual": False,
        }
    }

    # Dataset variables
    _folder = 5  # 5
    _base_path = _task_base_path
    _csv_path_train = os.path.join(_base_path, "sab_parsed_folders.csv")
    _csv_path_test = os.path.join(_base_path, "sab_parsed_test.csv")
    _imgs_folder_train = os.path.join(_base_path, "imagem-exemplo")

    _use_meta_data = False
    _neurons_reducer_block = 0  # original:
    _comb_method = None  # None, metanet, concat, or metablock / gcell
    _comb_config = 12  # Concat
    # _comb_config = (64, 81)  # cf=0.8 -> (324, 81) - Metablock
    _batch_size = 24  # 24; não definido na tese
    _epochs = 150
    _classifier = "linear"
    _num_workers = 8

    # Training variables
    _best_metric = "loss"
    _pretrained = True

    _keep_lr_prop = True
    # Keep lr x batch_size proportion. Batch size 30 was the original one. Read more in https://arxiv.org/abs/2006.09092
    # For adaptive optimizers
    # prop = np.sqrt(_batch_size/30.) if _keep_lr_prop else 1
    # For SGD
    prop = _batch_size / 30. if _keep_lr_prop else 1
    _lr_init = 0.001 * prop
    _sched_factor = 0.1 * prop
    _sched_min_lr = 1e-6 * prop

    _sched_patience = 10
    _early_stop = 15
    _metric_early_stop = None
    _weights = "frequency"

    _model_name = 'mobilenet'
    _save_path = "results"
    _save_folder = str(_save_path) + "/" + str(_comb_method) + "_" + _model_name.replace("/", "_") + "_reducer_" + str(
        _neurons_reducer_block) + "_fold_" + str(_folder) + "_" + str(time.time()).replace('.', '')

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)



@ex.automain
def main(_folder, _csv_path_train, _imgs_folder_train, _lr_init, _sched_factor, _sched_min_lr, _sched_patience,
         _batch_size, _epochs, _early_stop, _weights, _model_name, _pretrained, _save_folder, _csv_path_test,
         _best_metric, _neurons_reducer_block, _comb_method, _comb_config, _use_meta_data, _metric_early_stop,
         _num_workers, _task, _task_dict, _classifier, _experimental_cfg):
    meta_data_columns = _task_dict[_task]["features"]
    experimental_cfg = copy.deepcopy(_experimental_cfg)
    experimental_cfg["embedding"]["col_sparse"] = _task_dict[_task].get("col_sparse", None)
    experimental_cfg["embedding"]["col_dense"] = _task_dict[_task].get("col_dense", None)

    assert ("embedding" in experimental_cfg.keys() and "use_DS" in experimental_cfg["embedding"].keys() and
            "use_DDS" in experimental_cfg["embedding"].keys() and
            "emb_dim" in experimental_cfg["embedding"].keys() and
            "emb_out" in experimental_cfg["embedding"].keys())
    assert (
            experimental_cfg["embedding"]["use_DS"] is False or
            (experimental_cfg["embedding"]["col_sparse"] is not None and
             experimental_cfg["embedding"]["col_dense"] is not None)
    )

    assert ("poincare" in experimental_cfg.keys() and "use_Poincare" in experimental_cfg["poincare"].keys())
    assert (
            experimental_cfg["poincare"]["use_Poincare"] is False or
            (experimental_cfg["poincare"]["c"] is not None and experimental_cfg["poincare"]["type"] is not None)
    )

    _comb_config = len(experimental_cfg["embedding"]["col_sparse"] + experimental_cfg["embedding"]["col_dense"]) if \
        experimental_cfg["embedding"]["use_DS"] else len(meta_data_columns)

    _label_name = _task_dict[_task]["label"]
    _label_name_col = "label_number"
    _img_path_col = _task_dict[_task]["img_col"]

    _base_path = _task_dict[_task]["path"]
    _csv_path_train = os.path.join(_base_path, _task_dict[_task]["train_filename"])
    _csv_path_test = os.path.join(_base_path, _task_dict[_task]["test_filename"])
    _imgs_folder_train = os.path.join(_task_dict[_task]["img_path"])

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

    # Loading the csv file
    csv_all_folders = pd.read_csv(_csv_path_train)

    print("-" * 50)
    print("- Loading validation data...")
    if 'synthetic' in csv_all_folders.columns:
        synthetics = csv_all_folders["synthetic"]
    else:
        synthetics = False

    val_csv_folder = csv_all_folders[(csv_all_folders['folder'] == _folder) & ~synthetics]
    train_csv_folder = csv_all_folders[csv_all_folders['folder'] != _folder]

    transform_param = get_norm_and_size(_model_name)

    # Loading validation data
    val_imgs_id = val_csv_folder[_img_path_col].values
    val_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in val_imgs_id]
    val_labels = val_csv_folder[_label_name_col].values
    if _use_meta_data:
        val_meta_data = val_csv_folder[meta_data_columns]

        if experimental_cfg["embedding"]["use_DS"]:
            col_sparse = experimental_cfg["embedding"]["col_sparse"]
            col_dense = experimental_cfg["embedding"]["col_dense"]
            val_meta_data = revert_ohe(val_meta_data, col_sparse, col_dense)[col_sparse + col_dense]

        val_meta_data = val_meta_data.values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        val_meta_data = None
    val_data_loader = get_data_loader(val_imgs_path, val_labels, val_meta_data,
                                      transform=ImgEvalTransform(*transform_param),
                                      batch_size=_batch_size, shuf=True, num_workers=_num_workers, pin_memory=True)
    print("-- Validation partition loaded with {} images".format(len(val_data_loader) * _batch_size))

    print("- Loading training data...")
    train_imgs_id = train_csv_folder[_img_path_col].values
    train_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder[_label_name_col].values
    if _use_meta_data:
        train_meta_data = train_csv_folder[meta_data_columns]

        if experimental_cfg["embedding"]["use_DS"]:
            col_sparse = experimental_cfg["embedding"]["col_sparse"]
            col_dense = experimental_cfg["embedding"]["col_dense"]
            train_meta_data = revert_ohe(train_meta_data, col_sparse, col_dense)[col_sparse + col_dense]

        train_meta_data = train_meta_data.values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        train_meta_data = None
    train_data_loader = get_data_loader(train_imgs_path, train_labels, train_meta_data,
                                        transform=ImgTrainTransform(*transform_param),
                                        batch_size=_batch_size, shuf=True, num_workers=_num_workers, pin_memory=True,
                                        drop_last=True
                                        )
    print("-- Training partition loaded with {} images".format(len(train_data_loader) * _batch_size))

    print("-" * 50)
    ####################################################################################################################

    ser_lab_freq = get_labels_frequency(train_csv_folder, _label_name, _img_path_col)
    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values
    print(ser_lab_freq)
    ####################################################################################################################
    print("- Loading", _model_name)

    model = set_model(_model_name, len(_labels_name), neurons_reducer_block=_neurons_reducer_block,
                      comb_method=_comb_method, comb_config=_comb_config, pretrained=_pretrained,
                      classifier=experimental_cfg["late_fusion"]["pretrained_outer_classifier"], experimental_cfg=experimental_cfg)
    ####################################################################################################################
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).to(device))
    optimizer = optim.SGD(model.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                        patience=_sched_patience)
    ####################################################################################################################

    print("- Starting the training phase...")
    print("-" * 50)
    fit_model(model, train_data_loader, val_data_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs,
              epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None,
              metric_early_stop=_metric_early_stop,
              device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=False,
              history_plot=True, val_metrics=["balanced_accuracy"], best_metric=_best_metric)
    ####################################################################################################################

    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model(model, val_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
               partition_name='eval', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
               apply_softmax=True, verbose=False)
    ####################################################################################################################

    ####################################################################################################################

    print("- Loading test data...")
    csv_test = pd.read_csv(_csv_path_test)
    test_imgs_id = csv_test[_img_path_col].values
    test_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in test_imgs_id]
    test_labels = csv_test[_label_name_col].values
    if _use_meta_data:
        test_meta_data = csv_test[meta_data_columns]

        if experimental_cfg["embedding"]["use_DS"]:
            col_sparse = experimental_cfg["embedding"]["col_sparse"]
            col_dense = experimental_cfg["embedding"]["col_dense"]
            test_meta_data = revert_ohe(test_meta_data, col_sparse, col_dense)[col_sparse + col_dense]

        test_meta_data = test_meta_data.values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        test_meta_data = None
        print("-- No metadata")

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "test_pred_nn_classifier"),
        'pred_name_scores': 'predictions.csv',
        'normalize_conf_matrix': True}
    test_data_loader = get_data_loader(test_imgs_path, test_labels, test_meta_data,
                                       transform=ImgEvalTransform(*transform_param),
                                       batch_size=_batch_size, shuf=False, num_workers=_num_workers, pin_memory=True)
    print("-" * 50)
    # Testing the test partition
    print("\n- Evaluating the test partition...")
    test_model(model, test_data_loader, checkpoint_path=_checkpoint_best, metrics_to_comp="all",
               class_names=_labels_name, metrics_options=_metric_options, save_pred=True, verbose=False)
    ####################################################################################################################

    print("-" * 50)

    print("- Training non-nn classifier")

    non_nn_classifier = MultiClassifier(
        model, classifier_name=_classifier, features="mixed",
        checkpoint_path=_checkpoint_best, class_names=_labels_name, device=device
    )
    # non_nn_classifier = MultiClassifier(
    #     model, train_data_loader, val_data_loader,
    #     classifier_name=_classifier, optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs,
    #     epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None,
    #     metric_early_stop=_metric_early_stop,
    #     device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=False,
    #     history_plot=True, val_metrics=["balanced_accuracy"], best_metric=_best_metric
    # )
    non_nn_classifier.fit(train_data_loader, val_data_loader)

    print("\n- Evaluating the test partition...")

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "test_pred"),
        'pred_name_scores': 'predictions.csv',
        'normalize_conf_matrix': True}
    print("-" * 50)

    non_nn_classifier.eval(
        test_data_loader,
        metrics_to_comp="all",
        class_names=_labels_name, metrics_options=_metric_options, save_pred=True, verbose=False
    )

    if generate_shap:
        shap_explain(
            non_nn_classifier,
            test_data_loader,
            columns=np.concatenate([_labels_name, np.array(meta_data_columns)]).tolist(),
            labels=_labels_name.tolist(),
            save_path=os.path.join(_save_folder, "shap_output")
        )

