"""
Author: Leandro Lima
"""

############
# Setup sabpatch dataset
import sys

sys.path.insert(0, '../../')  # including the path to deep-tasks folder
sys.path.insert(0, '../../my_models')  # including the path to my_models folder
from constants import RAUG_PATH, DATA_PATH, config_bot

sys.path.insert(0, RAUG_PATH)
from raug.loader import get_data_loader
from raug.fsl.loader import get_data_loader as get_data_loader_fsl
from raug.fsl.train import fit_model
from raug.fsl.eval import test_model
from my_model import set_model, get_norm_and_size, FSLWrapper
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch
from aug_sab import ImgTrainTransform, ImgEvalTransform
from sacred.observers import FileStorageObserver
import time
from pathlib import Path

from sacred import Experiment
from raug.utils.loader import get_labels_frequency

# from easyfsl.methods import PrototypicalNetworks

from fsl import load_paper_backbone, PrototypicalNetworksDC, PrototypicalNetworks

config_bot = None

# Starting sacred experiment
ex = Experiment()


@ex.config
def cnfg():
    # Task
    _folder = 5  # 5
    _task = "Patch"
    _task_base_path = DATA_PATH
    _task_dict = {
        "Carcinoma_Leucoplasia": {
            "label": "diagnostico",
            "features": [
                "localizacao_ASSOALHO DE BOCA", "localizacao_LÍNGUA", "localizacao_LABIO", "localizacao_MUCOSA JUGAL",
                "localizacao_PALATO", "localizacao_FUNDO DE VESTÍBULO", "localizacao_MUCOSA LABIAL",
                "tamanho_maior",
                "sexo_M", "sexo_F",
                "uso_bebida_No passado", "uso_bebida_Não informado", "uso_bebida_Sim", "uso_bebida_Não",
                "uso_cigarro_Sim", "uso_cigarro_Não informado", "uso_cigarro_Não", "uso_cigarro_No passado",
            ],
            "path": F"{_task_base_path}/sab/diag_definitivo/extra/leucoplasia/18_features",
            "img_path": F"{_task_base_path}/sab/histopatologico/imagensHistopatologica",
            "img_col": "path",
        },

        "Displasia_Cancer": {
            "label": "displasia",
            "features": [
                "localizacao_ASSOALHO DE BOCA", "localizacao_LÍNGUA", "localizacao_LABIO", "localizacao_MUCOSA JUGAL",
                "localizacao_PALATO", "localizacao_FUNDO DE VESTÍBULO", "localizacao_MUCOSA LABIAL",
                "tamanho_maior",
                "sexo_M", "sexo_F",
                "uso_bebida_No passado", "uso_bebida_Não informado", "uso_bebida_Sim", "uso_bebida_Não",
                "uso_cigarro_Sim", "uso_cigarro_Não informado", "uso_cigarro_Não", "uso_cigarro_No passado",
            ],
            "path": F"{_task_base_path}/sab/diag_definitivo/extra/displasia_cancer/18_features",
            "img_path": F"{_task_base_path}/sab/histopatologico/imagensHistopatologica",
            "img_col": "path",
        },

        "Displasia": {
            "label": "ausencia_de_displasia",
            "features": [
                "localizacao_ASSOALHO DE BOCA", "localizacao_LÍNGUA", "localizacao_LABIO", "localizacao_MUCOSA JUGAL",
                "localizacao_PALATO", "localizacao_FUNDO DE VESTÍBULO", "localizacao_MUCOSA LABIAL",
                "tamanho_maior",
                "sexo_M", "sexo_F",
                "uso_bebida_No passado", "uso_bebida_Não informado", "uso_bebida_Sim", "uso_bebida_Não",
                "uso_cigarro_Sim", "uso_cigarro_Não informado", "uso_cigarro_Não", "uso_cigarro_No passado",
            ],
            "path": F"{_task_base_path}/sab/diag_definitivo/extra/displasia/18_features",
            "img_path": F"{_task_base_path}/sab/histopatologico/imagensHistopatologica",
            "img_col": "path",
        },

        "Patch": {
            "label": "diagnostico",
            "features": None,
            "path": F"{_task_base_path}/sabpatch/Patch",
            "img_path": F"{_task_base_path}/sabpatch/Patch",
            "img_col": "path",
        },

        "Patch_Displasia": {
            "label": "diagnostico",
            "features": None,
            "path": F"{_task_base_path}/sabpatch/Patch_Displasia",
            "img_path": F"{_task_base_path}/sabpatch/Patch_Displasia",
            "img_col": "path",
        }
    }
    _save_path = "results/fsl"

    _use_meta_data = False  # TODO: Add multimodal input
    _model_name = 'mobilenetv2_100'

    # Options: "protonet", "protonetdc"
    _learning_method = "protonet"

    _batch_size = 30
    # _batch_size = 30
    _epochs = 150
    # _epochs = 150

    _n_workers = 8

    _n_way = 3
    # _n_shot = 1
    _n_shot = 5
    _n_query = 15
    _n_validation_tasks = 50
    _n_test_tasks = 1000

    _save_features = False

    _optimizer = "SGD"  # "Adam", "SGD"
    _sched_patience = 10
    _early_stop = 15
    _metric_early_stop = None

    _best_metric = "loss"

    _neurons_reducer_block = 90  # original:
    _comb_method = None  # None, metanet, concat, or metablock / gcell
    _comb_config = 12  # Concat
    _pretrained = True

    _keep_lr_prop = True
    _weights = "frequency"

    # Keep lr x batch_size proportion. Batch size 30 was the original one. Read more in https://arxiv.org/abs/2006.09092
    # For adaptive optimizers
    # prop = np.sqrt(_batch_size/30.) if _keep_lr_prop else 1
    # For SGD
    prop = _batch_size / 30. if _keep_lr_prop else 1
    _lr_init = 0.001 * prop
    _sched_factor = 0.1 * prop
    _sched_min_lr = 1e-6 * prop

    _save_basepath = Path(_save_path, F"{_n_way}way{_n_shot}shot")
    _save_folder = str(_save_basepath) + "/" + str(
        _comb_method) + "_" + _learning_method + "+" + _optimizer + "+" + _model_name + "_reducer_" + str(
        _neurons_reducer_block) + "_fold_" + str(_folder) + "_" + str(time.time()).replace('.', '')
    _save_folder = Path(_save_folder)

    # _csv_path_train_file = "sabpatch_parsed_folders_aug.csv"
    _csv_path_train_file = "sab_parsed_folders.csv"
    _csv_path_test_file = "sab_parsed_test.csv"

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

##########################################################################
@ex.automain
def main(_folder, _lr_init, _sched_factor, _sched_min_lr, _sched_patience,
         _batch_size, _epochs, _early_stop, _weights, _model_name, _pretrained, _save_folder,
         _best_metric, _neurons_reducer_block, _comb_method, _comb_config, _use_meta_data, _metric_early_stop, _task,
         _task_dict, _csv_path_train_file, _csv_path_test_file, _n_workers, _optimizer,
         _n_way, _n_shot, _n_query, _learning_method, _n_validation_tasks, _n_test_tasks, _save_features):
    meta_data_columns = _task_dict[_task]["features"]
    _label_name = _task_dict[_task]["label"]
    _img_path_col = _task_dict[_task]["img_col"]

    transform_param = get_norm_and_size(_model_name)
    _base_path = _task_dict[_task]["path"]
    _csv_path_train = os.path.join(_base_path, _csv_path_train_file)

    _csv_path_test = os.path.join(_base_path, _csv_path_test_file)
    _imgs_folder_train = os.path.join(_task_dict[_task]["img_path"])

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

    # Loading validation data
    val_imgs_id = val_csv_folder[_img_path_col].values
    val_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in val_imgs_id]
    val_labels = val_csv_folder['label_number'].values
    if _use_meta_data:
        val_meta_data = val_csv_folder[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        val_meta_data = None
    # val_data_loader = get_data_loader(val_imgs_path, val_labels, val_meta_data,
    #                                   transform=ImgEvalTransform(),
    #                                   # transform=ImgEvalTransform(*transform_param),
    #                                   batch_size=_batch_size, shuf=True, num_workers=8, pin_memory=True)
    # print("-- Validation partition loaded with {} images".format(len(val_data_loader) * _batch_size))

    print("- Loading training data...")
    train_imgs_id = train_csv_folder[_img_path_col].values
    train_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder['label_number'].values
    if _use_meta_data:
        train_meta_data = train_csv_folder[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        print("-- No metadata")
        train_meta_data = None

    ####################################################################################################################

    ser_lab_freq = get_labels_frequency(train_csv_folder, _label_name, _img_path_col)
    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values
    print(ser_lab_freq)
    ####################################################################################################################
    # gamma = 2.8
    # batch_transform = ZetaMixup(len(_labels_name), _batch_size, gamma)
    train_data_loader = get_data_loader(train_imgs_path, train_labels, train_meta_data,
                                        transform=ImgTrainTransform(*transform_param),
                                        # batch_transform=batch_transform,
                                        batch_size=_batch_size, shuf=True, num_workers=_n_workers, pin_memory=True,
                                        drop_last=True
                                        )
    print("-- Training partition loaded with {} images".format(len(train_data_loader) * _batch_size))

    print("-" * 50)

    # Define model
    model = set_model(_model_name, len(_labels_name), neurons_reducer_block=_neurons_reducer_block,
                      comb_method=_comb_method, comb_config=_comb_config, pretrained=_pretrained)
    model = FSLWrapper(model, use_fc=True)
    ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # few_shot_classifier = PrototypicalNetworks(model).to(device)

    # modelPN = load_paper_backbone()
    # modelPN = FSLWrapper(modelPN, use_fc=True).to(device)
    if _learning_method == "protonet":
        few_shot_classifier = PrototypicalNetworks(model).to(device)
    elif _learning_method == "protonetdc":
        few_shot_classifier = PrototypicalNetworksDC(model, base_info_path=DATA_PATH).to(device)
    else:
        raise Exception(
            "Illegal Few-Shot Learning."
            "Expected 'protonet' or 'protonetdc"
        )

    ####################################################################################################################
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)

    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).to(device))

    if _optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=_lr_init, weight_decay=0.001)
    elif _optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001)
    else:
        raise Exception(
            "Illegal optimizer."
            "Expected 'Adam' or 'SGD'"
        )
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                        patience=_sched_patience)
    ####################################################################################################################
    # Define validation dataloader
    fsl_params = (_n_way, _n_shot, _n_query, _n_validation_tasks)
    val_data_loader, val_set = get_data_loader_fsl(val_imgs_path, val_labels, val_meta_data,
                                                   # transform=ImgEvalTransform(),
                                                   transform=ImgEvalTransform(*transform_param),
                                                   num_workers=_n_workers, pin_memory=True,
                                                   returnDataset=True, fsl_params=fsl_params
                                                   )
    print(F"Val - FSL (_n_way, _n_shot, _n_query, _n_validation_tasks): {fsl_params}")
    # Training (Classical version)
    # Train setup
    ####################################################################################################################
    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

    print("- Starting the training phase...")
    print("-" * 50)
    fit_model(few_shot_classifier, train_data_loader, val_data_loader, optimizer=optimizer, loss_fn=loss_fn,
              epochs=_epochs,
              epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None,
              metric_early_stop=_metric_early_stop,
              device=None, schedule_lr=scheduler_lr, config_bot=config_bot, model_name=_model_name, resume_train=False,
              history_plot=True, val_metrics=["balanced_accuracy"], best_metric=_best_metric)

    ####################################################################################################################

    ####################################################################################################################

    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model(few_shot_classifier, val_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
               partition_name='eval', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
               apply_softmax=True, verbose=False)
    ####################################################################################################################

    # Setup test data

    print("- Loading test data...")
    csv_test = pd.read_csv(_csv_path_test)
    test_imgs_id = csv_test[_img_path_col].values
    test_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in test_imgs_id]
    test_labels = csv_test['label_number'].values
    if _use_meta_data:
        test_meta_data = csv_test[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        test_meta_data = None
        print("-- No metadata")

    fsl_params = (_n_way, _n_shot, _n_query, _n_test_tasks)
    test_loader, test_set = get_data_loader_fsl(test_imgs_path, test_labels, test_meta_data,
                                                # transform=ImgEvalTransform(),
                                                transform=ImgEvalTransform(*transform_param),
                                                num_workers=_n_workers, pin_memory=True,
                                                returnDataset=True, fsl_params=fsl_params
                                                )

    print(F"Test - FSL (_n_way, _n_shot, _n_query, _n_validation_tasks): {fsl_params}")
    # Test model
    model.set_use_fc(False)
    if _save_features:
        model.set_save_features(True)

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "test_pred"),
        'pred_name_scores': 'predictions.csv',
        'normalize_conf_matrix': True}
    test_model(few_shot_classifier, test_loader, checkpoint_path=_checkpoint_best, metrics_to_comp="all",
               class_names=_labels_name, metrics_options=_metric_options, save_pred=True, verbose=False)
