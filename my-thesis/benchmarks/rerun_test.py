"""
Author: Leandro Lima
"""

import sys

import torch

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
# from aug_sab import ImgEvalTransform
from aug_pad import ImgEvalTransform


import pandas as pd
from pathlib import Path
import re
import json

import matplotlib
matplotlib.use('Agg')
#############
# _base_path = "/home/patcha/Datasets/PAD-UFES-20"
# _csv_path_test = os.path.join(_base_path, "pad-ufes-20_parsed_test.csv")
# _imgs_folder_train = Path(_base_path, "imgs")
# _model_name = 'resnet-50'
# _check_base_path = "/home/patcha/Codes/my-deep-ideas/benchmarks/pad/results/concat/resnet-50/5"
# _checkpoint_path = Path(_check_base_path, "best-checkpoint/best-checkpoint.pth")
# _labels_name = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
# _comb_method = "concat"
# _comb_config = 17
# _use_meta_data = True
# _neurons_reducer_block = 90
#############
path_dir = "/mnt/SSD-dados/LeandroLima/resultados/tmp/visualizacao/model-base/poincare_feature_conv_concat/"
path_dir = "/mnt/SSD-dados/LeandroLima/resultados/tmp/visualizacao/model-base/pfcc/"

path = Path(path_dir)
p_list = [i for i in path.glob('*') if i.is_dir()]

regex = r'(?P<fusion>[A-Za-z0-9-]+)_(?P<model>[A-Za-z0-9-_]+)_reducer_(?P<reducer>\d+)_fold_(?P<fold>\d+)_\d+'

for item in p_list:
    _check_base_path = item
    _checkpoint_path = Path(_check_base_path, "best-checkpoint/best-checkpoint.pth")

    curr_dir = item.stem

    match = re.compile(regex).search(curr_dir)
    p_info = match.groupdict()

    # Opening JSON file
    f = open(Path(item, "1/config.json"))

    data = json.load(f)
    data.pop('SACRED_OBSERVER')

    _task = data['_task']
    _task_dict = data['_task_dict'][_task]
    _base_path = _task_dict['path']
    _csv_path_test = Path(_base_path, Path(data["_csv_path_test"]).name)
    _imgs_folder_train = Path(_task_dict["img_path"])

    meta_data_columns = _task_dict["features"]

    data["_experimental_cfg"]["embedding"]["col_sparse"] = _task_dict["col_sparse"]
    data["_experimental_cfg"]["embedding"]["col_dense"] = _task_dict["col_dense"]

    _label_name = _task_dict["label"]
    _label_number = _task_dict["label"] + "_number"
    _img_path_col = _task_dict['img_col']

    transform_param = get_norm_and_size(data['_model_name'])

    print("- Loading test data...")
    csv_test = pd.read_csv(_csv_path_test)
    test_imgs_id = csv_test[_img_path_col].values
    test_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in test_imgs_id]
    test_labels = csv_test[_label_number].values
    _labels_name = csv_test[_label_name].unique()   # TODO: Check if need to be ordered
    if data['_use_meta_data']:
        test_meta_data = csv_test[meta_data_columns].values
        print("-- Using {} meta-data features".format(len(meta_data_columns)))
    else:
        test_meta_data = None
        print("-- No metadata")

    test_data_loader = get_data_loader(test_imgs_path, test_labels, test_meta_data,
                                       transform=ImgEvalTransform(*transform_param),
                                       batch_size=data['_batch_size'], shuf=False, num_workers=8, pin_memory=True)

    model = set_model(data['_model_name'], len(_labels_name),
                      neurons_reducer_block=data['_neurons_reducer_block'],
                      comb_method=data['_comb_method'], comb_config=data['_comb_config'],
                      experimental_cfg=data['_experimental_cfg'],
                      pretrained=False)

    # move test_pred to test_pred_last
    _metric_options = {
            'save_all_path': Path(_check_base_path, "test_pred_2"),
            'pred_name_scores': 'predictions.csv',
            'normalize_conf_matrix': True}

    # test_model(model, test_data_loader, checkpoint_path=_checkpoint_path, metrics_to_comp="all",
    #            class_names=_labels_name,
    #            metrics_options=_metric_options, save_pred=True, verbose=False, apply_softmax=False)
    teste_device = torch.device("cpu")
    test_model(model, test_data_loader, checkpoint_path=_checkpoint_path, metrics_to_comp="all",
               class_names=_labels_name, metrics_options=_metric_options, save_pred=True, verbose=False, device=teste_device)
