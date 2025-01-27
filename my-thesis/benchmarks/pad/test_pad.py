# -*- coding: utf-8 -*-
"""
Autor: André Pacheco, Leandro Lima
Email: pacheco.comp@gmail.com

"""

import sys
sys.path.insert(0,'../../') # including the path to deep-tasks folder
sys.path.insert(0,'../../my_models') # including the path to my_models folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
from raug.loader import get_data_loader
from raug.eval import test_model
from my_model import set_model, get_norm_and_size
import pandas as pd
import os
import torch.nn as nn
import torch
from aug_pad import ImgEvalTransform



_base_path = "/home/patcha/Datasets/PAD-UFES-20"
_csv_path_test = os.path.join(_base_path, "pad-ufes-20_parsed_test.csv")
_imgs_folder_train = os.path.join(_base_path, "imgs")
_model_name = 'resnet-50'
_check_base_path = "/home/patcha/Codes/my-deep-ideas/benchmarks/pad/results/concat/resnet-50/5"
_checkpoint_path = os.path.join(_check_base_path, "best-checkpoint/best-checkpoint.pth")
_labels_name = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
_comb_method = "concat"
_comb_config = 17
_use_meta_data = True
_neurons_reducer_block = 90

# meta_data_columns = ["smoke_False", "smoke_True", "drink_False", "drink_True", "background_father_POMERANIA",
#                      "background_father_GERMANY", "background_father_BRAZIL", "background_father_NETHERLANDS",
#                      "background_father_ITALY", "background_father_POLAND",	"background_father_UNK",
#                      "background_father_PORTUGAL", "background_father_BRASIL", "background_father_CZECH",
#                      "background_father_AUSTRIA", "background_father_SPAIN", "background_father_ISRAEL",
#                      "background_mother_POMERANIA", "background_mother_ITALY", "background_mother_GERMANY",
#                      "background_mother_BRAZIL", "background_mother_UNK", "background_mother_POLAND",
#                      "background_mother_NORWAY", "background_mother_PORTUGAL", "background_mother_NETHERLANDS",
#                      "background_mother_FRANCE", "background_mother_SPAIN", "age", "pesticide_False",
#                      "pesticide_True", "gender_FEMALE", "gender_MALE", "skin_cancer_history_True",
#                      "skin_cancer_history_False", "cancer_history_True", "cancer_history_False",
#                      "has_piped_water_True", "has_piped_water_False", "has_sewage_system_True",
#                      "has_sewage_system_False", "fitspatrick_3.0", "fitspatrick_1.0", "fitspatrick_2.0",
#                      "fitspatrick_4.0", "fitspatrick_5.0", "fitspatrick_6.0", "region_ARM", "region_NECK",
#                      "region_FACE", "region_HAND", "region_FOREARM", "region_CHEST", "region_NOSE", "region_THIGH",
#                      "region_SCALP", "region_EAR", "region_BACK", "region_FOOT", "region_ABDOMEN", "region_LIP",
#                      "diameter_1", "diameter_2", "itch_False", "itch_True", "itch_UNK", "grew_False", "grew_True",
#                      "grew_UNK", "hurt_False", "hurt_True", "hurt_UNK", "changed_False", "changed_True",
#                      "changed_UNK", "bleed_False", "bleed_True", "bleed_UNK", "elevation_False", "elevation_True",
#                      "elevation_UNK"]

meta_data_columns = ["age", "region_ARM", "region_NECK", "region_FACE", "region_HAND", "region_FOREARM",
                     "region_CHEST", "region_NOSE", "region_THIGH", "region_SCALP", "region_EAR", "region_BACK",
                     "region_FOOT", "region_ABDOMEN", "region_LIP", "gender_FEMALE", "gender_MALE"]

# Loading test data
print("- Loading test data...")
csv_test = pd.read_csv(_csv_path_test)
test_imgs_id = csv_test['img_id'].values
test_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in test_imgs_id]
test_labels = csv_test['diagnostic_number'].values
if _use_meta_data:
    test_meta_data = csv_test[meta_data_columns].values
    print("-- Using {} meta-data features".format(len(meta_data_columns)))
else:
    test_meta_data = None
    print("-- No metadata")

transform_param = get_norm_and_size(_model_name)

test_data_loader = get_data_loader (test_imgs_path, test_labels, test_meta_data, transform=ImgEvalTransform(*transform_param),
                                   batch_size=30, shuf=False, num_workers=16, pin_memory=True)

model = set_model(_model_name, len(_labels_name), neurons_reducer_block=_neurons_reducer_block,
                  comb_method=_comb_method, comb_config=_comb_config, pretrained=False)

# loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor().cuda())


_metric_options = {
        'save_all_path': os.path.join(_check_base_path, "test_pred_2"),
        'pred_name_scores': 'predictions.csv',
        'normalize_conf_matrix': True}

test_model(model, test_data_loader, checkpoint_path=_checkpoint_path, metrics_to_comp="all", class_names=_labels_name,
           metrics_options=_metric_options, save_pred=True, verbose=False, apply_softmax=False)


