#! /usr/bin/env python3

import os

from steves_utils.ORACLE.utils_v2 import (ALL_DISTANCES_FEET, ALL_RUNS,
                                          ALL_SERIAL_NUMBERS,
                                          serial_number_to_id)
from steves_utils.papermill_support import run_trials_with_papermill


###########################################
# papermill parameters
###########################################
TRIALS_PATH = "./trials"
NOTEBOOK_OUT_NAME = "trial.ipynb"
NOTEBOOK_TEMPLATE_PATH = os.path.realpath("../../../templates/cnn_template.ipynb")
BEST_MODEL_PATH = "./best_model.pth"
SAVE_BEST_MODEL=False

###########################################
# Build all experiment json parameters
###########################################
base_parameters = {}
base_parameters["experiment_name"] = "cnn_2:oracle.run2"
base_parameters["labels"] = ALL_SERIAL_NUMBERS
base_parameters["domains_source"] = [8,32,50,14,20,26,38,44,]
base_parameters["domains_target"] = [8,32,50,14,20,26,38,44,]
base_parameters["pickle_name_source"] = "oracle.Run2_10kExamples_stratified_ds.2022A.pkl"
base_parameters["pickle_name_target"] = "oracle.Run2_10kExamples_stratified_ds.2022A.pkl"

base_parameters["device"] = "cuda"

base_parameters["lr"] = 0.0001



base_parameters["batch_size"] = 128


base_parameters["normalize_source"] = False
base_parameters["normalize_target"] = False

base_parameters["num_examples_per_domain_per_label_source"]=-1
base_parameters["num_examples_per_domain_per_label_target"]=-1

base_parameters["torch_default_dtype"] = "torch.float32" 

base_parameters["n_epoch"] = 50

base_parameters["patience"] = 3
base_parameters["criteria_for_best"] = "target_accuracy"


base_parameters["x_net"] =     [
    {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 256]}},
    {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":[1,7], "bias":False, "padding":[0,3], },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":256}},

    {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":[2,7], "bias":True, "padding":[0,3], },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":80}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 80*256, "out_features": 256}}, # 80 units per IQ pair
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm1d", "kargs": {"num_features":256}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": len(base_parameters["labels"])}},
]

# Parameters relevant to results
# These parameters will basically never need to change
base_parameters["NUM_LOGS_PER_EPOCH"] = 10
base_parameters["BEST_MODEL_PATH"] = BEST_MODEL_PATH



parameters = base_parameters

custom_parameters = []

for seed in [1337, 420, 154325, 7, 500]:
    custom_parameters.extend([
        {   
            "dataset_seed": seed,
            "seed": seed
        },  
        {   
            "dataset_seed": seed,
            "seed": seed
        },  
        {   
            "dataset_seed": seed,
            "seed": seed
        }   
    ])  


trials = []
import copy

for custom in custom_parameters:
    parameters = copy.deepcopy(base_parameters)

    for key,val in custom.items():
        parameters[key] = val

    trials.append(parameters)

import random

random.seed(1337)
random.shuffle(trials)

###########################################
# Run each experiment using papermill
###########################################

run_trials_with_papermill(
    trials=trials,
    trials_dir_path=TRIALS_PATH,
    notebook_out_name=NOTEBOOK_OUT_NAME,
    notebook_template_path=NOTEBOOK_TEMPLATE_PATH,
    best_model_path=BEST_MODEL_PATH,
    save_best_model=False
)
