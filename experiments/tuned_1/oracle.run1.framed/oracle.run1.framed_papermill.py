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
NOTEBOOK_TEMPLATE_PATH = os.path.realpath("../../../templates/ptn_template.ipynb")
BEST_MODEL_PATH = "./best_model.pth"
SAVE_BEST_MODEL=False

###########################################
# Build all experiment json parameters
###########################################
base_parameters = {}
base_parameters["experiment_name"] = "tuned_1_oracle.run1.framed"
base_parameters["device"] = "cuda"

base_parameters["lr"] = 0.001
base_parameters["seed"] = 1337
base_parameters["dataset_seed"] = 1337

base_parameters["labels_source"] = ALL_SERIAL_NUMBERS
base_parameters["labels_target"] = ALL_SERIAL_NUMBERS


base_parameters["x_transforms_source"]       = []
base_parameters["x_transforms_target"]       = []
base_parameters["episode_transforms_source"] = []
base_parameters["episode_transforms_target"] = []

base_parameters["domains_source"] = [8,32,50]
base_parameters["domains_target"] = [14,20,26,38,44,]

base_parameters["num_examples_per_domain_per_label_source"]=1300
base_parameters["num_examples_per_domain_per_label_target"]=1300

base_parameters["n_shot"] = 3
base_parameters["n_way"]  = len(base_parameters["labels_source"])
base_parameters["n_query"]  = 2
base_parameters["train_k_factor"] = 3
base_parameters["val_k_factor"] = 2
base_parameters["test_k_factor"] = 2

base_parameters["torch_default_dtype"] = "torch.float32" 

base_parameters["n_epoch"] = 50

base_parameters["patience"] = 3
base_parameters["criteria_for_best"] = "target_loss"


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

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
]

# Parameters relevant to results
# These parameters will basically never need to change
base_parameters["NUM_LOGS_PER_EPOCH"] = 10
base_parameters["BEST_MODEL_PATH"] = BEST_MODEL_PATH

base_parameters["pickle_name"] = "oracle.Run1_framed_1300Examples_stratified_ds.2022A.pkl"

parameters = base_parameters

custom_parameters = []

custom_parameters.extend([
    {
        "x_transforms_source": [],
        "x_transforms_target": []
    },
    {
        "x_transforms_source": ["unit_mag"],
        "x_transforms_target": ["unit_mag"]
    },
    {
        "x_transforms_source": ["unit_power"],
        "x_transforms_target": ["unit_power"]
    },
    {
        "x_transforms_source": ["times_zero"],
        "x_transforms_target": ["times_zero"]
    },

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
