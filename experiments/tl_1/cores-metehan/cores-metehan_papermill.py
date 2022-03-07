#! /usr/bin/env python3

import os
import copy

from steves_utils.CORES.utils import (
    ALL_NODES,
    ALL_NODES_MINIMUM_1000_EXAMPLES,
    ALL_DAYS
)

from steves_utils.utils_v2 import get_datasets_base_path

from steves_utils.papermill_support import run_trials_with_papermill

###########################################
# papermill parameters
###########################################
TRIALS_PATH = "./trials"
NOTEBOOK_OUT_NAME = "trial.ipynb"
NOTEBOOK_TEMPLATE_PATH = os.path.realpath("../../../templates/tl_ptn_template.ipynb")
BEST_MODEL_PATH = "./best_model.pth"
SAVE_BEST_MODEL=False

###########################################
# Build all experiment json parameters
###########################################
base_parameters = {}
base_parameters["experiment_name"] = "tl_1_cores-metehan"
base_parameters["device"] = "cuda"

base_parameters["lr"] = 0.001
base_parameters["seed"] = 1337
base_parameters["dataset_seed"] = 1337


base_parameters["n_shot"] = 3
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

parameters = base_parameters



A_datasets = [
    {
        "labels": ALL_NODES,
        "domains": ALL_DAYS,
        "num_examples_per_domain_per_label": 100,
        "pickle_path": os.path.join(get_datasets_base_path(), "cores.stratified_ds.2022A.pkl"),
        "source_or_target_dataset": None, # Fill in later
        "x_transforms": [],
        "episode_transforms": [],
        "domain_prefix": "CORES_"
    }
]

B_datasets = [
    {
        "labels": list(range(19)),
        "domains": [0,1,2],
        "num_examples_per_domain_per_label": 100,
        "pickle_path": os.path.join(get_datasets_base_path(), "metehan.stratified_ds.2022A.pkl"),
        "source_or_target_dataset": None, # Fill in later
        "x_transforms": [],
        "episode_transforms": [],
        "domain_prefix": "Metehan_"
    }
]

base_parameters["n_way"]  = min(len(ALL_NODES), 19)

custom_parameters = []

# I am far too burned out to find a pythonic way of doing this
for transform in [["unit_power"], ["unit_mag"], []]:
    A_to_B = []
    for a_orig in A_datasets:
        a = copy.deepcopy(a_orig)
        a["source_or_target_dataset"] = "source"
        a["x_transforms"] = transform
        A_to_B.append(a)

    for b_orig in B_datasets:
        b = copy.deepcopy(b_orig)
        b["source_or_target_dataset"] = "target"
        b["x_transforms"] = transform
        A_to_B.append(b)
    
    custom_parameters.append(
        {
            "datasets": A_to_B
        }
    )

for transform in [["unit_power"], ["unit_mag"], []]:
    B_to_A = []
    for a_orig in A_datasets:
        a = copy.deepcopy(a_orig)
        a["source_or_target_dataset"] = "target"
        a["x_transforms"] = transform
        B_to_A.append(a)

    for b_orig in B_datasets:
        b = copy.deepcopy(b_orig)
        b["source_or_target_dataset"] = "source"
        b["x_transforms"] = transform
        B_to_A.append(b)
    
    custom_parameters.append(
        {
            "datasets": B_to_A
        }
    )

trials = []

for custom in custom_parameters:
    parameters = copy.deepcopy(base_parameters)

    for key,val in custom.items():
        parameters[key] = val

    trials.append(parameters)

###########################################
# Run each experiment using papermill
###########################################


# import pprint
# pp = pprint.PrettyPrinter()

# pp.pprint(trials)

run_trials_with_papermill(
    trials=trials,
    trials_dir_path=TRIALS_PATH,
    notebook_out_name=NOTEBOOK_OUT_NAME,
    notebook_template_path=NOTEBOOK_TEMPLATE_PATH,
    best_model_path=BEST_MODEL_PATH,
    save_best_model=False
)