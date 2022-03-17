#! /usr/bin/env python3

import os
import copy

from steves_utils.CORES.utils import (
    ALL_NODES,
    ALL_NODES_MINIMUM_1000_EXAMPLES,
    ALL_DAYS
)

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_DISTANCES_FEET_NARROWED,
    ALL_RUNS,
    ALL_SERIAL_NUMBERS,
)

from steves_utils.utils_v2 import get_datasets_base_path
import steves_utils.wisig.utils as wisig

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
base_parameters["experiment_name"] = "tl_3A:oracle.run2.framed -> cores+wisig"
base_parameters["device"] = "cuda"

base_parameters["lr"] = 0.001
base_parameters["seed"] = 1337
base_parameters["dataset_seed"] = 1337

base_parameters["x_shape"] = [2,200]

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
    {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 200]}},
    {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":[1,7], "bias":False, "padding":[0,3], },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":256}},

    {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":[2,7], "bias":True, "padding":[0,3], },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":80}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 80*200, "out_features": 256}}, # 80 units per IQ pair
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
        "x_transforms": ["take_200"],
        "episode_transforms": [],
        "domain_prefix": "C_A_"
    },
    {
        "labels": wisig.ALL_NODES_MINIMUM_100_EXAMPLES,
        "domains": wisig.ALL_DAYS,
        "num_examples_per_domain_per_label": 100,
        "pickle_path": os.path.join(get_datasets_base_path(), "wisig.node3-19.stratified_ds.2022A.pkl"),
        "source_or_target_dataset": None, # Fill in later
        "x_transforms": ["take_200"],
        "episode_transforms": [],
        "domain_prefix": "W_A_"
    }
]

B_datasets = [
    {
        "labels": ALL_SERIAL_NUMBERS,
        "domains": ALL_DISTANCES_FEET_NARROWED,
        "num_examples_per_domain_per_label": 2000,
        "pickle_path": os.path.join(get_datasets_base_path(), "oracle.Run2_framed_2000Examples_stratified_ds.2022A.pkl"),
        "source_or_target_dataset": None, # Fill in later
        "x_transforms": ["take_200", "resample_20Msps_to_25Msps"],
        "episode_transforms": [],
        "domain_prefix": "ORACLE.run2_"
    }
]


base_parameters["n_way"]  = min(
    len(ALL_NODES),
    len(ALL_SERIAL_NUMBERS),
    len(wisig.ALL_NODES_MINIMUM_100_EXAMPLES)
)

custom_parameters = []

# I am far too burned out to find a pythonic way of doing this
for transform in ["unit_power", "unit_mag", None]:
    A_to_B = []
    for a_orig in A_datasets:
        a = copy.deepcopy(a_orig)
        a["source_or_target_dataset"] = "target"
        if transform != None: a["x_transforms"].insert(0, transform)
        A_to_B.append(a)

    for b_orig in B_datasets:
        b = copy.deepcopy(b_orig)
        b["source_or_target_dataset"] = "source"
        if transform != None: b["x_transforms"].insert(0, transform)
        A_to_B.append(b)
    
    custom_parameters.append(
        {
            "datasets": A_to_B
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