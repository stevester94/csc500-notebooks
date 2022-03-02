#! /usr/bin/env python3

import os

from steves_utils.wisig.utils import (
    ALL_NODES_MINIMUM_100_EXAMPLES,
    ALL_NODES_MINIMUM_500_EXAMPLES,
    ALL_NODES_MINIMUM_1000_EXAMPLES,
    ALL_DAYS
)
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
base_parameters["experiment_name"] = "baseline_ptn_wisig"
base_parameters["lr"] = 0.001
base_parameters["device"] = "cuda"

base_parameters["seed"] = 1337
base_parameters["dataset_seed"] = 1337

base_parameters["labels_source"] = ALL_NODES_MINIMUM_100_EXAMPLES
base_parameters["labels_target"] = ALL_NODES_MINIMUM_100_EXAMPLES


base_parameters["x_transforms_source"]       = []
base_parameters["x_transforms_target"]       = []
base_parameters["episode_transforms_source"] = []
base_parameters["episode_transforms_target"] = []

base_parameters["num_examples_per_domain_per_label_source"]=100
base_parameters["num_examples_per_domain_per_label_target"]=100

base_parameters["n_shot"] = 3
base_parameters["n_way"]  = len(base_parameters["labels_source"])
base_parameters["n_query"]  = 2
base_parameters["train_k_factor"] = 1
base_parameters["val_k_factor"] = 2
base_parameters["test_k_factor"] = 2

base_parameters["torch_default_dtype"] = "torch.float64" 

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

base_parameters["pickle_name"] = "wisig.node3-19.stratified_ds.2022A.pkl"

parameters = base_parameters

# These will get permuted so we cover every possible case
custom_parameters = {}
custom_parameters["seed"] = [1337]

repetitions = 3

trials = []
import copy
import itertools
keys, values = zip(*custom_parameters.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

# ([], [1,2,3,4,5]),
for i in range(repetitions):
    for domains_source, domains_target in [
        ([3], [1,2,4,5]),
    ]:
        for d in permutations_dicts:
            parameters = copy.deepcopy(base_parameters)

            for key,val in d.items():
                parameters[key] = val

            parameters["domains_source"] = domains_source
            parameters["domains_target"] = domains_target
            
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