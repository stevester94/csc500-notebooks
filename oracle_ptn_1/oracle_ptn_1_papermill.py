#! /usr/bin/env python3


import papermill
import json
import os

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

###########################################
# papermill parameters
###########################################
EXPERIMENTS_PATH = "./experiments"
RESULTS_DIR_NAME = "results"
NOTEBOOK_OUT_NAME = "experiment.ipynb"
NOTEBOOK_TEMPLATE_PATH = os.path.realpath("../templates/oracle_ptn_template.ipynb")

###########################################
# Build all experiment json parameters
###########################################
base_parameters = {}
base_parameters["experiment_name"] = "oracle_ptn_1"
base_parameters["lr"] = 0.0001
base_parameters["device"] = "cuda"

base_parameters["seed"] = 1337
base_parameters["desired_classes_source"] = ALL_SERIAL_NUMBERS
base_parameters["desired_classes_target"] = ALL_SERIAL_NUMBERS

# base_parameters["source_domains"] = [38,]
# base_parameters["target_domains"] = [20,44,
#     2,
#     8,
#     14,
#     26,
#     32,
#     50,
#     56,
#     62
# ]

base_parameters["num_examples_per_class_per_domain_source"]=100
base_parameters["num_examples_per_class_per_domain_target"]=100

base_parameters["n_shot"] = 3
base_parameters["n_way"]  = len(base_parameters["desired_classes_source"])
base_parameters["n_query"]  = 2
base_parameters["train_k_factor"] = 1
base_parameters["val_k_factor"] = 2
base_parameters["test_k_factor"] = 2


base_parameters["n_epoch"] = 3

base_parameters["patience"] = 10
base_parameters["criteria_for_best"] = "source"
base_parameters["normalize_source"] = False
base_parameters["normalize_target"] = False


base_parameters["x_net"] =     [
    {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 256]}},
    {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":(1,7), "bias":False, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":256}},

    {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":(2,7), "bias":True, "padding":(0,3), },},
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
base_parameters["BEST_MODEL_PATH"] = "./best_model.pth"


parameters = base_parameters

# These will get permuted so we cover every possible case
custom_parameters = {}
custom_parameters["seed"] = [1337, 1984, 2020, 18081994, 4321326]

experiment_jsons = []
import copy
import itertools
keys, values = zip(*custom_parameters.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

#([], [2,8,14,20,26,32,38,44,50,56,62]),
for source_domains, target_domains in [
    ([2], [8,14,20,26,32,38,44,50,56,62]),
    #([8], [2,14,20,26,32,38,44,50,56,62]),
    ([14], [2,8,20,26,32,38,44,50,56,62]),
    #([20], [2,8,14,26,32,38,44,50,56,62]),
    ([26], [2,8,14,20,32,38,44,50,56,62]),
    #([32], [2,8,14,20,26,38,44,50,56,62]),
    ([38], [2,8,14,20,26,32,44,50,56,62]),
    #([44], [2,8,14,20,26,32,38,50,56,62]),
    ([50], [2,8,14,20,26,32,38,44,56,62]),
    #([56], [2,8,14,20,26,32,38,44,50,62]),
    ([62], [2,8,14,20,26,32,38,44,50,56]),
]:
    for d in permutations_dicts:
        parameters = copy.deepcopy(base_parameters)

        for key,val in d.items():
            parameters[key] = val

        parameters["source_domains"] = source_domains
        parameters["target_domains"] = target_domains
        
        j = json.dumps(parameters, indent=2)
        experiment_jsons.append(j)

import random
random.seed(1337)
random.shuffle(experiment_jsons)

###########################################
# Run each experiment using papermill
###########################################

"""
Slightly confusing. We change directory to each experiment dir when 
running since papermill uses the cwd, not where the notebook being executed lives.
We save the original cwd of the script so we can cd back up before cd'ing into the next
experiment.

It's basically pushd and popd
"""
script_original_cwd = os.getcwd()

os.mkdir(EXPERIMENTS_PATH)

for i, experiment_json in enumerate(experiment_jsons[:2]):
    os.chdir(script_original_cwd)
    experiment_path = os.path.join(
        EXPERIMENTS_PATH,
        f"{i+1}"
    )
    os.mkdir(experiment_path)

    os.chdir(experiment_path)

    papermill.execute_notebook(
        NOTEBOOK_TEMPLATE_PATH,
        NOTEBOOK_OUT_NAME,
        parameters = {"parameters": json.loads(experiment_json)}
    )