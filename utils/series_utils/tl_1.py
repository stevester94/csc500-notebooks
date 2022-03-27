#! /usr/bin/env python3
import pandas as pd
import os
import seaborn as sb

from steves_utils.summary_utils import (
    get_experiments_from_path
)

from steves_utils.utils_v2 import (
    get_experiments_base_path
)

PTN_SERIES = [
    "tuned_1v2",
    "baseline_ptn_32bit",
    "baseline_ptn",
    # "tl_3",
    # "tl_1",

    "hyperparameters_3",
    # "cnn_2",
    # "cnn_3",
    "tuned_1",
    # "cnn_1",
    "hyperparameters_2",
    "hyperparameters_1",


    # "tl_2",
]

CNN_SERIES = [

]

TL_SERIES = [
    # "tl_2v2",
    # "tl_3v2",
    "tl_1v2",
]

class tl_1_Helper:
    def __init__(self, series_path = os.path.join(get_experiments_base_path(), "tl_1v2")):
        self.series_name = "tl_1v2"
        self.series_path = series_path

        self.independent_vars = ["source_name", "target_name", "x_transform"]

        self.result_columns = [
            'source_test_label_accuracy',
            'source_test_label_loss',
            'target_test_label_accuracy',
            'target_test_label_loss',
            'source_val_label_accuracy',
            'source_val_label_loss',
            'target_val_label_accuracy',
            'target_val_label_loss',
            'total_experiment_time_secs',
            'per_domain_accuracy',
            'history',
            'total_epochs_trained'
            'confusion'    
        ]
        self.parameter_columns_minus_seed = [
            'device'
            'lr'
            'n_shot'
            'n_query'
            'train_k_factor'
            'val_k_factor'
            'test_k_factor'
            'torch_default_dtype'
            'n_epoch'
            'patience'
            'criteria_for_best'
            'n_way'
            'datasets'
            'x_shape'
        ]

        self.eh_columns = [
            'dataset_metrics',
            'x_net',
            'experiment_name',
            'series_name',
            'dataset_seed',
            'seed',
            'NUM_LOGS_PER_EPOCH'
            'BEST_MODEL_PATH'
        ]
    
    # For posterity, the below is only going to work on single dataset source + single dataset target series
    def get_all_trials(self):
        all_trials = []

        trials = get_experiments_from_path(
            self.series_path
        )

        for t in trials:
            t["series_name"] = self.series_name
            for key, value in t["parameters"].items():
                t[key] = value
            del t["parameters"]

            for key, value in t["results"].items():
                t[key] = value
            del t["results"]

            for key, value in t.items():
                if type(value) == list:
                    t[key] = tuple(value)
            
            all_trials.append(t)
        
        tl = pd.DataFrame(all_trials)

        tl = tl.drop(columns=["domains_source", "domains_target"]) # These are bogus parameters I've injected in trial notebooks
        tl.columns

        # Get the trials only from the series we want
        p = tl[tl["series_name"].isin(["tl_1v2"])]
        p = p.reset_index(drop=True)

        pickle_name_mapping = {
            "cores.stratified_ds.2022A.pkl": "cores",
            "metehan.stratified_ds.2022A.pkl": "metehan",
            "oracle.Run1_10kExamples_stratified_ds.2022A.pkl": "oracle.run1",
            "oracle.Run1_framed_2000Examples_stratified_ds.2022A.pkl": "oracle.run1.framed",
            "oracle.Run2_10kExamples_stratified_ds.2022A.pkl": "oracle.run2",
            "oracle.Run2_framed_2000Examples_stratified_ds.2022A.pkl": "oracle.run2.framed",
            "wisig.node3-19.stratified_ds.2022A.pkl": "wisig",
        }

        """
        This requires some knowledge of how the series was structured
        For instance, we know that if original oracle was used, but not the max amount of 
        items then it should be called oracle.*.limited

        We also know that the x_transforms are the same between source and target
        """
        def datasets_to_columns(datasets):
            assert len(datasets) == 2

            if datasets[0]["source_or_target_dataset"] == "source":
                source = datasets[0]
                target = datasets[1]
            else:
                target = datasets[0]
                source = datasets[1]
            
            source_name = os.path.basename(source["pickle_path"])
            target_name = os.path.basename(target["pickle_path"])
            
            source_name = pickle_name_mapping[source_name]
            target_name = pickle_name_mapping[target_name]

            if source_name in ["oracle.run1", "oracle.run2"]:
                if source["num_examples_per_domain_per_label"] not in [-1, 10000]:
                    assert source["num_examples_per_domain_per_label"] == 2000
                    source_name = source_name+".limited"

            if target_name in ["oracle.run1", "oracle.run2"]:
                if target["num_examples_per_domain_per_label"] not in [-1, 10000]:
                    assert target["num_examples_per_domain_per_label"] == 2000
                    target_name = target_name+".limited"


            assert source["x_transforms"] == target["x_transforms"]
            transforms = source["x_transforms"]
            assert len(transforms) == 1 or len(transforms) == 0

            if len(transforms) == 0:
                transforms = "None"
            else:
                transforms = str(transforms[0])

            return source_name, target_name, transforms

        # p["datasets"] = p.apply(lambda row : str(row["datasets"]), axis=1)
        p[["source_name", "target_name", "x_transform"]] = [datasets_to_columns(r["datasets"]) for i,r in p.iterrows()]

        # Ensure that the indpendent vars are the only variables that are changing between trials
        control_vars = list(set(self.parameter_columns_minus_seed) - set(self.independent_vars+["seed", "dataset_seed"]))
        for i, size in p.groupby(self.independent_vars+["seed", "dataset_seed"]).size().iteritems():
            assert size == 1

        p["Transfer"] = p.apply(lambda row : str(row["source_name"]) + "->" + str(row["target_name"]), axis=1)
        p = p.rename(columns={"source_val_label_accuracy": "Source Val Accuracy", "target_val_label_accuracy": "Target Val Accuracy"})
        
        return p

    def get_best_trials(self):
        t = self.get_all_trials()

        idx = t.groupby(self.independent_vars)["Target Val Accuracy"].transform(max) == t["Target Val Accuracy"] # This will return _ALL_ rows with max in a group
        maxed = t[idx]
        maxed = maxed.groupby(self.independent_vars).first() # If there are multiple of max target val acc, we just drop the others
        maxed = maxed.reset_index()
        return maxed
        # return t[idx]
        # i = t.set_index(independent_vars)
        i[["Target Val Accuracy", "Source Val Accuracy"]] = t.groupby(independent_vars)[["Target Val Accuracy", "Source Val Accuracy"]].max()
        i = i[["Target Val Accuracy", "Source Val Accuracy"]]
        i = i.drop_duplicates()
        i = i.reset_index()
        return i



if __name__ == "__main__":
    pass