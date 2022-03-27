import pandas as pd
import os
from pexpect import ExceptionPexpect
import seaborn as sb

from steves_utils.summary_utils import (
    get_experiments_from_path
)

from steves_utils.utils_v2 import (
    get_experiments_base_path
)

class tl_2_Helper:
    def __init__(self, series_path = os.path.join(get_experiments_base_path(), "tl_2v2")):
        self.series_name = "tl_2v2"
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

        print(self.series_path)

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

        # Get the trials only from the series we want
        p = tl.reset_index(drop=True)

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
            assert len(datasets) == 3

            names_source = []
            names_target = []

            transforms_source = []
            transforms_target = []
            for ds in datasets:
                # Parse the name, note the special attention to ORACLE so we can properly add ".limited" to the name
                ds_name = os.path.basename(ds["pickle_path"])
                ds_name = pickle_name_mapping[ds_name]

                if ds_name in ["oracle.run1", "oracle.run2"] and ds["num_examples_per_domain_per_label"] not in [-1, 10000]:
                    assert ds["num_examples_per_domain_per_label"] == 2000
                    ds_name = ds_name+".limited"
                
                # Parse the transforms. Only handle one or 0 transforms
                transforms = None
                if len(ds["x_transforms"]) == 0:
                    transforms = "None"
                elif len(ds["x_transforms"]) > 1:
                    raise Exception("Too many transforms, dying now")
                else:
                    transforms = str(ds["x_transforms"][0])

                if ds["source_or_target_dataset"] == "source":
                    names_source.append(ds_name)
                    transforms_source.append(transforms)
                else:
                    names_target.append(ds_name)
                    transforms_target.append(transforms)
            
            # tl_2v2, all x_transforms will be the same, but different counts
            assert set(transforms_source) == set(transforms_target)

            source_name = "+".join(names_source)
            target_name = "+".join(names_target)

            if len(set(transforms_source)) > 1:
                raise Exception("Too many transforms")
            if len(set(transforms_target)) > 1:
                raise Exception("Too many transforms")
            transforms = set(transforms_target).pop()

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