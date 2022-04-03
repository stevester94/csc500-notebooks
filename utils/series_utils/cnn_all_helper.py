#! /usr/bin/env python3
import pandas as pd
import os

from steves_utils.summary_utils import (
    get_experiments_from_path
)

from steves_utils.utils_v2 import (
    get_experiments_base_path
)

class cnn_All_Helper:
    def __init__(self, which_cnn:int):
        self.series_name = "cnn_"+str(which_cnn)
        self.series_path = os.path.join(get_experiments_base_path(), "cnn_"+str(which_cnn))
        self.independent_vars = ["Source Dataset", "Target Dataset"]


    def get_all_trials(self):
        # raise Exception("Whatever")
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
        
        p = pd.DataFrame(all_trials)

        p = p.drop(columns=["domains_source", "domains_target"]) # These are bogus parameters I've injected in trial notebooks
        p.columns

        pickle_name_mapping = {
            "cores.stratified_ds.2022A.pkl": "cores",
            "metehan.stratified_ds.2022A.pkl": "metehan",
            "oracle.Run1_10kExamples_stratified_ds.2022A.pkl": "oracle.run1",
            "oracle.Run1_framed_2000Examples_stratified_ds.2022A.pkl": "oracle.run1.framed",
            "oracle.Run2_10kExamples_stratified_ds.2022A.pkl": "oracle.run2",
            "oracle.Run2_framed_2000Examples_stratified_ds.2022A.pkl": "oracle.run2.framed",
            "wisig.node3-19.stratified_ds.2022A.pkl": "wisig",
        }
        def row_to_columns(row):
            def _ds_name(ds):
                ds_name = pickle_name_mapping[ds]
                
                if ds_name in ["oracle.run1", "oracle.run2"]:
                    if row["num_examples_per_domain_per_label_source"] not in [-1, 10000]:
                        assert row["num_examples_per_domain_per_label_source"] == 2000
                        ds_name = ds_name+".limited"
                return ds_name

            ds_name_source = _ds_name(row["pickle_name_source"])
            ds_name_target = _ds_name(row["pickle_name_target"])

            return ds_name_source, ds_name_target


        p[["Source Dataset", "Target Dataset"]] = [row_to_columns(r) for i,r in p.iterrows()]

        # I screwed up CNN trials, and did 3 for each seec*dataset_seed
        # So just take the first one from each group, they're all equivalent
        p = p.groupby(self.independent_vars+["seed", "dataset_seed"]).nth(0)

        # p = p.set_index(self.independent_vars+["seed", "dataset_seed"])
        # import sys
        # for row in p.iterrows():
        #     # print(row)
        #     if len(row[1]) > 1:
        #         return row[0]
        # print("reee")
        # sys.exit(1)

        for i, size in p.groupby(self.independent_vars+["seed", "dataset_seed"]).size().iteritems():
            if not size == 1:
                print(i)
                raise Exception("I am in hell")

        p = p.rename(columns={"source_val_label_accuracy": "Source Val Accuracy", "target_val_label_accuracy": "Target Val Accuracy"})

        return p
    def get_best_trials(self):
        t = self.get_all_trials()

        idx = t.groupby(self.independent_vars)["Target Val Accuracy"].transform(max) == t["Target Val Accuracy"] # This will return _ALL_ rows with max in a group
        maxed = t[idx]
        maxed = maxed.groupby(self.independent_vars).first() # If there are multiple of max target val acc, we just drop the others
        maxed = maxed.reset_index()
        return maxed