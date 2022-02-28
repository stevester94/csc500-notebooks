#! /usr/bin/env python3

import json
import matplotlib.pyplot as plt

from steves_utils.ptn_do_report import (
    get_jig_diagram,
    get_results_table,
    get_parameters_table,
    get_domain_accuracies,
)

PATH = "/mnt/wd500GB/CSC500/csc500-main/csc500-notebooks/templates/oracle_ptn_template.ipynb"

with open(PATH, "r") as f:
    j = json.load(f)

for cell in j["cells"]:
    if "tags" in cell["metadata"] and cell["metadata"]["tags"] == ["experiment_json"]:
        assert len(cell["outputs"]) == 1
        assert len(cell["outputs"][0]["data"]["text/plain"]) == 1

        break

# The string itself is surrounded by single quotes. I'm being lazy here and just eval'ing it
experiment_json = eval(cell["outputs"][0]["data"]["text/plain"][0])
experiment = json.loads(experiment_json)
print(experiment)

get_jig_diagram(experiment)
plt.show()
get_results_table(experiment)
plt.show()
get_parameters_table(experiment)
plt.show()
get_domain_accuracies(experiment)
plt.show()