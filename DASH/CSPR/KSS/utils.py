import os
from longling import json_load, path_append, abs_current_dir
from DASH.CSPR.KES.KS import KS
import csv
from longling import as_io

"""
Example
-------
>>> load_item("filepath_that_do_not_exsit/not_exsit77250")
{}
"""


def load_ks_from_csv(edges):
    with as_io(edges) as f:
        for line in csv.reader(f, delimiter=","):
            yield line


def load_items(filepath):
    if os.path.exists(filepath):
        return json_load(filepath)
    else:
        return {}


def load_knowledge_structure(filepath):
    knowledge_structure = KS()
    knowledge_structure.add_edges_from([list(map(int, edges)) for edges in load_ks_from_csv(filepath)])
    return knowledge_structure


def load_learning_order(filepath):
    return json_load(filepath)


def load_configuration(filepath):
    return json_load(filepath)


def load_environment_parameters(directory=None):
    if directory is None:
        directory = path_append(abs_current_dir(__file__), "meta_data")
    return {
        "configuration": load_configuration(path_append(directory, "configuration.json")),
        "knowledge_structure": load_knowledge_structure(path_append(directory, "knowledge_structure.csv")),
        "learning_order": load_learning_order(path_append(directory, "learning_order.json")),
        "items": load_items(path_append(directory, "items.json"))
    }
