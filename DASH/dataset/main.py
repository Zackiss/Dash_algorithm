# coding: utf-8
# 2019/11/20 @ tongshiwei


__all__ = ["extract_relations", "build_json_sequence"]

from longling import path_append
from junyi import build_knowledge_graph
from knowledge_tracing import select_n_most_frequent_students


def extract_relations(src_root: str = "./dataset_working/", tar_root: str = "./dataset_cleaned/"):
    build_knowledge_graph(
        src_root, tar_root,
        ku_dict_path="graph_vertex.json",
        prerequisite_path="prerequisite.json",
        similarity_path="similarity.json",
        difficulty_path="difficulty.json",
    )


def build_json_sequence(src_root: str = "./dataset_working/", tar_root: str = "./dataset_cleaned/",
                        ku_dict_path: str = "./dataset_cleaned/graph_vertex.json", n: int = 1000):
    select_n_most_frequent_students(
        path_append(src_root, "junyi_ProblemLog_original.csv", to_str=True),
        path_append(tar_root, "student_log_kt_", to_str=True),
        ku_dict_path,
        n,
    )


extract_relations()
build_json_sequence()
