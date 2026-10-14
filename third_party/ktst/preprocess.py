import argparse
import importlib
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question


DATASETS: dict[str, dict[str, str | list[str]]] = {
    "algebra2005": {
        "url": "https://pslcdatashop.web.cmu.edu/KDDCup/",
        "files": ["algebra_2005_2006_train.txt"],
    },
    "assist2009": {
        "url": "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010",
        "files": ["skill_builder_data_corrected_collapsed.csv"],
    },
    "assist2015": {
        "url": "https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data",
        "files": ["2015_100_skill_builders_main_problems.csv"],
    },
    "bridge2algebra2006": {
        "url": "https://pslcdatashop.web.cmu.edu/KDDCup/",
        "files": ["bridge_to_algebra_2006_2007_train.txt"],
    },
    "ednet": {
        "url": "https://github.com/riiid/ednet => Download the complete folder with subfolders 'content' and 'KT1'",
        "files": [""],
    },
    "nips_task34": {
        "url": "https://eedi.com/projects/neurips-education-challenge => Also add the 'metadata' subfolder",
        "files": ["train_task_3_4.csv"],
    },
    "poj": {
        "url": "https://drive.google.com/drive/folders/1LRljqWfODwTYRMPw6wEJ_mMt1KZ4xBDk",
        "files": ["poj_log.csv"],
    },
    "statics2011": {
        "url": "https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507",  # Other channel: https://drive.google.com/drive/folders/1CqYkJWno9oh0kPAt3eHRWowFR6BqeQsS
        "files": ["AllData_student_step_2011F.csv"],
    },
}

# write description used in readme + help for cli
description = f"""Preprocess datasets with pykt-toolkit.
First, download the necessary files from the provided URLs. After you have successfully downloaded the files, store them in your designated `file_path`.
Make sure to organize these files into folders that correspond to the specific dataset's name.
While preprocessing, information about the dataset is written to config.json in the root of the `file_path`.\n\n"""
for n, d in DATASETS.items():
    description += (
        f"""{n:20}{d.get('url')}\n{'':19} (Relevant files: {d.get('files', [])})\n"""
    )


def get_args(description) -> tuple[str, str, int, int, int]:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="file path to your data folder",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="name of the dataset",
    )
    parser.add_argument(
        "-m",
        "--min_len",
        type=int,
        help="min length of sequences (default: 3)",
        default=3,
    )
    parser.add_argument(
        "-l",
        "--max_len",
        type=int,
        help="max length of a sequence before being split (default: 200)",
        default=200,
    )
    parser.add_argument(
        "-k",
        "--k_fold",
        type=int,
        help="num train/test folds (default: 5)",
        default=5,
    )
    args = parser.parse_args()
    return args.file_path, args.dataset, args.min_len, args.max_len, args.k_fold


def generate_selectmasks(select_only_last: bool, length: int) -> list[str]:
    if select_only_last:
        selectmasks = ["-1"] * (length - 1) + ["1"]
    else:
        selectmasks = ["1"] * (length)
    return selectmasks


def pad_with_minus_ones(x: list[str], max_len: int):
    return x + ["-1"] * (max_len - len(x))


if __name__ == "__main__":
    # init cli
    file_path, dataset, min_len, max_len, k_fold = get_args(description)

    # script ...
    path_data = str(Path(file_path) / dataset)
    path_src = Path(file_path) / dataset / DATASETS[dataset]["files"][0]
    path_tgt = Path(file_path) / dataset / "data.txt"
    path_cfg = Path(file_path) / "config.json"

    try:
        mod = importlib.import_module(f"pykt.preprocess.{dataset}_preprocess")
        read_data_from_csv = getattr(mod, "read_data_from_csv")
    except ModuleNotFoundError:
        raise ValueError(f"Cannot import functions from package `{dataset}`")

    if dataset == "ednet":
        path_data, path_tgt = read_data_from_csv(
            str(path_src), str(path_tgt), dataset_name=dataset
        )
    elif dataset == "nips_task34":
        metap = os.path.join(path_data, "metadata")
        read_data_from_csv(path_src, metap, "task_3_4", path_tgt)
    else:
        read_data_from_csv(path_src, path_tgt)

    ### PREPROCESS DATASETS WITH PYKT (this writes/overwrites files) ###

    # concept level models
    split_concept(
        dname=path_data,
        fname=path_tgt,
        dataset_name=dataset,
        configf=path_cfg,
        min_seq_len=min_len,
        maxlen=max_len,
        kfold=k_fold,
    )

    # question level models
    split_question(
        dname=path_data,
        fname=path_tgt,
        dataset_name=dataset,
        configf=path_cfg,
        min_seq_len=min_len,
        maxlen=max_len,
        kfold=k_fold,
    )

    # overwrite `test_window_sequences_quelevel.csv` files with length-corrected version
    test_quelevel = pd.read_csv(Path(file_path) / dataset / "test_quelevel.csv")

    # generate corrected sequences
    if "questions" in test_quelevel.columns and "concepts" in test_quelevel.columns:
        print(
            "Overwrite `test_window_sequences_quelevel.csv` files with length-corrected version ..."
        )
        fold = -1
        corrected_test_window_sequences_quelevel_list = []
        for i, row in test_quelevel.iterrows():
            assert row["fold"] == fold
            uid = row["uid"]
            questions = row["questions"].split(",")
            concepts = row["concepts"].split(",")
            responses = row["responses"].split(",")

            # get number of questions
            num_concepts = np.array([c.count("_") + 1 for c in concepts])
            assert len(num_concepts) == len(questions)

            # loop
            start = 0
            first_end_idx = (num_concepts.cumsum() > max_len).argmax()
            if first_end_idx == 0:
                first_end_idx = len(questions)
            for end in range(first_end_idx, len(questions) + 1):
                while sum(num_concepts[start:end]) > max_len:
                    start += 1

                corrected_test_window_sequences_quelevel_list.append(
                    dict(
                        fold=fold,
                        uid=uid,
                        questions=pad_with_minus_ones(
                            questions[start:end], max_len=max_len
                        ),
                        concepts=pad_with_minus_ones(
                            concepts[start:end], max_len=max_len
                        ),
                        responses=pad_with_minus_ones(
                            responses[start:end], max_len=max_len
                        ),
                        selectmasks=pad_with_minus_ones(
                            generate_selectmasks(
                                select_only_last=(end != first_end_idx),
                                length=end - start,
                            ),
                            max_len=max_len,
                        ),
                    )
                )

        # to DataFrame, lists as string + saving
        corrected_test_window_sequences_quelevel_df = pd.DataFrame(
            corrected_test_window_sequences_quelevel_list
        )
        for column in ["questions", "concepts", "responses", "selectmasks"]:
            corrected_test_window_sequences_quelevel_df[column] = (
                corrected_test_window_sequences_quelevel_df[column].apply(
                    lambda x: ",".join(x)
                )
            )

        corrected_target_path = Path(path_data) / "test_window_sequences_quelevel.csv"
        corrected_test_window_sequences_quelevel_df.to_csv(
            corrected_target_path, index=False
        )

    ### generate `unique_concept_mapping.pkl` (for KTST) ###

    # pd.read train_valid_sequences.csv, test.csv
    train_data = pd.read_csv(Path(file_path) / dataset / "train_valid_sequences.csv")
    test_data = pd.read_csv(Path(file_path) / dataset / "test.csv")

    # concat
    df = pd.concat([train_data, test_data])

    # get_tensor_dataset_from_pykt_dataset
    set_of_concept_sets = set()

    # Iterate over rows due to inhomogeneous shapes
    if "questions" in df.columns and "concepts" in df.columns:
        for i, row in df.iterrows():
            get_tensor = lambda x: np.array([int(s) for s in x.split(",")])
            q, c, r = (
                get_tensor(row[x]) for x in ["questions", "concepts", "responses"]
            )
            qic = get_tensor(row["is_repeat"]) == 1
            assert q.shape == c.shape == r.shape == qic.shape
            assert qic.ndim == 1
            qic[0] = (
                False  # this change is consistent with how we load data in __init__.py
            )
            m = q != -1
            q, c, r, qic = (
                q[m],
                c[m],
                r[m],
                qic[m],
            )  # masked tensors is what we work with in `format_question_combinatorial_dense`

            q_reduced = q[~qic]
            individual_questions = np.cumsum(~qic) - 1
            assert individual_questions.min() == 0

            # Make dense_concept_tensor
            list_of_concept_sets = [set() for _ in range(len(q_reduced))]
            # Add concepts of individual questions to separate lists
            for iq, ic in zip(individual_questions.tolist(), c.tolist()):
                list_of_concept_sets[iq].add(ic)
            for s in list_of_concept_sets:
                set_of_concept_sets.add(tuple(sorted(s)))

        assert (-1,) not in set_of_concept_sets
        set_of_concepts_to_unique_id = {}

        for i, concept_set in enumerate(set_of_concept_sets):
            set_of_concepts_to_unique_id[concept_set] = i

        # format_question_combinatorial_dense -> save mapping for unique concept combs.
        with open(Path(file_path) / dataset / "unique_concept_mapping.pkl", "wb") as f:
            pickle.dump(
                set_of_concepts_to_unique_id, f, protocol=pickle.HIGHEST_PROTOCOL
            )
