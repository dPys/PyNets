#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2016
@authors: Derek Pisner
"""
from pynets.stats.prediction import *


def main():
    import json
    import pandas as pd
    import os
    import sys
    import dill
    from pynets.stats.utils import make_feature_space_dict, \
        make_subject_dict, cleanNullTerms
    from pynets.core.utils import mergedicts
    from colorama import Fore, Style
    try:
        import pynets
    except ImportError:
        print(
            "PyNets not installed! Ensure that you are referencing the correct"
            " site-packages and using Python3.6+"
        )

    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h"
              " flag.\n")
        sys.exit(1)

    base_dir = "/working/tuning_set/outputs_final"
    df = pd.read_csv(
        "/working/tuning_set/outputs_shaeffer/df_rum_persist_all.csv",
        index_col=False
    )

    # User-Specified #
    embedding_type = 'OMNI'
    modality = "dwi"
    target_vars = ["rumination_persist_phenotype",
                   "depression_persist_phenotype",
                   "dep_2", 'rum_2', 'rum_1', 'dep_1']

    rsns = ["triple", "kmeans", "language"]

    sessions = ["1"]

    # Hard-Coded #
    thr_type = "MST"
    template = "MNI152_T1"
    mets = [
        "global_efficiency",
        "average_shortest_path_length",
        "average_degree_centrality",
        "average_eigenvector_centrality",
        "average_betweenness_centrality",
        "modularity",
        "smallworldness",
    ]
    hyperparams_func = ["rsn", "res", "model", "hpass", "extract", "smooth"]
    hyperparams_dwi = ["rsn", "res", "model", "directget", "minlength", "tol"]

    subject_dict_file_path = (
        f"{base_dir}/pynets_subject_dict_{modality}_{'_'.join(rsns)}_"
        f"{embedding_type}_{template}_{thr_type}.pkl"
    )
    subject_mod_grids_file_path = (
        f"{base_dir}/pynets_modality_grids_{modality}_{'_'.join(rsns)}_"
        f"{embedding_type}_{template}_{thr_type}.pkl"
    )
    missingness_summary = (
        f"{base_dir}/pynets_missingness_summary_{modality}_{'_'.join(rsns)}_"
        f"{embedding_type}_{template}_{thr_type}.csv"
    )

    if not os.path.isfile(subject_dict_file_path) or not os.path.isfile(
        subject_mod_grids_file_path
    ):
        subject_dict, modality_grids, missingness_frames = make_subject_dict(
            [modality], base_dir, thr_type, mets, [embedding_type], template,
            sessions, rsns
        )
        sub_dict_clean = cleanNullTerms(subject_dict)
        missingness_frames = [i for i in missingness_frames if
                              isinstance(i, pd.DataFrame)]
        if len(missingness_frames) != 0:
            if len(missingness_frames) > 1:
                final_missingness_summary = pd.concat(missingness_frames)
            elif len(missingness_frames) == 1:
                final_missingness_summary = missingness_frames[0]
            final_missingness_summary.to_csv(missingness_summary, index=False)
        with open(subject_dict_file_path, "wb") as f:
            dill.dump(sub_dict_clean, f)
        f.close()
        with open(subject_mod_grids_file_path, "wb") as f:
            dill.dump(modality_grids, f)
        f.close()
    else:
        with open(subject_dict_file_path, "rb") as f:
            sub_dict_clean = dill.load(f)
        f.close()
        with open(subject_mod_grids_file_path, "rb") as f:
            modality_grids = dill.load(f)
        f.close()

    # Subset only those participants which have usable data
    for ID in df["participant_id"]:
        if len(ID) == 1:
            df.loc[df.participant_id == ID, "participant_id"] = "s00" + str(ID)
        if len(ID) == 2:
            df.loc[df.participant_id == ID, "participant_id"] = "s0" + str(ID)

    df = df[df["participant_id"].isin(list(sub_dict_clean.keys()))]
    df['sex'] = df['sex'].map({1:0, 2:1})
    df = df[
        ["participant_id", "age", "num_visits", "sex"] + target_vars
    ]

    ml_dfs_dict = {}
    ml_dfs_dict[modality] = {}
    dict_file_path = f"{base_dir}/pynets_ml_dict_{modality}_" \
                     f"{'_'.join(rsns)}_{embedding_type}_{template}_" \
                     f"{thr_type}.pkl"
    if not os.path.isfile(dict_file_path) or not \
        os.path.isfile(dict_file_path):
        ml_dfs = {}
        ml_dfs = make_feature_space_dict(
            base_dir,
            ml_dfs,
            df,
            modality,
            sub_dict_clean,
            sessions[0],
            modality_grids,
            embedding_type,
            mets
        )

        with open(dict_file_path, "wb") as f:
            dill.dump(ml_dfs, f)
        f.close()
        ml_dfs_dict[modality][embedding_type] = dict_file_path
        del ml_dfs
    else:
        ml_dfs_dict[modality][embedding_type] = dict_file_path

    outs = []
    with open(ml_dfs_dict[modality][embedding_type], "rb") as f:
        outs.append(dill.load(f))
    f.close()

    ml_dfs = outs[0]
    for d in outs:
        ml_dfs = dict(mergedicts(ml_dfs, d))

    ml_dfs = cleanNullTerms(ml_dfs)

    feature_spaces = {}

    iter = f"{modality}_{embedding_type}"
    out_dict = {}
    for recipe in ml_dfs[modality][embedding_type].keys():
        try:
            out_dict[str(recipe)] = ml_dfs[modality][embedding_type][recipe].to_json()
        except:
            print(f"{recipe} recipe not found...")
            continue
    out_json_path = f"{base_dir}/{iter}.json"
    if os.path.isfile(out_json_path):
        os.remove(out_json_path)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)
    f.close()
    feature_spaces[iter] = out_json_path

    del ml_dfs

    args = {}
    args["base_dir"] = base_dir
    args["feature_spaces"] = feature_spaces
    args["modality_grids"] = modality_grids
    args["target_vars"] = target_vars
    args["embedding_type"] = embedding_type
    args["modality"] = modality

    return args


if __name__ == "__main__":
    import warnings
    import sys
    import gc
    import json
    from multiprocessing import set_start_method, Process, Manager
    from pynets.stats.prediction import build_predict_workflow

    try:
        set_start_method("forkserver")
    except:
        pass
    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"
    args = main()

    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_predict_workflow, args=(args, retval))
        p.start()
        p.join()

        if p.exitcode != 0:
            sys.exit(p.exitcode)

        gc.collect()
    mgr.shutdown()
