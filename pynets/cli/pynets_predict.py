#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2016
@authors: Derek Pisner
"""
from pynets.stats.prediction import *


def get_parser():
    """Parse command-line inputs"""
    import argparse
    from pynets.__about__ import __version__

    verstr = f"pynets v{__version__}"

    # Parse args
    parser = argparse.ArgumentParser(
        description="PyNets: A Fully-Automated Workflow for Reproducible"
                    " Functional and Structural Connectome Ensemble Learning")
    parser.add_argument(
        "-basedir",
        metavar="Output directory",
        help="Specify the path to the base output directory with group-level"
             " pynets derivatives.\n",
    )
    parser.add_argument(
        "-pheno",
        metavar="Phenotype Data",
        help="Path to a .csv or .pkl dataframe to define nuisance covariates and "
             "target variables\n",
    )
    parser.add_argument(
        "-modality",
        nargs=1,
        default="func",
        choices=["dwi", "func"],
        help="Specify data modality from which to collect data. Options are"
             " `dwi` and `func`. Currently, only one can be specified at a "
             "time. Default is functional",
    )
    parser.add_argument(
        "-session_label",
        help="""The label(s) of the session that should be analyzed.
        The label  corresponds to ses-<participant_label> from the BIDS spec
        (so it does not include "ses-"). If this parameter is not provided
        all sessions should be analyzed. Multiple sessions can be specified
         with a space separated list.""",
        nargs=1,
        default=None,
    )
    parser.add_argument(
        "-et",
        metavar="Embedding type",
        default="topology",
        nargs="+",
        choices=["ASE", "OMNI", "MASE", "eigenvector", "betweenness",
                 "clustering", "degree"],
        help="Specify the embedding method of interest.\n",
    )
    parser.add_argument(
        "-tv",
        metavar="Target Variable",
        default="age",
        nargs="+",
        help="Specify the target outcome variables of interest, "
             "separated by space.\n",
    )
    parser.add_argument(
        "-dc",
        metavar="Miscellaneous Columns to Drop",
        nargs="+",
        help="Specify column header names, separated by space.\n",
    )
    parser.add_argument(
        "-conf",
        metavar="Nuisance random effects",
        nargs="+",
        help="Specify column header names, separated by space.\n",
    )
    parser.add_argument(
        "-nets",
        metavar="Networks of Interest",
        nargs="+",
        help="Specify the names of the networks of interest.\n",
    )
    parser.add_argument(
        "-n_boots",
        metavar="Number of Bootstrapped Predictions for Monte Carlo Simulation",
        default=50,
        help="An integer >1. Default is 50.\n",
    )
    parser.add_argument(
        "-dr",
        default=False,
        action="store_true",
        help="Optionally use this flag if you wish to make random predictions"
             " with a dummy regressor.\n",
    )
    parser.add_argument(
        "-stack",
        default=False,
        action="store_true",
        help="Optionally use this flag if you wish to stack multiple "
             "estimators.\n",
    )
    parser.add_argument(
        "-sp",
        metavar="Miscellaneous Columns to Drop",
        nargs="+",
        help="Specify feature column header prefixes of shared feature spaces,"
             " separated by space. Only applicable if `-stack` flag is"
             " also used\n",
    )
    parser.add_argument(
        "-search",
        default="grid",
        nargs=1,
        choices=[
            "grid",
            "random"],
        help="Specify the GridSearchCV method to use. Default is `grid`.\n",
    )
    parser.add_argument(
        "-thrtype",
        default="MST",
        nargs=1,
        choices=[
            "MST",
            "PROP",
            "DISP"],
        help="Specify the thresholding method used when sampling the "
             "connectome ensemble. Default is MST.\n",
    )
    parser.add_argument(
        "-temp",
        metavar="MNI Template",
        default="any",
        nargs=1,
        choices=[
            "colin27",
            "MNI152_T1",
            "CN200",
            "any"
        ],
        help="Include this flag to specify a specific template"
             "if multiple were used to sample the connectome ensemble.\n",
    )
    parser.add_argument(
        "-pm",
        metavar="Cores,memory",
        default="auto",
        help="Number of cores to use, number of GB of memory to use for single"
             " subject run, entered as two integers seperated by comma. "
             "Otherwise, default is `auto`, which uses all resources detected"
             " on the current compute node.\n",
    )
    parser.add_argument(
        "-plug",
        metavar="Scheduler type",
        default="MultiProc",
        nargs=1,
        choices=[
            "Linear",
            "MultiProc",
            "SGE",
            "PBS",
            "SLURM",
            "SGEgraph",
            "SLURMgraph",
            "LegacyMultiProc",
        ],
        help="Include this flag to specify a workflow plugin other than the"
             " default MultiProc.\n",
    )
    parser.add_argument(
        "-v",
        default=False,
        action="store_true",
        help="Verbose print for debugging.\n")
    parser.add_argument(
        "-work",
        metavar="Working directory",
        default="/tmp/work",
        help="Specify the path to a working directory for pynets to run."
             " Default is /tmp/work.\n",
    )
    parser.add_argument("--version", action="version", version=verstr)
    return parser


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

    args = {}
    pre_args = get_parser().parse_args()
    args["base_dir"] = pre_args.basedir
    base_dir = args["base_dir"]
    n_boots = pre_args.n_boots
    args["target_vars"] = pre_args.tv
    target_vars = args["target_vars"]
    args["embedding_type"] = pre_args.et
    embedding_types = args["embedding_type"]
    args["modality"] = pre_args.modality[0]
    modality = args["modality"]
    thr_type = pre_args.thrtype
    template = pre_args.temp[0]
    data_file = pre_args.pheno
    drop_cols = pre_args.dc
    nuisance_cols = pre_args.conf
    dummy_run = pre_args.dr
    search_method = pre_args.search[0]
    stack = pre_args.stack
    stack_prefix_list = pre_args.sp

    if not drop_cols:
        drop_cols = []

    if not nuisance_cols:
        nuisance_cols = []

    rsns = pre_args.nets
    sessions = pre_args.session_label

    # Percent if subjects with usable data for a particular universe
    grid_thr = 0.75

    # mets = [
    #     "global_efficiency",
    #     "average_shortest_path_length",
    #     "average_degree_centrality",
    #     "average_eigenvector_centrality",
    #     "average_betweenness_centrality",
    #     "modularity",
    #     "degree_assortativity_coefficient"
    #     "smallworldness",
    # ]
    mets = []

    # import sys
    # print(f"rsns = {rsns}")
    # print(f"sessions = {sessions}")
    # print(f"base_dir = {base_dir}")
    # print(f"n_boots = {n_boots}")
    # print(f"target_vars = {target_vars}")
    # print(f"embedding_types = {embedding_types}")
    # print(f"modality = {modality}")
    # print(f"thr_type = {thr_type}")
    # print(f"template = {template}")
    # print(f"data_file = {data_file}")
    # print(f"drop_cols = {drop_cols}")
    # print(f"nuisance_cols = {nuisance_cols}")
    # sys.exit(0)

    hyperparams_func = ["rsn", "res", "model", "hpass", "extract", "smooth"]
    hyperparams_dwi = ["rsn", "res", "model", "directget", "minlength", "tol"]

    subject_dict_file_path = (
        f"{base_dir}/pynets_subject_dict_{modality}_{'_'.join(rsns)}_"
        f"{embedding_types}_{template}_{thr_type}.pkl"
    )
    subject_mod_grids_file_path = (
        f"{base_dir}/pynets_modality_grids_{modality}_{'_'.join(rsns)}_"
        f"{embedding_types}_{template}_{thr_type}.pkl"
    )
    missingness_summary = (
        f"{base_dir}/pynets_missingness_summary_{modality}_{'_'.join(rsns)}_"
        f"{embedding_types}_{template}_{thr_type}.csv"
    )

    if not os.path.isfile(subject_dict_file_path) or not os.path.isfile(
        subject_mod_grids_file_path
    ):
        subject_dict, modality_grids, missingness_frames = make_subject_dict(
            [modality], base_dir, thr_type, mets, embedding_types, template,
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

    # Load in data
    if data_file.endswith(".csv"):
        df = pd.read_csv(
            data_file,
            index_col=False
        )
    elif data_file.endswith(".pkl"):
        df = pd.read_pickle(data_file)
    else:
        raise ValueError("File format not recognized for phenotype data.")

    if 'tuning_set' in data_file or 'dysphoric' in data_file:
        for ID in df["participant_id"]:
            if len(ID) == 1:
                df.loc[df.participant_id == ID, "participant_id"] = "s00" +\
                                                                    str(ID)
            if len(ID) == 2:
                df.loc[df.participant_id == ID, "participant_id"] = "s0" + \
                                                                    str(ID)

    # Subset only those participants which have usable data
    df = df[df["participant_id"].isin(list(sub_dict_clean.keys()))]

    if len(drop_cols) > 0:
        df = df.drop(columns=[i for i in drop_cols if i in df.columns])

    good_grids = []
    for embedding_type in embedding_types:
        for grid_param in modality_grids[modality]:
            if not any(n in grid_param for n in rsns):
                print(f"{rsns} not found in recipe. Skipping...")
                continue
            grid_finds = []
            for ID in df["participant_id"]:
                if ID not in sub_dict_clean.keys():
                    print(f"ID: {ID} not found...")
                    continue

                if str(sessions[0]) not in sub_dict_clean[ID].keys():
                    print(f"Session: {sessions[0]} not found for ID {ID}...")
                    continue

                if modality not in sub_dict_clean[ID][str(sessions[0])].keys():
                    print(f"Modality: {modality} not found for ID {ID}, "
                          f"ses-{sessions[0]}...")
                    continue

                if embedding_type not in \
                    sub_dict_clean[ID][str(sessions[0])][modality].keys():
                    print(
                        f"Modality: {modality} not found for ID {ID}, "
                        f"ses-{sessions[0]}, {embedding_type}..."
                    )
                    continue

                if grid_param in \
                    list(sub_dict_clean[ID][str(sessions[0])][modality][
                             embedding_type].keys()):
                    grid_finds.append(grid_param)
            if len(grid_finds) < grid_thr*len(df["participant_id"]):
                print(
                    f"Less than {100*grid_thr}% of {grid_param} found. "
                    f"Removing from grid...")
                continue
            else:
                good_grids.append(grid_param)

    modality_grids[modality] = good_grids

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
            df = ml_dfs[modality][embedding_type][recipe]
            df.reset_index(inplace=True)
            if 'index' in df.columns:
                df = df.drop(columns=['index'])
            out_dict[str(recipe)] = df.to_json()
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

    args["base_dir"] = base_dir
    args["feature_spaces"] = feature_spaces
    args["modality_grids"] = modality_grids
    args["target_vars"] = target_vars
    args["embedding_type"] = embedding_type
    args["modality"] = modality
    args["n_boots"] = n_boots
    args["nuisance_cols"] = nuisance_cols
    args["dummy_run"] = dummy_run
    args["search_method"] = search_method
    args["stack"] = stack
    args["stack_prefix_list"] = stack_prefix_list

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
