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
        choices=["ASE", "OMNI", "topology"],
        help="Specify the embedding method of interest.\n",
    )
    parser.add_argument(
        "-tv",
        metavar="Target Variable",
        default="dep_1",
        nargs="+",
        choices=["rumination_persist_phenotype",
                 "depression_persist_phenotype", "dep_2", 'rum_2', 'rum_1',
                 'dep_1'],
        help="Specify the target outcome variables of interest, "
             "separated by space.\n",
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
        default="MNI152_T1",
        nargs=1,
        choices=[
            "colin27",
            "MNI152_T1",
            "CN200",
        ],
        help="Include this flag to specify a template, other than MNI152_T1 "
             "that was used to sample the connectome ensemble.\n",
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

    # pre_args = get_parser().parse_args()
    #
    # args["base_dir"] = pre_args.basedir
    # base_dir = args["base_dir"]
    base_dir = "/working/tuning_set/outputs_final"

    # args["target_vars"] = pre_args.tv
    # target_vars = args["target_vars"]
    target_vars = ["rumination_persist_phenotype",
                   "depression_persist_phenotype",
                   "dep_2", 'rum_2', 'rum_1', 'dep_1']

    # args["embedding_type"] = pre_args.et[0]
    # embedding_type = args["embedding_type"]
    embedding_type = 'ASE'

    # args["modality"] = pre_args.modality[0]
    # modality = args["modality"]
    modality = "func"

    # args["thr_type"] = pre_args.thrtype[0]
    # thr_type = pre_args.thrtype
    thr_type = "MST"

    # args["template"] = pre_args.thrtype[0]
    # template = pre_args.temp
    template = "MNI152_T1"

    data_file = "/working/tuning_set/outputs_final/df_rum_persist_all.csv"

    rsns = ["triple", "kmeans", "language"]

    # args["sessions"] = pre_args.thrtype[0]
    # sessions = pre_args.session_label
    sessions = ["1"]

    # Hard-Coded #
    mets = [
        "global_efficiency",
        "average_shortest_path_length",
        "average_degree_centrality",
        "average_eigenvector_centrality",
        "average_betweenness_centrality",
        "modularity",
        "degree_assortativity_coefficient"
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

    # Load in data
    df = pd.read_csv(
        data_file,
        index_col=False
    )

    # Subset only those participants which have usable data
    for ID in df["participant_id"]:
        if len(ID) == 1:
            df.loc[df.participant_id == ID, "participant_id"] = "s00" + str(ID)
        if len(ID) == 2:
            df.loc[df.participant_id == ID, "participant_id"] = "s0" + str(ID)

    df = df[df["participant_id"].isin(list(sub_dict_clean.keys()))]
    df['sex'] = df['sex'].map({1: 0, 2: 1})
    df = df[
        ["participant_id", "age", "sex", "num_visits", "DAY_LAG",
         'dataset'] + target_vars
    ]

    good_grids = []
    for grid_param in modality_grids[modality]:
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
        if len(grid_finds) < 0.75*len(df["participant_id"]):
            print(
                f"Less than 75% of {grid_param} found. Removing from grid...")
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
