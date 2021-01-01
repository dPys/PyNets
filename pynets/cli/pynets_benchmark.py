#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2016
@authors: Derek Pisner
"""
from pynets.stats.benchmarking import *


def main():
    import sys
    import os
    from datetime import datetime
    from joblib import Parallel, delayed
    import tempfile
    import dill
    from pynets.stats.prediction import make_subject_dict, cleanNullTerms, \
        get_ensembles_top, get_ensembles_embedding, \
        build_grid
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

    #### Parse inputs
    base_dir = '/scratch/04171/dpisner/HNU/HNU_outs/triple'
    # base_dir = '/scratch/04171/dpisner/HNU/HNU_outs/outputs_language'
    thr_type = "MST"
    icc = True
    disc = True
    int_consist = True
    modality = 'dwi'

    embedding_types = ['ASE']
    rsns = ['triple', 'kmeans']
    template = 'CN200'
    # template = 'MNI152_T1'
    mets = ["global_efficiency",
            "average_shortest_path_length",
            "degree_assortativity_coefficient",
            "average_betweenness_centrality",
            "average_eigenvector_centrality",
            "smallworldness",
            "modularity"]

    hyperparams_func = ["rsn", "res", "model", 'hpass', 'extract',
                        'smooth']
    hyperparams_dwi = ["rsn", "res", "model", 'directget', 'minlength',
                       'tol']

    sessions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    ####

    print(f"{Fore.LIGHTBLUE_EX}\nBenchmarking API\n")

    print(Style.RESET_ALL)

    print(f"{Fore.LIGHTGREEN_EX}Gathering sampled data...")

    print(Style.RESET_ALL)

    subject_dict_file_path = (
        f"{base_dir}/pynets_subject_dict_{modality}_{'_'.join(embedding_types)}_{template}.pkl"
    )
    subject_mod_grids_file_path = (
        f"{base_dir}/pynets_modality_grids_{modality}_{'_'.join(embedding_types)}_{template}.pkl"
    )
    missingness_summary = (
        f"{base_dir}/pynets_missingness_summary_{modality}_{'_'.join(embedding_types)}_{template}.csv"
    )
    icc_tmps_dir = f"{base_dir}/icc_tmps/{modality}_{'_'.join(embedding_types)}"
    os.makedirs(icc_tmps_dir, exist_ok=True)
    if not os.path.isfile(subject_dict_file_path):
        subject_dict, modality_grids, missingness_frames = make_subject_dict(
            [modality], base_dir, thr_type, mets, embedding_types, template,
            sessions, rsns
        )
        sub_dict_clean = cleanNullTerms(subject_dict)
        missingness_frames = [i for i in missingness_frames if
                              isinstance(i, pd.DataFrame)]
        if len(missingness_frames) != 0:
            if len(missingness_frames) > 0:
                if len(missingness_frames) > 1:
                    final_missingness_summary = pd.concat(missingness_frames)
                    final_missingness_summary.to_csv(missingness_summary,
                                                     index=False)
                    final_missingness_summary.id = final_missingness_summary.id.str.split('_', expand=True)[0]
                elif len(missingness_frames) == 1:
                    final_missingness_summary = missingness_frames[0]
                    final_missingness_summary.to_csv(missingness_summary, index=False)
                    final_missingness_summary.id = final_missingness_summary.id.str.split('_', expand=True)[0]
                else:
                    final_missingness_summary = pd.Series()
            else:
                final_missingness_summary = pd.Series()
        else:
            final_missingness_summary = pd.Series()
        with open(subject_dict_file_path, "wb") as f:
            dill.dump(sub_dict_clean, f)
        f.close()
        with open(subject_mod_grids_file_path, "wb") as f:
            dill.dump(modality_grids, f)
        f.close()
    else:
        with open(subject_dict_file_path, 'rb') as f:
            sub_dict_clean = dill.load(f)
        f.close()
        with open(subject_mod_grids_file_path, "rb") as f:
            modality_grids = dill.load(f)
        f.close()
        if os.path.isfile(missingness_summary):
            final_missingness_summary = pd.read_csv(missingness_summary)
            final_missingness_summary.id = final_missingness_summary.id.str.split('_', expand=True)[0]
        else:
            final_missingness_summary = pd.Series()
    ids = sub_dict_clean.keys()

    print(f"MODALITY: {modality}")
    hyperparams = eval(f"hyperparams_{modality}")
    hyperparam_dict = {}

    for alg in embedding_types:
        print(f"EMBEDDING TYPE: {alg}")
        # if os.path.isfile(f"{base_dir}/grid_clean_{modality}_{alg}.csv"):
        #     continue

        if alg == 'topology':
            ensembles, df_top = get_ensembles_top(modality, thr_type,
                                                  f"{base_dir}/pynets")
        else:
            ensembles = get_ensembles_embedding(modality, alg,
                                                base_dir)
        grid = build_grid(
            modality, hyperparam_dict, sorted(list(set(hyperparams))),
            ensembles)[1]

        grid = [i for i in grid if '200' not in i and '400' not in i and '600' not in i and '800' not in i]
        # In the case that we are using all of the 3 RSN connectomes
        # (pDMN, coSN, and fECN) in the feature-space,
        # rather than varying them as hyperparameters (i.e. we assume
        # they each add distinct variance
        # from one another) Create an abridged grid, where

        if modality == "func":
            modality_grids[modality] = grid
        else:
            modality_grids[modality] = grid

        cache_dir = tempfile.mkdtemp()

        with Parallel(
            n_jobs=-1, require="sharedmem", backend='threading',
            verbose=10, max_nbytes='20000M',
            temp_folder=cache_dir
        ) as parallel:
            outs = parallel(
                delayed(benchmark_reproducibility)(
                    base_dir, comb, modality, alg, sub_dict_clean,
                    disc, final_missingness_summary, icc_tmps_dir, icc,
                    mets, ids
                )
                for comb in grid
            )

        df_summary = pd.concat([i for i in outs if i is not None and not i.empty], axis=0)
        df_summary = df_summary.dropna(axis=0, how='all')
        print(f"Saving to {base_dir}/grid_clean_{modality}_{alg}_"
              f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv...")
        df_summary.to_csv(f"{base_dir}"
                          f"/grid_clean_{modality}_{alg}_"
                          f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv", index=False)

        # int_consist
        if int_consist is True and alg == 'topology':
            try:
                import pingouin as pg
            except ImportError:
                print(
                    "Cannot evaluate test-retest int_consist. pingouin"
                    " must be installed!")

            df_summary_cronbach = pd.DataFrame(
                columns=['modality', 'embedding', 'cronbach'])
            df_summary_cronbach.at[0, "modality"] = modality
            df_summary_cronbach.at[0, "embedding"] = alg

            for met in mets:
                cronbach_ses_list = []
                for ses in range(1, 10):
                    id_dict = {}
                    for ID in ids:
                        id_dict[ID] = {}
                        for comb in grid:
                            if modality == 'func':
                                try:
                                    extract, hpass, model, res, atlas, smooth = comb
                                except BaseException:
                                    print(f"Missing {comb}...")
                                    extract, hpass, model, res, atlas = comb
                                    smooth = '0'
                                comb_tuple = (
                                atlas, extract, hpass, model, res,
                                smooth)
                            else:
                                directget, minlength, model, res, atlas, tol = comb
                                comb_tuple = (
                                atlas, directget, minlength, model,
                                res, tol)
                            if comb_tuple in sub_dict_clean[ID][str(ses)][modality][alg].keys():
                                if isinstance(sub_dict_clean[ID][str(ses)][modality][alg][comb_tuple], np.ndarray):
                                    id_dict[ID][comb] = sub_dict_clean[ID][str(ses)][modality][alg][comb_tuple][mets.index(met)][0]
                                else:
                                    continue
                            else:
                                continue
                    df_wide = pd.DataFrame(id_dict)
                    if df_wide.empty is True:
                        continue
                    else:
                        df_wide = df_wide.add_prefix(f"{met}_comb_")
                        df_wide.replace(0, np.nan, inplace=True)
                        print(df_wide)
                    try:
                        c_alpha = pg.cronbach_alpha(data=df_wide.dropna(axis=1, how='all'), nan_policy='listwise')
                        cronbach_ses_list.append(c_alpha[0])
                    except BaseException:
                        print('FAILED...')
                        print(df_wide)
                        del df_wide
                    del df_wide
                df_summary_cronbach.at[0, f"average_cronbach_{met}"] = np.nanmean(cronbach_ses_list)
            print(f"Saving to {base_dir}/grid_clean_{modality}_{alg}_cronbach_"
                  f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv...")
            df_summary_cronbach.to_csv(f"{base_dir}/grid_clean_{modality}_{alg}_cronbach{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv", index=False)

    return


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"
    main()
