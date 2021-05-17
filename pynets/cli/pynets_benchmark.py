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
    from pynets.stats.utils import make_subject_dict, cleanNullTerms, \
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

    # Parse inputs
    #base_dir = '/scratch/04171/dpisner/HNU/HNU_outs/triple'
    base_dir = '/scratch/04171/dpisner/HNU/HNU_outs/outputs_language'
    thr_type = "MST"
    icc = True
    disc = True
    int_consist = False
    modality = 'dwi'

    embedding_types = ['eigenvector', 'betweenness']
    #rsns = ['language']
    rsns = ['ventral']
    template = 'CN200'
    # template = 'MNI152_T1'
    mets = ["global_efficiency",
            "average_shortest_path_length",
            "degree_assortativity_coefficient",
            "average_betweenness_centrality",
            "average_eigenvector_centrality",
            "smallworldness",
            "modularity"]

    metaparams_func = ["rsn", "res", "model", 'hpass', 'extract',
                       'smooth']
    metaparams_dwi = ["rsn", "res", "model", 'directget', 'minlength',
                      'tol']

    sessions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    ####

    print(f"{Fore.LIGHTBLUE_EX}\nBenchmarking API\n")

    print(Style.RESET_ALL)

    print(f"{Fore.LIGHTGREEN_EX}Gathering sampled data...")

    print(Style.RESET_ALL)

    for embedding_type in embedding_types:
        subject_dict_file_path = (
            f"{base_dir}/pynets_subject_dict_{modality}_"
            f"{embedding_type}_{template}_{rsns}.pkl"
        )
        subject_mod_grids_file_path = (
            f"{base_dir}/pynets_modality_grids_{modality}_"
            f"{embedding_type}_{template}_{rsns}.pkl"
        )
        missingness_summary = (
            f"{base_dir}/pynets_missingness_summary_{modality}_"
            f"{embedding_type}_{template}_{rsns}.csv"
        )
        icc_tmps_dir = f"{base_dir}/icc_tmps/{rsns}_{modality}_" \
                       f"{embedding_type}"
        os.makedirs(icc_tmps_dir, exist_ok=True)
        if not os.path.isfile(subject_dict_file_path):
            subject_dict, modality_grids, missingness_frames = \
                make_subject_dict(
                    [modality], base_dir, thr_type, mets, [embedding_type],
                    template, sessions, rsns
                )
            sub_dict_clean = cleanNullTerms(subject_dict)
            missingness_frames = [i for i in missingness_frames if
                                  isinstance(i, pd.DataFrame)]
            if len(missingness_frames) != 0:
                if len(missingness_frames) > 0:
                    if len(missingness_frames) > 1:
                        final_missingness_summary = pd.concat(
                            missingness_frames)
                        final_missingness_summary.to_csv(missingness_summary,
                                                         index=False)
                        final_missingness_summary.id = \
                            final_missingness_summary.id.astype(
                                'str').str.split('_', expand=True)[0]
                    elif len(missingness_frames) == 1:
                        final_missingness_summary = missingness_frames[0]
                        final_missingness_summary.to_csv(missingness_summary,
                                                         index=False)
                        final_missingness_summary.id = \
                            final_missingness_summary.id.astype(
                                'str').str.split('_', expand=True)[0]
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
                final_missingness_summary.id = \
                    final_missingness_summary.id.astype('str').str.split(
                        '_', expand=True)[0]
            else:
                final_missingness_summary = pd.Series()
        ids = sub_dict_clean.keys()

        # print(f"MODALITY: {modality}")
        metaparams = eval(f"metaparams_{modality}")
        metaparam_dict = {}

        # print(f"EMBEDDING TYPE: {embedding_type}")
        # if os.path.isfile(f"{base_dir}/grid_clean_{modality}_{alg}.csv"):
        #     continue

        if embedding_type == 'topology':
            ensembles, df_top = get_ensembles_top(modality, thr_type,
                                                  f"{base_dir}/pynets")
        else:
            ensembles = get_ensembles_embedding(modality, embedding_type,
                                                base_dir)
        grid = build_grid(
            modality, metaparam_dict, sorted(list(set(metaparams))),
            ensembles)[1]

        grid = [i for i in grid if any(n in i for n in rsns)]

        good_grids = []
        for grid_param in grid:
            grid_finds = []
            for ID in ids:
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
            if len(grid_finds) < 0.75 * len(ids):
                print(
                    f"Less than 75% of {grid_param} found. Removing from "
                    f"grid...")
                continue
            else:
                good_grids.append(grid_param)

        modality_grids[modality] = good_grids

        cache_dir = tempfile.mkdtemp()

        with Parallel(
            n_jobs=-1, require="sharedmem", backend='threading',
            verbose=10, max_nbytes='200000M',
            temp_folder=cache_dir
        ) as parallel:
            outs = parallel(
                delayed(benchmark_reproducibility)(
                    base_dir, comb, modality, embedding_type, sub_dict_clean,
                    disc, final_missingness_summary, icc_tmps_dir, icc,
                    mets, ids, template
                )
                for comb in grid
            )
        # outs = []
        # for comb in grid:
        #     outs.append(benchmark_reproducibility(base_dir, comb, modality,
        #     embedding_type, sub_dict_clean,
        #             disc, final_missingness_summary, icc_tmps_dir, icc,
        #             mets, ids))

        df_summary = pd.concat([i for i in outs if i is not None and not
                                i.empty], axis=0)
        df_summary = df_summary.dropna(axis=0, how='all')
        print(f"Saving to {base_dir}/grid_clean_{modality}_{embedding_type}_"
              f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv...")
        df_summary.to_csv(f"{base_dir}"
                          f"/grid_clean_{modality}_{embedding_type}_"
                          f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
                          f".csv", index=False)

        # int_consist
        if int_consist is True and embedding_type == 'topology':
            try:
                import pingouin as pg
            except ImportError:
                print(
                    "Cannot evaluate test-retest int_consist. pingouin"
                    " must be installed!")

            df_summary_cronbach = pd.DataFrame(
                columns=['modality', 'embedding', 'cronbach'])
            df_summary_cronbach.at[0, "modality"] = modality
            df_summary_cronbach.at[0, "embedding"] = embedding_type

            for met in mets:
                cronbach_ses_list = []
                for ses in range(1, 10):
                    id_dict = {}
                    for ID in ids:
                        id_dict[ID] = {}
                        for comb in grid:
                            if modality == 'func':
                                try:
                                    extract, hpass, model, res, atlas, \
                                        smooth = comb
                                except BaseException:
                                    print(f"Missing {comb}...")
                                    extract, hpass, model, res, atlas = comb
                                    smooth = '0'
                                comb_tuple = (
                                    atlas, extract, hpass, model, res,
                                    smooth)
                            else:
                                directget, minlength, model, res, atlas, \
                                    tol = comb
                                comb_tuple = (
                                    atlas, directget, minlength, model,
                                    res, tol)
                            if comb_tuple in sub_dict_clean[ID][str(ses)][
                                    modality][embedding_type].keys():
                                if isinstance(sub_dict_clean[ID][str(ses)][
                                    modality][embedding_type
                                              ][comb_tuple], np.ndarray):
                                    id_dict[ID][comb] = sub_dict_clean[ID][
                                        str(ses)][modality][embedding_type
                                                            ][comb_tuple][
                                        mets.index(met)][0]
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
                        c_alpha = pg.cronbach_alpha(data=df_wide.dropna(
                            axis=1, how='all'), nan_policy='listwise')
                        cronbach_ses_list.append(c_alpha[0])
                    except BaseException:
                        print('FAILED...')
                        print(df_wide)
                        del df_wide
                    del df_wide
                df_summary_cronbach.at[0, f"average_cronbach_{met}"] = \
                    np.nanmean(cronbach_ses_list)
            print(f"Saving to {base_dir}/grid_clean_{modality}_"
                  f"{embedding_type}_cronbach_"
                  f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.csv...")
            df_summary_cronbach.to_csv(
                f"{base_dir}/grid_clean_{modality}_"
                f"{embedding_type}_cronbach"
                f"{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
                f".csv", index=False)

    return


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"
    main()
