#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2016
@author: Derek Pisner (dPys)
"""
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import sys
if sys.platform.startswith('win') is False:
    import indexed_gzip


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
        choices=["dwi", "func"],
        help="Specify data modality from which to collect data. Options are"
             " `dwi` and `func`.",
    )
    parser.add_argument(
        "-dc",
        metavar="Column strings to exclude",
        default=None,
        nargs="+",
        help="Space-delimited list of strings.\n",
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


def load_pd_dfs(file_):
    import gc
    import os
    import os.path as op
    import pandas as pd
    import numpy as np
    from colorama import Fore, Style

    pd.set_option("display.float_format", lambda x: f"{x:.8f}")

    if file_:
        if op.isfile(file_) and not file_.endswith("_clean.csv"):
            try:
                df = pd.read_csv(file_, chunksize=100000, encoding="utf-8",
                                 engine='python').read()
            except:
                print(f"Load failed for {file_}. Trying again with c engine.")
                try:
                    df = pd.read_csv(file_, chunksize=100000, encoding="utf-8",
                                     engine='c').read()
                except:
                    print(f"Cannot load {file_}")
                    df = pd.DataFrame()
                    return df
            if "Unnamed: 0" in df.columns:
                df.drop(df.filter(regex="Unnamed: 0"), axis=1, inplace=True)
            id = op.basename(file_).split("_topology")[0]
            if 'sub-sub-' in id:
                id = id.replace('topology_auc_sub-', '')
            else:
                id = id.replace('topology_auc_', '')
            if 'ses-ses-' in id:
                id = id.replace('ses-ses-', 'ses-')

            id = ('_').join(id.split('_')[0:2])

            if '.csv' in id:
                id = id.replace('.csv', '')

            #print(id)

            df["id"] = id
            df["id"] = df["id"].astype('str')
            df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
            # bad_cols1 = df.columns[df.columns.str.endswith("_x")]
            # if len(bad_cols1) > 0:
            #     for col in bad_cols1:
            #         if np.isnan(df[col][0]) is False:
            #             df.rename(columns=dict(zip(bad_cols1, [bad_col.split(
            #                 "_x")[0] for bad_col in bad_cols1])), inplace=True)
            #         else:
            #             df.drop(columns=[col], inplace=True)
            #     del col
            # bad_cols2 = df.columns[df.columns.str.endswith("_y")]
            # if len(bad_cols2) > 0:
            #     for col in bad_cols2:
            #         if np.isnan(df[col][0]) is False:
            #             df.rename(columns=dict(zip(bad_cols2, [bad_col.split(
            #                 "_y")[0] for bad_col in bad_cols2])),
            #                       inplace=True)
            #         else:
            #             df.drop(columns=[col], inplace=True)
            #     del col

            df = df.loc[:, ~df.columns.str.contains(r".?\d{1}$", regex=True)]

            # Find empty duplicate columns
            df_dups = df.loc[:, df.columns.duplicated()]
            if df_dups.empty is False:
                empty_cols = [col for col in df.columns if
                              df_dups[col].isnull().all()]
                # Drop these columns from the dataframe
                print(f"{Fore.LIGHTYELLOW_EX}"
                      f"ropping duplicated empty columns: "
                      f"{empty_cols}{Style.RESET_ALL}")
                df.drop(empty_cols,
                        axis=1,
                        inplace=True)
            if "Unnamed: 0" in df.columns:
                df.drop(df.filter(regex="Unnamed: 0"), axis=1, inplace=True)
            # summarize_missingness(df)
            if os.path.isfile(f"{file_.split('.csv')[0]}{'_clean.csv'}"):
                os.remove(f"{file_.split('.csv')[0]}{'_clean.csv'}")
            df.to_csv(f"{file_.split('.csv')[0]}{'_clean.csv'}", index=False)
            del id

        else:
            print(f"{Fore.RED}Cleaned {file_} missing...{Style.RESET_ALL}")
            df = pd.DataFrame()
    else:
        print(f"{Fore.RED}{file_} missing...{Style.RESET_ALL}")
        df = pd.DataFrame()
    gc.collect()

    return df


def df_concat(dfs, working_path, modality, drop_cols, args):
    import os
    import pandas as pd
    import numpy as np
    from joblib import Parallel, delayed
    import tempfile
    from pynets.cli.pynets_collect import recover_missing

    # from colorama import Fore, Style

    pd.set_option("display.float_format", lambda x: f"{x:.8f}")

    def harmonize_dtypes(df):
        for i in [j for j in df.columns if j != 'id']:
            df[i] = df[i].astype("float32")
        return df

    def fill_columns(df, all_cols):
        import numpy as np
        diverging_cols = list(np.setdiff1d(all_cols, df.columns.tolist()))
        if len(diverging_cols) > 0:
            df = df.reindex(
                columns=[*df.columns.tolist(), *diverging_cols],
                fill_value=np.nan)
        df = df.reindex(sorted(df.columns), axis=1)
        return df

    def mergedicts(dict1, dict2):
        for k in set(dict1.keys()).union(dict2.keys()):
            if k in dict1 and k in dict2:
                if isinstance(dict1[k], dict) and \
                    isinstance(dict2[k], dict):
                    yield (k, dict(mergedicts(dict1[k],
                                              dict2[k])))
                else:
                    yield (k, dict2[k])
            elif k in dict1:
                yield (k, dict1[k])
            else:
                yield (k, dict2[k])

    print('Harmonizing columnn types across dataframes...')
    dfs = [harmonize_dtypes(df) for df in dfs if df is not None and
           df.empty is False]
    all_cols = []
    for df in dfs:
        all_cols.extend(df.columns.tolist())
    all_cols = list(set(all_cols))

    print('Harmonizing columnn names across dataframes...')
    out_dfs = [fill_columns(df, all_cols) for df in dfs]

    # for df in out_dfs:
    #     print(len(df.columns))

    print('Joining...')
    frame = pd.concat(out_dfs, axis=0, join="outer", sort=False,
                      ignore_index=False)
    frame = frame.loc[:, ~frame.columns.str.contains(r"thr_auc$", regex=True)]
    frame.dropna(axis='columns', how='all', inplace=True)
    for drop_col in drop_cols:
        frame = frame.loc[:, ~frame.columns.str.contains(f"{drop_col}",
                                                         regex=True)]
    try:
        frame = frame.set_index('id')
    except:
        pass

    drop = [i for i in frame.columns if 'participation' in i]
    frame = frame.drop(columns=drop)
    # drop = [i for i in frame.columns if 'degree_centrality' in i]
    # frame = frame.drop(columns=drop)
    # drop = [i for i in frame.columns if 'betweenness_centrality' in i]
    # frame = frame.drop(columns=drop)

    if os.path.isfile(f"{working_path}/all_subs_neat_{modality}.csv"):
        frame_fill = pd.read_csv(f"{working_path}/"
                                 f"all_subs_neat_{modality}.csv")
        if len(frame_fill.columns) == len(frame.columns):
            print("Found existing dataframe. Using this to fill in "
                  "missing values...")
            try:
                frame_fill = frame_fill.set_index('id')
                frame[frame.isnull()] = frame_fill
            except:
                pass

    # frame = frame.loc[:, (frame == 0).mean() < .5]
    # frame = frame.loc[:, frame.isnull().mean() <= 0.1]
    # frame.dropna(thresh=0.50*len(frame.columns), inplace=True)
    missingness_dict = summarize_missingness(frame)[0]
    bad_cols = []
    for col in missingness_dict.keys():
        if missingness_dict[col] > 0.10:
            bad_cols.append(col)
    del col

    bad_cols_dict = {}
    for col in bad_cols:
        bad_cols_dict[col] = frame[col].index[frame[col].apply(np.isnan)]

    rerun_dict = {}
    print('Fill in any missing cells if auc files are detected, '
          'otherwise create an inventory of missingness...')
    par_dict = rerun_dict.copy()
    cache_dir = tempfile.mkdtemp()
    with Parallel(n_jobs=-1, backend='loky', verbose=10,
                  temp_folder=cache_dir) as parallel:
        outs = parallel(delayed(recover_missing)(bad_col, bad_cols_dict,
                                                 par_dict, modality,
                        working_path, drop_cols, frame) for
                        bad_col in bad_cols_dict.keys())

    if os.path.isfile(f"{working_path}/all_subs_neat_{modality}.csv"):
        os.remove(f"{working_path}/all_subs_neat_{modality}.csv")
    frame.to_csv(f"{working_path}/all_subs_neat_{modality}.csv", index=True)

    rerun_dicts = []
    reruns = []
    for rd, rerun in outs:
        rerun_dicts.append(rd)
        reruns.append(rerun)

    for rd in rerun_dicts:
        rerun_dict = dict(mergedicts(rerun_dict, rd))

    # # Re-run collection...
    # if sum(reruns) > 1:
    #     build_collect_workflow(args, outs)

    # if len(bad_cols) > 0:
        # print(f"{Fore.LIGHTYELLOW_EX}Dropping columns with excessive "
        #       f"missingness: {bad_cols}{Style.RESET_ALL}")
        # frame = frame.drop(columns=bad_cols)

    # frame['missing'] = frame.apply(lambda x: x.count(), axis=1)
    # frame = frame.loc[frame['missing'] > np.mean(frame['missing'])]
    # frame = frame.sort_values(by=['missing'], ascending=False)

    return frame, rerun_dict


def recover_missing(bad_col, bad_cols_dict, rerun_dict, modality,
                    working_path, drop_cols, frame):
    import glob
    import os
    atlas = bad_col.split('_')[0] + '_' + bad_col.split('_')[1]
    rerun = False

    for lab in bad_cols_dict[bad_col]:
        sub = lab.split('_')[0]
        ses = lab.split('_')[1]
        if sub not in rerun_dict.keys():
            rerun_dict[sub] = {}
        if ses not in rerun_dict[sub].keys():
            rerun_dict[sub][ses] = {}
        if modality not in rerun_dict[sub][ses].keys():
            rerun_dict[sub][ses][modality] = {}
        if atlas not in rerun_dict[sub][ses][modality].keys():
            rerun_dict[sub][ses][modality][atlas] = []
        search_str = bad_col.replace(f"{atlas}_", '').split('_thrtype')[0]
        if not os.path.isdir(f"{working_path}/{sub}/{ses}/"
                             f"{modality}/{atlas}/topology/auc"):
            if not os.path.isdir(
                f"{working_path}/{sub}/{ses}/{modality}/{atlas}/topology"):
                print(f"Missing graph analysis for {sub}, {ses} for "
                      f"{atlas}...")
            else:
                from pynets.stats.netstats import collect_pandas_df_make
                collect_pandas_df_make(glob.glob(f"{working_path}/{sub}/{ses}/"
                                                 f"{modality}/{atlas}/"
                                                 f"topology/*_neat.csv"),
                                       f"{sub}_{ses}", None, False)
                rerun = True
        outs = [i for i in glob.glob(f"{working_path}/{sub}/{ses}/"
                                     f"{modality}/{atlas}/topology/auc/*")
                if search_str in i]

        if len(outs) == 1:
            # Fill in gaps (for things that get dropped during earlier
            # stages)
            try:
                df_tmp = pd.read_csv(
                    outs[0], chunksize=100000, compression="gzip",
                    encoding="utf-8", engine='python').read()
            except:
                try:
                    df_tmp = pd.read_csv(
                        outs[0], chunksize=100000, compression="gzip",
                        encoding="utf-8", engine='c').read()
                except:
                    print(f"Cannot load {outs[0]}")
                    continue
            if not df_tmp.empty:
                for drop in drop_cols:
                    if drop in bad_col:
                        print(f"Removing column: {drop}")
                        frame = frame.drop(columns=bad_col)

                if bad_col not in frame.columns:
                    continue
                    from pynets.stats.netstats import \
                        collect_pandas_df_make
                    collect_pandas_df_make(
                        glob.glob(f"{working_path}/{sub}/{ses}/"
                                  f"{modality}/{atlas}/topology/*_neat.csv"),
                        f"{sub}_{ses}", None, False)
                try:
                    frame.loc[lab, bad_col] = df_tmp.filter(
                        regex=bad_col.split('auc_')[1:][0]
                    ).values.tolist()[0][0]
                    print(f"Recovered missing data from {sub}, {ses} for "
                          f"{bad_col}...")
                except:
                    from pynets.stats.netstats import \
                        collect_pandas_df_make
                    collect_pandas_df_make(
                        glob.glob(f"{working_path}/{sub}/{ses}/"
                                  f"{modality}/{atlas}/topology/*_neat.csv"),
                        f"{sub}_{ses}", None, False)
                    continue
                del df_tmp
            else:
                from pynets.stats.netstats import collect_pandas_df_make
                collect_pandas_df_make(glob.glob(f"{working_path}/{sub}/{ses}/"
                                                 f"{modality}/{atlas}/topology/*_neat.csv"),
                                       f"{sub}_{ses}", None, False)
                rerun_dict[sub][ses][modality][atlas].append(bad_col)
                continue
        elif len(outs) > 1:
            for out in outs:
                try:
                    df_tmp = pd.read_csv(
                        out, chunksize=100000, compression="gzip",
                        encoding="utf-8", engine='python').read()
                except:
                    try:
                        df_tmp = pd.read_csv(
                            out, chunksize=100000, compression="gzip",
                            encoding="utf-8", engine='c').read()
                    except:
                        print(f"Cannot load {out}")
                        continue
                if not df_tmp.empty:
                    print(f"Recovered missing data from {sub}, {ses} for "
                          f"{bad_col}...")

                    for drop in drop_cols:
                        if drop in bad_col:
                            print(f"Removing column: {drop}")
                            frame = frame.drop(columns=bad_col)
                    try:
                        frame.loc[lab, bad_col] = df_tmp.filter(
                            regex=bad_col.split('auc_')[1:][0]
                        ).values.tolist()[0][0]
                    except:
                        from pynets.stats.netstats import \
                            collect_pandas_df_make
                        collect_pandas_df_make(
                            glob.glob(f"{working_path}/{sub}/{ses}/"
                                      f"{modality}/{atlas}/topology/*_neat.csv"),
                            f"{sub}_{ses}", None, False)
                        continue
                    del df_tmp
        else:
            # Add to missingness inventory if not found
            rerun_dict[sub][ses][modality][atlas].append(bad_col)
            from pynets.stats.netstats import \
                collect_pandas_df_make
            collect_pandas_df_make(
                glob.glob(f"{working_path}/{sub}/{ses}/"
                          f"{modality}/{atlas}/topology/*_neat.csv"),
                f"{sub}_{ses}", None, False)
    return rerun_dict, rerun


def summarize_missingness(df):
    import numpy as np
    from colorama import Fore, Style
    missingness_dict = dict(df.apply(lambda x: x.isna().sum() /
                                               (x.count() + x.isna().sum()),
                                     axis=0))
    missingness_mean = np.mean(list(missingness_dict.values()))
    if missingness_mean > 0.50:
        print(f"{Fore.RED} {df} missing {100*missingness_mean}% "
              f"values!{Style.RESET_ALL}")

    return missingness_dict, missingness_mean


def load_pd_dfs_auc(atlas_name, prefix, auc_file, modality, drop_cols):
    from colorama import Fore, Style
    import pandas as pd
    import re
    import os

    pd.set_option("display.float_format", lambda x: f"{x:.8f}")

    try:
        df = pd.read_csv(
            auc_file, chunksize=100000, compression="gzip",
            encoding="utf-8", engine='c').read()
    except:
        try:
            df = pd.read_csv(
                auc_file, chunksize=100000, compression="gzip",
                encoding="utf-8", engine='python').read()
        except:
            df_pref = pd.DataFrame()
            return df_pref

    #print(f"{'Atlas: '}{atlas_name}")
    prefix = f"{atlas_name}{'_'}{prefix}{'_'}"
    df_pref = df.add_prefix(prefix)
    if modality == 'dwi':
        df_pref = df_pref.rename(
            columns=lambda x: re.sub(
                "nodetype-parc_samples-\d{1,5}0000streams_tracktype-local_",
                "", x))
    bad_cols = [i for i in df_pref.columns if any(ele in i for ele in
                                                  drop_cols)]
    #print(f"{Fore.YELLOW} Dropping {len(bad_cols)}: {bad_cols} containing
    # exclusionary strings...{Style.RESET_ALL}")
    df_pref.drop(columns=bad_cols, inplace=True)

    print(df_pref)
    # Find empty duplicate columns
    df_dups = df_pref.loc[:, df_pref.columns.duplicated()]
    if df_dups.empty is False:
        empty_cols = [col for col in df_pref.columns if
                      df_dups[col].isnull().all()]
        # Drop these columns from the dataframe
        df_pref.drop(empty_cols, axis=1, inplace=True)
    if "Unnamed: 0" in df_pref.columns:
        df_pref.drop(df_pref.filter(regex="Unnamed: 0"), axis=1, inplace=True)

    if df_pref.empty:
        print(f"{Fore.RED}Empty raw AUC: {df_pref} from {auc_file}..."
              f"{Style.RESET_ALL}")

    return df_pref


def build_subject_dict(sub, working_path, modality, drop_cols):
    import shutil
    import os
    import glob
    from pathlib import Path
    from colorama import Fore, Style
    from pynets.cli.pynets_collect import load_pd_dfs_auc

    def is_non_zero_file(fpath):
        return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

    subject_dict = {}
    print(sub)
    subject_dict[sub] = {}
    sessions = sorted(
        [i for i in os.listdir(f"{working_path}{'/'}{sub}") if i.startswith(
            "ses-")],
        key=lambda x: x.split("-")[1],
    )
    atlases = list(
        set(
            [
                os.path.basename(str(Path(i).parent.parent))
                for i in glob.glob(f"{working_path}/{sub}/*/{modality}/*/"
                                   f"topology/*", recursive=True)
            ]
        )
    )
    print(atlases)

    files_ = []
    for ses in sessions:
        print(ses)
        subject_dict[sub][ses] = {}
        for atlas in atlases:
            subject_dict[sub][ses][atlas] = []
            #atlas_name = "_".join(atlas.split("_")[1:])
            auc_csvs = glob.glob(
                f"{working_path}/{sub}/{ses}/{modality}/{atlas}/topology/auc/*"
            )
            for auc_file in auc_csvs:
                prefix = (
                    os.path.basename(auc_file)
                    .split(".csv")[0]
                    .split("model-")[1]
                    .split(modality)[0]
                )
                if os.path.isfile(auc_file) and is_non_zero_file(auc_file):
                    df_sub = load_pd_dfs_auc(atlas, prefix, auc_file,
                                             modality, drop_cols)
                    df_sub['id'] = f"{sub}_{ses}"
                    if df_sub.empty:
                        print(f"{Fore.RED}Empty auc file for {sub} {ses}..."
                              f"{Style.RESET_ALL}")
                    else:
                        subject_dict[sub][ses][atlas].append(df_sub)
                else:
                    print(f"{Fore.RED}Missing auc file for {sub} {ses}..."
                          f"{Style.RESET_ALL}")
        list_ = [subject_dict[sub][ses][i] for i in
                 subject_dict[sub][ses].keys()]
        list_ = [item for sublist in list_ for item in sublist]
        if len(list_) > 0:
            df_base = list_[0][[c for c in list_[
                0].columns if c.endswith("auc") or c == 'id']]
            try:
                df_base.set_index('id', inplace=True)
            except:
                pass
            for m in range(len(list_))[1:]:
                df_to_be_merged = list_[m][[c for c in list_[m].columns if
                                            c.endswith("auc") or c == 'id']]
                try:
                    df_to_be_merged.set_index('id', inplace=True)
                except:
                    pass
                df_base = df_base.merge(
                    df_to_be_merged,
                    left_index=True,
                    right_index=True,
                )
            if os.path.isdir(
                    f"{working_path}{'/'}{sub}{'/'}{ses}{'/'}{modality}"):
                out_path = (
                    f"{working_path}/{sub}/{ses}/{modality}/all_combinations"
                    f"_auc.csv"
                )
                if os.path.isfile(out_path):
                    os.remove(out_path)
                df_base.to_csv(out_path, index=False)
                out_path_new = f"{str(Path(working_path))}/{modality}_" \
                               f"group_topology_auc/topology_auc_{sub}_" \
                               f"{ses}.csv"
                files_.append(out_path_new)
                shutil.copyfile(out_path, out_path_new)

            del df_base
        else:
            print(f"{Fore.RED}Missing data for {sub} {ses}..."
                  f"{Style.RESET_ALL}")
        del list_

    return files_


def collect_all(working_path, modality, drop_cols):
    from pathlib import Path
    import shutil
    import os

    import_list = [
        "import warnings",
        'warnings.filterwarnings("ignore")',
        "import os",
        "import numpy as np",
        "import nibabel as nib",
        "import glob",
        "import pandas as pd",
        "import shutil",
        "from pathlib import Path",
        "from colorama import Fore, Style"
    ]

    shutil.rmtree(f"{str(Path(working_path))}/{modality}_group_topology_auc",
                  ignore_errors=True)

    os.makedirs(f"{str(Path(working_path))}/{modality}_group_topology_auc")

    wf = pe.Workflow(name="load_pd_dfs")

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["working_path", "modality",
                                      "drop_cols"]),
        name="inputnode"
    )
    inputnode.inputs.working_path = working_path
    inputnode.inputs.modality = modality
    inputnode.inputs.drop_cols = drop_cols

    build_subject_dict_node = pe.Node(
        niu.Function(
            input_names=["sub", "working_path", "modality", "drop_cols"],
            output_names=["files_"],
            function=build_subject_dict,
        ),
        name="build_subject_dict_node",
        imports=import_list,
    )
    build_subject_dict_node.iterables = (
        "sub",
        [i for i in os.listdir(working_path) if i.startswith("sub-")],
    )
    build_subject_dict_node.synchronize = True

    df_join_node = pe.JoinNode(
        niu.IdentityInterface(fields=["files_"]),
        name="df_join_node",
        joinfield=["files_"],
        joinsource=build_subject_dict_node,
    )

    load_pd_dfs_map = pe.MapNode(
        niu.Function(
            input_names=["file_"],
            outputs_names=["df"],
            function=load_pd_dfs),
        name="load_pd_dfs",
        imports=import_list,
        iterfield=["file_"],
        nested=True,
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["dfs"]),
        name="outputnode")

    wf.connect(
        [
            (inputnode, build_subject_dict_node,
             [("working_path", "working_path"), ('modality', 'modality'),
              ('drop_cols', 'drop_cols')]),
            (build_subject_dict_node, df_join_node, [("files_", "files_")]),
            (df_join_node, load_pd_dfs_map, [("files_", "file_")]),
            (load_pd_dfs_map, outputnode, [("df", "dfs")]),
        ]
    )

    return wf


def build_collect_workflow(args, retval):
    import os
    import glob
    import warnings
    warnings.filterwarnings("ignore")
    import ast
    import pkg_resources
    from pathlib import Path
    import yaml
    import uuid
    from time import strftime
    import shutil

    try:
        import pynets

        print(f"\n\nPyNets Version:\n{pynets.__version__}\n\n")
    except ImportError:
        print(
            "PyNets not installed! Ensure that you are using the correct"
            " python version."
        )

    # Set Arguments to global variables
    resources = args.pm
    if resources == "auto":
        from multiprocessing import cpu_count
        import psutil
        nthreads = cpu_count() - 1
        procmem = [int(nthreads),
                   int(list(psutil.virtual_memory())[4]/1000000000)]
    else:
        procmem = list(eval(str(resources)))
    plugin_type = args.plug
    if isinstance(plugin_type, list):
        plugin_type = plugin_type[0]
    verbose = args.v
    working_path = args.basedir
    work_dir = args.work
    modality = args.modality
    drop_cols = args.dc
    if isinstance(modality, list):
        modality = modality[0]

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)

    os.makedirs(
        f"{str(Path(working_path))}/{modality}_group_topology_auc",
        exist_ok=True)

    wf = collect_all(working_path, modality, drop_cols)

    with open(
        pkg_resources.resource_filename("pynets", "runconfig.yaml"), "r"
    ) as stream:
        try:
            hardcoded_params = yaml.load(stream)
            runtime_dict = {}
            execution_dict = {}
            for i in range(len(hardcoded_params["resource_dict"])):
                runtime_dict[
                    list(hardcoded_params["resource_dict"][i].keys())[0]
                ] = ast.literal_eval(
                    list(hardcoded_params["resource_dict"][i].values())[0][0]
                )
            for i in range(len(hardcoded_params["execution_dict"])):
                execution_dict[
                    list(hardcoded_params["execution_dict"][i].keys())[0]
                ] = list(hardcoded_params["execution_dict"][i].values())[0][0]
        except FileNotFoundError:
            print("Failed to parse runconfig.yaml")

    run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
    os.makedirs(f"{work_dir}/pynets_out_collection{run_uuid}", exist_ok=True)
    wf.base_dir = f"{work_dir}/pynets_out_collection{run_uuid}"

    if verbose is True:
        from nipype import config, logging

        cfg_v = dict(
            logging={
                "workflow_level": "DEBUG",
                "utils_level": "DEBUG",
                "interface_level": "DEBUG",
                "filemanip_level": "DEBUG",
                "log_directory": str(wf.base_dir),
                "log_to_file": True,
            },
            monitoring={
                "enabled": True,
                "sample_frequency": "0.1",
                "summary_append": True,
                "summary_file": str(wf.base_dir),
            },
        )
        logging.update_logging(config)
        config.update_config(cfg_v)
        config.enable_debug_mode()
        config.enable_resource_monitor()

        import logging

        callback_log_path = f"{wf.base_dir}{'/run_stats.log'}"
        logger = logging.getLogger("callback")
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(callback_log_path)
        logger.addHandler(handler)

    execution_dict["crashdump_dir"] = str(wf.base_dir)
    execution_dict["plugin"] = str(plugin_type)
    cfg = dict(execution=execution_dict)
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            wf.config[key][setting] = value
    try:
        wf.write_graph(graph2use="colored", format="png")
    except BaseException:
        pass
    if verbose is True:
        from nipype.utils.profiler import log_nodes_cb

        plugin_args = {
            "n_procs": int(procmem[0]),
            "memory_gb": int(procmem[1]),
            "status_callback": log_nodes_cb,
            "scheduler": "mem_thread",
        }
    else:
        plugin_args = {
            "n_procs": int(procmem[0]),
            "memory_gb": int(procmem[1]),
            "scheduler": "mem_thread",
        }
    print("%s%s%s" % ("\nRunning with ", str(plugin_args), "\n"))
    wf.run(plugin=plugin_type, plugin_args=plugin_args)
    if verbose is True:
        from nipype.utils.draw_gantt_chart import generate_gantt_chart

        print("Plotting resource profile from run...")
        generate_gantt_chart(callback_log_path, cores=int(procmem[0]))
        handler.close()
        logger.removeHandler(handler)
    return


def main():
    """Initializes collection of pynets outputs."""
    import gc
    import sys
    import glob
    from pynets.cli.pynets_collect import build_collect_workflow
    from types import SimpleNamespace
    from pathlib import Path

    try:
        from pynets.core.utils import do_dir_path
    except ImportError:
        print(
            "PyNets not installed! Ensure that you are referencing the correct"
            " site-packages and using Python3.5+"
        )

    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h"
              " flag.\n")
        sys.exit()

    # args = get_parser().parse_args()
    args_dict_all = {}
    args_dict_all['plug'] = 'MultiProc'
    args_dict_all['v'] = False
    args_dict_all['pm'] = '48,57'
    #args_dict_all['basedir'] = '/working/tuning_set/outputs_shaeffer/pynets'
    #args_dict_all['basedir'] = '/scratch/04171/dpisner/HNU/HNU_outs/triple/pynets'
    #args_dict_all['basedir'] = '/scratch/04171/dpisner/HNU/HNU_outs/visual/pynets'
    args_dict_all['basedir'] = '/scratch/04171/dpisner/tuning_set/outputs_shaeffer/pynets'
    args_dict_all['work'] = '/tmp/work/func'
    args_dict_all['modality'] = 'func'
    args_dict_all['dc'] = ['diversity_coefficient',
                           'participation_coefficient',
                           'average_local_efficiency',
                           'average_clustering',
                           'average_local_clustering_nodewise',
                           'average_local_efficiency_nodewise',
                           'degree_centrality',
                        #   "_minlength-0",
                           "_minlength-20", "_minlength-30", "variance",
                           "res-1000"]
    args = SimpleNamespace(**args_dict_all)

    from multiprocessing import set_start_method, Process, Manager

    try:
        set_start_method("forkserver")
    except:
        pass

    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_collect_workflow, args=(args, retval))
        p.start()
        p.join()

        if p.exitcode != 0:
            sys.exit(p.exitcode)

        # Clean up master process before running workflow, which may create
        # forks
        gc.collect()
    mgr.shutdown()

    working_path = args_dict_all['basedir']
    modality = args_dict_all['modality']
    drop_cols = args_dict_all['dc']
    # working_path = args.basedir
    # modality = args.modality
    # drop_cols = args.dc

    all_files = glob.glob(
        f"{str(Path(working_path))}/{modality}_group_topology_auc/*.csv"
    )

    files_ = [i for i in all_files if '_clean.csv' in i]

    dfs = []
    #missingness_dict = {}
    for file_ in files_:
        try:
            df = pd.read_csv(file_, chunksize=100000, encoding="utf-8",
                             engine='python').read()
        except:
            try:
                df = pd.read_csv(file_, chunksize=100000, encoding="utf-8",
                                 engine='c').read()
            except:
                print(f"Cannot load {file_}...")
                continue

        if "Unnamed: 0" in df.columns:
            df.drop(df.filter(regex="Unnamed: 0"), axis=1, inplace=True)
        #missingness_dict[file_] = summarize_missingness(df)[1]
        df.set_index('id', inplace=True)
        df.index = df.index.map(str)
        dfs.append(df)
        del df

    print("Aggregating dataframes...")
    frame, rerun_dict = df_concat(dfs, working_path, modality, drop_cols, args)

    print("Missingness Summary:")
    summarize_missingness(frame)

    print(f"Rerun:\n{rerun_dict}")

    # Cleanup
    for j in all_files:
        if j not in files_:
            os.remove(j)

    print('\nDone!')
    return


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen" \
               "_importlib.BuiltinImporter'>)"
    main()
