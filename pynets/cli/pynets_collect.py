#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner (dPys)
"""
import os
import pandas as pd
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
import warnings
warnings.filterwarnings("ignore")


def get_parser():
    """Parse command-line inputs"""
    import argparse
    from pynets.__about__ import __version__
    verstr = f'pynets v{__version__}'

    # Parse args
    parser = argparse.ArgumentParser(description='PyNets: A Fully-Automated Workflow for Reproducible Ensemble '
                                                 'Sampling of Functional and Structural Connectomes')
    # Debug/Runtime settings
    parser.add_argument('-basedir',
                        metavar='Output directory',
                        help='Specify the path to the base output directory with group-level pynets derivatives.\n')
    parser.add_argument('-pm',
                        metavar='Cores,memory',
                        default='4,8',
                        help='Number of cores to use, number of GB of memory to use for single subject run, entered as '
                             'two integers seperated by comma.\n')
    parser.add_argument('-plug',
                        metavar='Scheduler type',
                        default='MultiProc',
                        nargs=1,
                        choices=['Linear', 'MultiProc', 'SGE', 'PBS', 'SLURM', 'SGEgraph', 'SLURMgraph',
                                 'LegacyMultiProc'],
                        help='Include this flag to specify a workflow plugin other than the default MultiProc.\n')
    parser.add_argument('-v',
                        default=False,
                        action='store_true',
                        help='Verbose print for debugging.\n')
    parser.add_argument('-work',
                        metavar='Working directory',
                        default='/tmp/work',
                        help='Specify the path to a working directory for pynets to run. Default is /tmp/work.\n')
    parser.add_argument('--version', action='version', version=verstr)
    return parser


def load_pd_dfs(file_):
    import gc
    import os.path as op
    import pandas as pd
    import numpy as np
    pd.set_option('display.float_format', lambda x: f'{x:.8f}')

    if file_:
        if op.isfile(file_) and not file_.endswith('_clean.csv'):
            df = pd.read_csv(file_, chunksize=100000).read()
            try:
                df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
            except:
                pass
            id = op.basename(file_).split('_netmets')[0]
            print(id)
            df['id'] = id
            try:
                df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            except:
                pass
            try:
                df.set_index('id', inplace=True)
            except:
                pass
            bad_cols1 = df.columns[df.columns.str.contains('_x')]
            if len(bad_cols1)>0:
                df.rename(columns=dict(zip(bad_cols1, [bad_col.split('_x')[0] for
                                                       bad_col in bad_cols1])), inplace=True)
            bad_cols2 = df.columns[df.columns.str.contains('_y')]
            if len(bad_cols2)>0:
                df.rename(columns=dict(zip(bad_cols2, [bad_col.split('_y')[0] for
                                                       bad_col in bad_cols2])), inplace=True)
            try:
                df = df.loc[:, ~df.columns.str.contains(r'.?\d{1}$', regex=True)]
            except:
                pass
            try:
                df = df.loc[:, ~df.columns.duplicated()]
            except:
                pass
            df.to_csv(f"{file_.split('.csv')[0]}{'_clean.csv'}", index=True)
            del bad_cols2
            del bad_cols1
            del id
            try:
                df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
            except:
                pass

        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    gc.collect()

    return df


def df_concat(dfs, working_path):
    import re
    import pandas as pd
    pd.set_option('display.float_format', lambda x: f'{x:.8f}')

    dfs = [df for df in dfs if df is not None]
    frame = pd.concat(dfs, axis=0, join='outer', sort=True, ignore_index=False)

    for i in list(frame.columns)[1:]:
        try:
            frame[i] = frame[i].astype('float32')
        except:
            try:
                frame[i] = pd.to_numeric(frame[i].apply(lambda x: re.sub('-', '', str(x))))
            except:
                pass

    frame = frame.drop_duplicates(subset='id')
    frame = frame.loc[:, ~frame.columns.str.contains(r'thr_auc$', regex=True)]
    # frame = frame.loc[:, (frame == 0).mean() < .5]
    # frame = frame.loc[:, frame.isnull().mean() <= 0.1]
    frame.to_csv(f"{working_path}{'/all_subs_neat.csv'}", index=False)
    return frame


def build_subject_dict(sub, working_path, modality='func'):
    import shutil
    import os
    import glob
    from pathlib import Path

    def load_pd_dfs_auc(atlas_name, prefix, auc_file):
        import pandas as pd
        pd.set_option('display.float_format', lambda x: f'{x:.8f}')

        df = pd.read_csv(auc_file, chunksize=100000, compression='gzip', encoding='utf-8').read()
        print(f"{'Atlas: '}{atlas_name}")
        prefix = f"{atlas_name}{'_'}{prefix}{'_'}"
        df_pref = df.add_prefix(prefix)
        return df_pref

    subject_dict = {}
    print(sub)
    subject_dict[sub] = {}
    sessions = sorted([i for i in os.listdir(f"{working_path}{'/'}{sub}") if i.startswith('ses-')],
                      key = lambda x: x.split("-")[1])
    atlases = list(
        set([os.path.basename(str(Path(i).parent.parent)) for i in
             glob.glob(f"{working_path}{'/'}{sub}{'/*/*/*/netmetrics/*'}")]))
    print(atlases)

    files_ = []
    for ses in sessions:
        print(ses)
        subject_dict[sub][ses] = []
        for atlas in atlases:
            atlas_name = '_'.join(atlas.split('_')[1:])
            auc_csvs = glob.glob(
                f"{working_path}/{sub}/{ses}/{modality}/{atlas}/netmetrics/auc/*")
            for auc_file in auc_csvs:
                prefix = os.path.basename(auc_file).split('.csv')[0].split('est-')[1].split("%s%s" % (modality,
                                                                                            'net_mets'))[0]
                try:
                    subject_dict[sub][ses].append(load_pd_dfs_auc(atlas_name, prefix, auc_file))
                except:
                    print('Missing auc file...')
                    continue
        list_ = subject_dict[sub][ses]
        print(list_)
        if len(list_) > 0:
            df_base = list_[0][[c for c in list_[0].columns if c.endswith('auc')]]
            for m in range(len(list_))[1:]:
                df_base = df_base.merge(list_[m][[c for c in list_[m].columns if c.endswith('auc')]], how='right',
                                        right_index=True, left_index=True)

            if os.path.isdir(f"{working_path}{'/'}{sub}{'/'}{ses}{'/'}{modality}"):
                out_path = f"{working_path}/{sub}/{ses}/{modality}/all_combinations_auc.csv"
                df_base.to_csv(out_path)
                out_path_new = f"{str(Path(working_path).parent)}/all_visits_netmets_auc/{sub}_{ses}_netmets_auc.csv"
                files_.append(out_path_new)
                shutil.copyfile(out_path, out_path_new)

            del df_base
        else:
            continue
        del list_

    return files_


def collect_all(working_path):
    import_list = ["import warnings", "warnings.filterwarnings(\"ignore\")", "import os",
                   "import numpy as np",  "import indexed_gzip", "import nibabel as nib",
                   "import glob", "import pandas as pd", "import shutil", "from pathlib import Path"]

    wf = pe.Workflow(name="load_pd_dfs")

    inputnode = pe.Node(niu.IdentityInterface(fields=['working_path']), name='inputnode')
    inputnode.inputs.working_path = working_path

    build_subject_dict_node = pe.Node(niu.Function(input_names=['sub', 'working_path'], output_names=['files_'],
                                                   function=build_subject_dict), name="build_subject_dict_node",
                                      imports=import_list)
    build_subject_dict_node.iterables = ('sub', [i for i in os.listdir(working_path) if i.startswith('sub-')])
    build_subject_dict_node.synchronize = True

    df_join_node = pe.JoinNode(niu.IdentityInterface(fields=['files_']), name='df_join_node', joinfield=['files_'],
                               joinsource=build_subject_dict_node)

    load_pd_dfs_map = pe.MapNode(niu.Function(input_names=['file_'], outputs_names=['df'], function=load_pd_dfs),
                                 name="load_pd_dfs", imports=import_list, iterfield=['file_'], nested=True)

    outputnode = pe.Node(niu.IdentityInterface(fields=['dfs']), name='outputnode')

    wf.connect([
        (inputnode, build_subject_dict_node, [('working_path', 'working_path')]),
        (build_subject_dict_node, df_join_node, [('files_', 'files_')]),
        (df_join_node, load_pd_dfs_map, [('files_', 'file_')]),
        (load_pd_dfs_map, outputnode, [('df', 'dfs')]),
    ])

    return wf


def build_collect_workflow(args, retval):
    import re
    import glob
    import warnings
    warnings.filterwarnings("ignore")
    import ast
    from pathlib import Path
    import yaml
    try:
        import pynets
        print(f"\n\nPyNets Version:\n{pynets.__version__}\n\n")
    except ImportError:
        print('PyNets not installed! Ensure that you are using the correct python version.')

    # Set Arguments to global variables
    resources = args.pm
    if resources:
        procmem = list(eval(str(resources)))
    else:
        from multiprocessing import cpu_count
        nthreads = cpu_count()
        procmem = [int(nthreads), int(float(nthreads) * 2)]
    plugin_type = args.plug
    if type(plugin_type) is list:
        plugin_type = plugin_type[0]
    verbose = args.v
    working_path = args.basedir
    work_dir = args.work

    os.makedirs(f"{str(Path(working_path).parent)}/all_visits_netmets_auc", exist_ok=True)

    wf = collect_all(working_path)

    #with open('/opt/conda/lib/python3.6/site-packages/pynets-0.9.94-py3.6.egg/pynets/runconfig.yaml', 'r') as stream:
    with open(f"{str(Path(__file__).parent.parent)}{'/runconfig.yaml'}", 'r') as stream:
        try:
            hardcoded_params = yaml.load(stream)
            runtime_dict = {}
            execution_dict = {}
            for i in range(len(hardcoded_params['resource_dict'])):
                runtime_dict[list(hardcoded_params['resource_dict'][i].keys())[0]] = ast.literal_eval(list(
                    hardcoded_params['resource_dict'][i].values())[0][0])
            for i in range(len(hardcoded_params['execution_dict'])):
                execution_dict[list(hardcoded_params['execution_dict'][i].keys())[0]] = list(
                    hardcoded_params['execution_dict'][i].values())[0][0]
        except FileNotFoundError:
            print('Failed to parse runconfig.yaml')

    os.makedirs(f"{work_dir}{'/pynets_out_collection'}", exist_ok=True)
    wf.base_dir = f"{work_dir}{'/pynets_out_collection'}"

    if verbose is True:
        from nipype import config, logging
        cfg_v = dict(logging={'workflow_level': 'DEBUG', 'utils_level': 'DEBUG', 'interface_level': 'DEBUG',
                              'filemanip_level': 'DEBUG', 'log_directory': str(wf.base_dir), 'log_to_file': True},
                     monitoring={'enabled': True, 'sample_frequency': '0.1', 'summary_append': True,
                                 'summary_file': str(wf.base_dir)})
        logging.update_logging(config)
        config.update_config(cfg_v)
        config.enable_debug_mode()
        config.enable_resource_monitor()

        import logging
        callback_log_path = f"{wf.base_dir}{'/run_stats.log'}"
        logger = logging.getLogger('callback')
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(callback_log_path)
        logger.addHandler(handler)

    execution_dict['crashdump_dir'] = str(wf.base_dir)
    execution_dict['plugin'] = str(plugin_type)
    cfg = dict(execution=execution_dict)
    for key in cfg.keys():
        for setting, value in cfg[key].items():
            wf.config[key][setting] = value
    try:
        wf.write_graph(graph2use="colored", format='png')
    except:
        pass
    if verbose is True:
        from nipype.utils.profiler import log_nodes_cb
        plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]),
                       'status_callback': log_nodes_cb, 'scheduler': 'mem_thread'}
    else:
        plugin_args = {'n_procs': int(procmem[0]), 'memory_gb': int(procmem[1]), 'scheduler': 'mem_thread'}
    print("%s%s%s" % ('\nRunning with ', str(plugin_args), '\n'))
    wf.run(plugin=plugin_type, plugin_args=plugin_args)
    if verbose is True:
        from nipype.utils.draw_gantt_chart import generate_gantt_chart
        print('Plotting resource profile from run...')
        generate_gantt_chart(callback_log_path, cores=int(procmem[0]))
        handler.close()
        logger.removeHandler(handler)

    files_ = glob.glob(f"{str(Path(working_path).parent)}{'/all_visits_netmets_auc/*clean.csv'}")

    print('Aggregating dataframes...')
    dfs = []
    for file_ in files_:
        df = pd.read_csv(file_, chunksize=100000).read()
        try:
            df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
        except:
            pass
        dfs.append(df)
        del df
    df_concat(dfs, working_path)

    return


def main():
    """Initializes collection of pynets outputs."""
    import gc
    import sys
    try:
        from pynets.core.utils import do_dir_path
    except ImportError:
        print('PyNets not installed! Ensure that you are referencing the correct site-packages and using Python3.5+')

    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h flag.\n")
        sys.exit()

    args = get_parser().parse_args()
    # args_dict_all = {}
    # args_dict_all['plug'] = 'MultiProc'
    # args_dict_all['v'] = False
    # args_dict_all['pm'] = '40,40'
    # args_dict_all['basedir'] = '/scratch/04171/dpisner/HNU/HNU_outs'
    # args_dict_all['work'] = '/scratch/04171/dpisner/pynets_scratch'
    # from types import SimpleNamespace
    # args = SimpleNamespace(**args_dict_all)

    from multiprocessing import set_start_method, Process, Manager
    set_start_method('forkserver')
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_collect_workflow, args=(args, retval))
        p.start()
        p.join()

        if p.exitcode != 0:
            sys.exit(p.exitcode)

        # Clean up master process before running workflow, which may create forks
        gc.collect()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
