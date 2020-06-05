#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from pynets.core.utils import flatten


def _omni_embed(pop_array, atlas, graph_path, ID, subgraph_name='whole_brain'):
    from graspy.embed import OmnibusEmbed, ClassicalMDS
    variance_threshold = VarianceThreshold(threshold=0.00001)
    diags = np.array([np.triu(pop_array[i]) for i in range(len(pop_array))])
    graphs_ix_keep = variance_threshold.fit(diags.reshape(diags.shape[0],
                                                          diags.shape[1] *
                                                          diags.shape[2]).T).get_support(indices=True)
    pop_array_red = [pop_array[i] for i in graphs_ix_keep]

    # Omnibus embedding -- random dot product graph (rdpg)
    print(f"{'Embedding unimodal omnetome for atlas: '}{atlas}{' and '}{subgraph_name}{'...'}")
    omni = OmnibusEmbed(check_lcc=False)
    mds = ClassicalMDS()
    try:
        omni_fit = omni.fit_transform(pop_array_red)
    except:
        omni_fit = omni.fit_transform(pop_array)

    # Transform omnibus tensor into dissimilarity feature
    mds_fit = mds.fit_transform(omni_fit)

    dir_path = str(Path(os.path.dirname(graph_path)).parent)

    namer_dir = dir_path + '/embeddings'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    out_path = f"{namer_dir}/{list(flatten(ID))[0]}_omnetome_{atlas}_{subgraph_name}.npy"
    print('Saving...')
    np.save(out_path, mds_fit)
    del mds, mds_fit, omni, omni_fit
    return out_path


def _mase_embed(pop_array, atlas, graph_path, ID, subgraph_name='whole_brain'):
    from graspy.embed import MultipleASE

    variance_threshold = VarianceThreshold(threshold=0.00001)
    diags = np.array([np.triu(pop_array[i]) for i in range(len(pop_array))])
    graphs_ix_keep = variance_threshold.fit(diags.reshape(diags.shape[0],
                                                          diags.shape[1] *
                                                          diags.shape[2]).T).get_support(indices=True)
    pop_array_red = [pop_array[i] for i in graphs_ix_keep]

    # Omnibus embedding -- random dot product graph (rdpg)
    print(f"{'Embedding multimodal masetome for atlas: '}{atlas}{' and '}{subgraph_name}{'...'}")
    mase = MultipleASE()
    try:
        mase_fit = mase.fit_transform(pop_array_red)
    except:
        mase_fit = mase.fit_transform(pop_array)

    dir_path = str(Path(os.path.dirname(graph_path)))
    namer_dir = dir_path + '/embeddings'
    if not os.path.isdir(namer_dir):
        os.makedirs(namer_dir, exist_ok=True)

    out_path = f"{namer_dir}/{list(flatten(ID))[0]}_masetome_{atlas}_{subgraph_name}.npy"
    print('Saving...')
    np.save(out_path, mase.scores_)
    del mase, mase_fit

    return out_path


def build_masetome(est_path_iterlist, ID):
    """
    Embeds structural-functional graph pairs into a common invariant subspace.

    Parameters
    ----------
    est_path_iterlist : list
        List of list of pairs of file paths (.npy) corresponding to
        structural and functional connectomes matched at a given node resolution.
    ID : str
        A subject id or other unique identifier.
    """
    from pynets.stats.embeddings import _mase_embed

    out_paths = []
    for pairs in est_path_iterlist:
        pop_list = []
        for _file in pairs:
            pop_list.append(np.load(_file))
        res = '_'.join(pairs[0].split('/')[-1].split('_')[1:]).split('_est')[0]
        if 'rsn' in res:
            subgraph = res.split('_rsn-')[1]
        else:
            subgraph = 'whole_brain'
        out_path = _mase_embed(pop_list, res.split('_')[0], _file, ID, subgraph_name=subgraph)
        out_paths.append(out_path)

    return out_paths


def build_omnetome(est_path_iterlist, ID):
    """
    Embeds ensemble population of graphs into an embedded ensemble feature vector.

    Parameters
    ----------
    est_path_iterlist : list
        List of file paths to .npy file containing graph without thresholding applied.
    ID : str
        A subject id or other unique identifier.
    """
    import yaml
    import pkg_resources
    from pynets.stats.embeddings import _omni_embed

    # Available functional and structural connectivity models
    with open(pkg_resources.resource_filename("pynets", "runconfig.yaml"), 'r') as stream:
        hardcoded_params = yaml.load(stream)
        try:
            func_models = hardcoded_params['available_models']['func_models']
        except KeyError:
            print('ERROR: available functional models not sucessfully extracted from runconfig.yaml')
        try:
            struct_models = hardcoded_params['available_models']['struct_models']
        except KeyError:
            print('ERROR: available structural models not sucessfully extracted from runconfig.yaml')
    stream.close()

    atlases = list(set([x.split('/')[-3].split('/')[0] for x in est_path_iterlist]))
    parcel_dict_func = dict.fromkeys(atlases)
    parcel_dict_dwi = dict.fromkeys(atlases)

    est_path_iterlist_dwi = list(set([i for i in est_path_iterlist if i.split('est-')[1].split('_')[0] in
                                      struct_models]))
    est_path_iterlist_func = list(set([i for i in est_path_iterlist if i.split('est-')[1].split('_')[0] in
                                       func_models]))

    func_subnets = list(set([i.split('_est')[0].split('/')[-1] for i in est_path_iterlist_func]))

    dwi_subnets = list(set([i.split('_est')[0].split('/')[-1] for i in est_path_iterlist_dwi]))

    out_paths_func = []
    out_paths_dwi = []
    for atlas in atlases:
        if len(func_subnets) > 1:
            parcel_dict_func[atlas] = {}
            for sub_net in func_subnets:
                parcel_dict_func[atlas][sub_net] = []
        else:
            parcel_dict_func[atlas] = []

        if len(dwi_subnets) > 1:
            parcel_dict_dwi[atlas] = {}
            for sub_net in dwi_subnets:
                parcel_dict_dwi[atlas][sub_net] = []
        else:
            parcel_dict_dwi[atlas] = []

        for graph_path in est_path_iterlist_dwi:
            if atlas in graph_path:
                if len(dwi_subnets) > 1:
                    for sub_net in dwi_subnets:
                        if sub_net in graph_path:
                            parcel_dict_dwi[atlas][sub_net].append(graph_path)
                else:
                    parcel_dict_dwi[atlas].append(graph_path)

        for graph_path in est_path_iterlist_func:
            if atlas in graph_path:
                if len(func_subnets) > 1:
                    for sub_net in func_subnets:
                        if sub_net in graph_path:
                            parcel_dict_func[atlas][sub_net].append(graph_path)
                else:
                    parcel_dict_func[atlas].append(graph_path)

        pop_list = []
        for pop_ref in parcel_dict_func[atlas]:
            # RSN case
            if isinstance(pop_ref, dict):
                rsns = [i.split('_')[1] for i in list(pop_ref.keys())]
                i = 0
                for rsn in rsns:
                    pop_rsn_list = []
                    for graph in pop_ref[rsn]:
                        pop_list.append(np.load(graph))
                    if len(pop_rsn_list) > 1:
                        if len(list(set([i.shape for i in pop_rsn_list]))) > 1:
                            raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population '
                                                 'that precludes embedding')
                        out_path = _omni_embed(pop_list, atlas, graph_path, ID, rsns[i])
                        out_paths_func.append(out_path)
                    else:
                        print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                        pass
                    i = i + 1
            else:
                pop_list.append(np.load(pop_ref))
        if len(pop_list) > 1:
            if len(list(set([i.shape for i in pop_list]))) > 1:
                raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population that '
                                     'precludes embedding')
            out_path = _omni_embed(pop_list, atlas, graph_path, ID)
            out_paths_func.append(out_path)
        else:
            print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
            pass

        pop_list = []
        for pop_ref in parcel_dict_dwi[atlas]:
            # RSN case
            if isinstance(pop_ref, dict):
                rsns = [i.split('_')[1] for i in list(pop_ref.keys())]
                i = 0
                for rsn in rsns:
                    pop_rsn_list = []
                    for graph in pop_ref[rsn]:
                        pop_list.append(np.load(graph))
                    if len(pop_rsn_list) > 1:
                        if len(list(set([i.shape for i in pop_rsn_list]))) > 1:
                            raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population '
                                                 'that precludes embedding')
                        out_path = _omni_embed(pop_list, atlas, graph_path, ID, rsns[i])
                        out_paths_dwi.append(out_path)
                    else:
                        print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
                        pass
                    i = i + 1
            else:
                pop_list.append(np.load(pop_ref))
        if len(pop_list) > 1:
            if len(list(set([i.shape for i in pop_list]))) > 1:
                raise RuntimeWarning('ERROR: Inconsistent number of vertices in graph population that '
                                     'precludes embedding')
            out_path = _omni_embed(pop_list, atlas, graph_path, ID)
            out_paths_dwi.append(out_path)
        else:
            print('WARNING: Only one graph sampled, omnibus embedding not appropriate.')
            pass

    return out_paths_dwi, out_paths_func
