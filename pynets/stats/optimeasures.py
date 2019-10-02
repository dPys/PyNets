#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Kamil Bona
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def identifiability(sub_list, ses_list, gv_array, measure, ses1, ses2):
    '''
    This function calculates the identifiability of subjects as I_diff=I_self-I_others
    where I_self is similarity between the same subject in two different sessions averaged over all subjects
    and I_others is similarity between a given subject and all the others in two different sessions averaged
    over all subjects.

    Parameters
    ----------
    sub_list : list
        Vector of subjects.
    ses_list : list
        Vector with session numbers.
    gv_array : ndarray
        Array of shape (number of subjects * number of sessions) x (number of graph measures).
    measure : str
        Distance measure. Options are 'euclidean', 'cityblock', 'braycurtis'.
    ses1 : int
        Index of reference session number.
    ses2 : int
        Index of comparison session number.

    Returns
    -------
    I_diff : float64
        Identifiability value.
    '''

    from scipy.spatial.distance import cityblock, euclidean, braycurtis

    # Define cosine similarity between two vectors
    def dot(A, B):
        return sum(a * b for a, b in zip(A, B))

    # Find number of subjects and number of sessions
    N_ses = int(max(ses_list))
    N_sub = (len(sub_list))

    # Calculate identifiability matrix
    I_mat = np.zeros((N_sub, N_sub))
    if measure == 'euclidean':
        for sub1 in range(N_sub):
            for sub2 in range(N_sub):
                I_mat[int(sub1) - 1, int(sub2) - 1] = euclidean(gv_array[int(sub1) * N_ses + ses1 - 3, :],
                                                                gv_array[int(sub2) * N_ses + ses2 - 3, :])
    elif measure == 'cityblock':
        for sub1 in range(N_sub):
            for sub2 in range(N_sub):
                I_mat[int(sub1) - 1, int(sub2) - 1] = cityblock(gv_array[int(sub1) * N_ses + ses1 - 3, :],
                                                                gv_array[int(sub2) * N_ses + ses2 - 3, :])
    elif measure == 'braycurtis':
        for sub1 in range(N_sub):
            for sub2 in range(N_sub):
                I_mat[int(sub1) - 1, int(sub2) - 1] = braycurtis(gv_array[int(sub1) * N_ses + ses1 - 3, :],
                                                                 gv_array[int(sub2) * N_ses + ses2 - 3, :])

    # Create an out-of-diagonal elements mask
    out = np.ones((N_sub, N_sub), dtype=bool)
    np.fill_diagonal(out, 0)

    # Similarity of subject to others, averaged over all subjects
    I_others = np.mean(I_mat[out])

    # Similarity of subject to himself, averaged over all subjects
    I_self = np.mean(np.diagonal(I_mat))
    I_diff = I_self / I_others

    return I_diff


def beta_lin_comb(beta, GVDAT, meta):
    '''
    This function calculates linear combinations of graph vectors stored in GVDAT
    for all subjects and all sessions given the weights vector beta.

    Parameters
    ----------
    beta : list
        List of metaparameter weights.
    GVDAT : ndarray
        5d data structure storing graph vectors.

    Returns
    -------
    gv_array : ndarray
        2d array of aggregated graph vectors for all scans.
    '''
    import numpy as np
    import math

    def normalize_beta(beta):
        sum_weight = sum(
            [b1 * b2 * b3 for b1 in beta[:N_atl] for b2 in beta[N_atl:N_atl + N_mod] for b3 in beta[-N_thr:]])
        return [b / math.pow(sum_weight, 1 / 3) for b in beta]

    # Dataset dimensionality
    N_sub = meta['N_sub']
    N_ses = meta['N_ses']
    N_gvm = meta['N_gvm']
    N_thr = len(meta['thr'])
    N_atl = len(meta['atl'])
    N_mod = len(meta['mod'])

    # Normalize and split full beta vector
    beta = normalize_beta(beta)
    beta_atl = beta[:N_atl]
    beta_mod = beta[N_atl:N_atl + N_mod]
    beta_thr = beta[-N_thr:]

    # Calculate linear combintations
    gv_array = np.zeros((N_sub * N_ses, N_gvm), dtype='float')
    for scan in range(N_sub * N_ses):
        gvlc = 0  # graph-vector linear combination
        for atl in range(N_atl):
            for mod in range(N_mod):
                for thr in range(N_thr):
                    gvlc += GVDAT[scan][atl][mod][thr] * beta_atl[atl] * beta_mod[mod] * beta_thr[thr]
        gv_array[scan] = gvlc
    return gv_array


def calc_graph_vector(dir_path):
    '''
    This function calculates graph measures for connectivity matrix loaded from textfile
    and save results under the same name with additional superscript +'_GV' (in same dir
    filename is located)

    Parameters
    ----------
    dir_path : str
        Path to base directory of PyNets outputs.
    '''
    import os
    import glob
    import pandas as pd
    from itertools import groupby

    measures = ['global_efficiency', 'degree_pearson_correlation_coefficient', 'average_clustering',
                'average_local_efficiency', 'transitivity', 'average_eigenvector_centrality',
                'average_participation_coefficient', 'average_betweenness_centrality', 'average_degree_cent',
                'average_diversity_coefficient']

    filenames = [i for i in glob.glob(dir_path + '/*/netmetrics/*neat.csv') if 'frag' not in i]

    sub = dir_path.split('sub-')[1].split('/')[0]
    ses = dir_path.split('ses-')[1].split('/')[0]

    # Check inputs
    filtered_filenames = []
    for _file in filenames:
        if not os.path.exists(_file):
            raise Exception('{} does not exist'.format(_file))
        df = pd.read_csv(_file)
        na_cols = [i for i in df.columns[df.isna().any()].tolist() if i in measures]
        if len(na_cols) > 0:
            continue
        else:
            filtered_filenames.append(_file)

    # Number of graph measures
    N_measures = len(measures)

    thr_ix = filtered_filenames[0].split('_').index('thr-' + filtered_filenames[0].split('thr-')[1].split('_')[0])
    filtered_list_grouped_by_thr = [list(j) for i, j in groupby(filtered_filenames, lambda x: '_'.join(x.split('_')[:thr_ix] + x.split('_')[1+thr_ix:]))]

    if len(filtered_list_grouped_by_thr) > 0:
        namer_dir = dir_path + '/optimeasures'
        if not os.path.isdir(namer_dir):
            os.mkdir(namer_dir)
    else:
        raise ValueError('No files contain all of the graph measures specified.')

    # Calculate output vector
    for _list in filtered_list_grouped_by_thr:
        thresholds = []
        for _file in _list:
            thresholds.append(_file.split('_')[-3])

        # Create empty output matrix
        graph_measures = np.zeros([len(thresholds), N_measures])

        # Calculate output
        for thr in thresholds:
            thr_file = [i for i in _list if str(thr) in i][0]
            df = pd.read_csv(thr_file)
            measure_val_list = []
            for measure in measures:
                measure_val_list.append(df[measure][0])

            graph_measures[thresholds.index(thr)] = measure_val_list

        graph_vec_filename = namer_dir + '/sub-' + sub + '_ses-' + ses + '_' + ['_'.join(j.split('/')[-3:]).split('.csv')[0] for j in ['_'.join(i.split('_')[:-3:]) for i in _list]][0].replace('_netmetrics_', '_').replace('_' + sub + '_', '_') + '.npy'

        # Save vector to file
        np.save(graph_vec_filename, graph_measures)

    return


def quality_function(sub_list, gv_array, similarity):
    '''
    This function calculates identifiability quality function comparing within-subject
    similarity (wss) in graph vectors with between-subject similartiy (bss). Similarity
    is measured by cosine distance between graph vectors.

    Parameters
    ----------
    sub_list : list
        List of subject numbers corresponding to rows of gv_array.
    gv_array : np.ndarray
        Each row contains subject graph vector, shape is (N_sub*N_ses, N_gvm)
    similarity : str
        Name of similarity measure between graph vectors: 'cosine': cosine similarity,
        'euclid': euclidean distance (dissimilarity),
        'braycurtis': Bray-Curtis dissimilarity measure

    Returns
    -------
    output : float64
        Scalar value of quality function = wss / bss
    '''
    from scipy.special import comb
    import itertools

    if similarity == 'euclid':
        from scipy.spatial.distance import euclidean

        def vector_similarity(a, b):
            return euclidean(a, b)
    elif similarity == 'braycurtis':
        from scipy.spatial.distance import braycurtis

        def vector_similarity(a, b):
            return braycurtis(a, b)
    elif similarity == 'cosine':

        def dot(A, B):
            return (sum(a * b for a, b in zip(A, B)))

        def vector_similarity(a, b):
            return dot(a, b) / ((dot(a, a) ** .5) * (dot(b, b) ** .5))
    else:
        return Exception('Incorrect similarity measure!')

    # Create dictionary
    sub_dict = {}
    for sub in set(sub_list):
        sub_dict[sub] = [idx for idx, x in enumerate(sub_list) if sub == x]
    N_sub = len(sub_dict)
    N_ses = 4

    # Within-subject similarity
    within_sub_sum = 0
    for sub in set(sub_list):
        for index_pair in itertools.combinations(sub_dict[sub], r=2):
            within_sub_sum += vector_similarity(gv_array[index_pair[0]], gv_array[index_pair[1]])
    within_sub_sum /= comb(N=N_ses, k=2) * N_sub

    # Between-subject similarity
    betwen_sub_sum = 0
    for sub_pair in itertools.combinations(set(sub_list), r=2):
        for index_pair in itertools.product(sub_dict[sub_pair[0]], sub_dict[sub_pair[1]]):
            betwen_sub_sum += vector_similarity(gv_array[index_pair[0]], gv_array[index_pair[1]])
    betwen_sub_sum /= comb(N=N_sub, k=2) * N_ses ** 2

    output = within_sub_sum / betwen_sub_sum

    return output


def load_data(om_path, meta):
    '''
    This functions is loading the graph vector data from separate .npy files into
    one 5d-numpy-array.

    Parameters
    ----------
    om_path : str
        Path to folder containing .npy graph vector files
    meta : dict
        Dataset and metaparameter description

    Returns
    -------
    GVDAT : ndarray
        5d array with all graph data.
    sub_list : list
        :ist with subject numbers corresponding to 1st dimension in GVDAT array.
    '''
    import os
    import numpy as np

    # Dataset dimensionality
    N_sub = meta['N_sub']
    N_ses = meta['N_ses']
    N_gvm = meta['N_gvm']
    N_thr = len(meta['thr'])
    N_atl = len(meta['atl'])
    N_mod = len(meta['mod'])

    # Get files, ensure correct extension, extract subjects
    gv_files = os.listdir(om_path)
    gv_files = [file for file in gv_files if file.find('.npy') != -1]
    subs = sorted(list(set([file[11:13] for file in gv_files])))  # bring out sub number (as str)
    sub_list = [[sub for ses in range(N_ses)] for sub in subs]  # include multiple sessions
    sub_list = [sub for ses in sub_list for sub in ses]  # unnest list

    # Load data and store in GVDAT array
    GVDAT = np.zeros((N_ses * N_sub, N_atl, N_mod, N_thr, N_gvm), dtype='float')
    for sub in range(N_sub):  # subjects
        for ses in range(N_ses):  # sessions
            for idx_atl, atl in enumerate(meta['atl']):  # atlas
                for idx_mod, mod in enumerate(meta['mod']):  # model
                    filename = [f for f in os.listdir(om_path) if
                                'sub{subs[sub]}' in f and
                                'ses{str(ses + 1)}' in f and
                                '{atl}' in f and
                                '{mod}' in f]
                    if len(filename) != 1:
                        print(filename)
                        raise Exception('Missing file. Aborting data loading...')
                    GVDAT[4 * sub + ses, idx_atl, idx_mod] = np.load(om_path + filename[0])
    return GVDAT, sub_list


