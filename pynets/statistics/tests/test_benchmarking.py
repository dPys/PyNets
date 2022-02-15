#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:25:59 2021
@author: Kaelen Saythongkham
"""

import os
from sklearn.metrics.pairwise import (
    cosine_distances,
    haversine_distances,
    manhattan_distances,
    euclidean_distances,
)
import numpy as np
from sklearn.datasets import make_classification
from statsmodels.stats.weightstats import ttost_paired
from functools import reduce
from pathlib import Path
from pynets.statistics.group import benchmarking
import pytest
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def NestedDictValues(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from NestedDictValues(v)
    else:
      yield v


# @pytest.mark.parametrize("dissimilarity", ["euclidean", "precomputed"])
# @pytest.mark.parametrize("remove_isolates", [True, False])
# @pytest.mark.parametrize("n_informative", [5, 25, 49])
# @pytest.mark.parametrize("n_redundant", [0, 5, 25, 49])
# @pytest.mark.parametrize("n_features", [100, 1000])
# @pytest.mark.parametrize("n_samples", [100, 1000])
def test_discr_stat():
    mega_dict = {}

    dissimilarity = "euclidean"
    for remove_isolates in [True, False]:
        mega_dict[remove_isolates] = {}
        for n_informative in [5, 25, 49]:
            mega_dict[remove_isolates][f"n_informative_{n_informative}"] = {}
            for n_redundant in [5, 25, 49]:
                mega_dict[remove_isolates][f"n_informative_{n_informative}"][f"n_redundant_{n_redundant}"] = {}
                for n_features in [100, 1000]:
                    mega_dict[remove_isolates][f"n_informative_{n_informative}"][f"n_redundant_{n_redundant}"][f"n_features_{n_features}"] = {}
                    for n_samples in [100, 1000]:
                        X, Y = make_classification(n_samples, n_features, n_informative,
                                                    n_redundant, n_repeated=0, n_classes=2,
                                                    n_clusters_per_class=2, weights=None,
                                                    flip_y=0.01, class_sep=1.0, hypercube=True,
                                                    shift=0.0, scale=1.0, shuffle=True, random_state=42)
                        disc = benchmarking.discr_stat(X, Y, dissimilarity=dissimilarity)[0]
                        mega_dict[remove_isolates][f"n_informative_{n_informative}"][f"n_redundant_{n_redundant}"][f"n_features_{n_features}"][f"n_samples_{n_samples}"] = disc

    # Test for mean difference based on n samples

    d1 = np.array(list(reduce(dict.get, ['n_samples_100'], mega_dict)))
    d2 = np.array(list(reduce(dict.get, ['n_samples_1000'], mega_dict)))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.9961632262904134

    # Test for mean difference based on n informative
    d1 = np.array(list(NestedDictValues(reduce(dict.get, ['n_informative_49'], mega_dict)[0])))
    d2 = np.array(list(NestedDictValues(reduce(dict.get, ['n_informative_25'], mega_dict)[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.9988616434959907

    # Test for mean difference based on n informative
    d1 = np.array(list(NestedDictValues(reduce(dict.get, ['n_informative_25'], mega_dict)[0])))
    d2 = np.array(list(NestedDictValues(reduce(dict.get, ['n_informative_5'], mega_dict)[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.9999219809378421

    # Test for mean difference based on n informative
    d1 = np.array(list(NestedDictValues(reduce(dict.get, ['n_informative_49'], mega_dict)[0])))
    d2 = np.array(list(NestedDictValues(reduce(dict.get, ['n_informative_5'], mega_dict)[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.9999600191080593

    # Test for mean difference based on n redundant
    d1 = np.array(list(NestedDictValues(reduce(dict.get, ['n_redundant_49'], mega_dict)[0])))
    d2 = np.array(list(NestedDictValues(reduce(dict.get, ['n_redundant_25'], mega_dict)[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.13073863298448854

    # Test for mean difference based on n redundant
    d1 = np.array(list(NestedDictValues(reduce(dict.get, ['n_redundant_25'], mega_dict)[0])))
    d2 = np.array(list(NestedDictValues(reduce(dict.get, ['n_redundant_49'], mega_dict)[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.017425800816993416

    # Test for mean difference based on n redundant
    d1 = np.array(list(NestedDictValues(reduce(dict.get, ['n_redundant_49'], mega_dict)[0])))
    d2 = np.array(list(NestedDictValues(reduce(dict.get, ['n_redundant_5'], mega_dict)[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.012477146713475993

    # Test for mean difference based on n features
    d1 = np.array(list(NestedDictValues(reduce(dict.get, ['n_features_100'], mega_dict)[0])))
    d2 = np.array(list(NestedDictValues(reduce(dict.get, ['n_features_1000'], mega_dict)[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.18587215728638398


@pytest.mark.parametrize("n_informative", [5, 25, 49])
@pytest.mark.parametrize("n_redundant", [0, 5, 25, 49])
@pytest.mark.parametrize("n_features", [100, 1000])
@pytest.mark.parametrize("n_samples", [100, 1000])
def test_discr_rdf(
        n_informative,
        n_redundant,
        n_features,
        n_samples):
    X, Y = make_classification(n_samples, n_features, n_informative,
                                   n_redundant, n_repeated=0, n_classes=2,
                                   n_clusters_per_class=2, weights=None,
                                   flip_y=0.01, class_sep=1.0, hypercube=True,
                                   shift=0.0, scale=1.0, shuffle=True, random_state=42)
    uniques, counts = np.unique(Y, return_counts=True)
    idx = np.isin(Y, uniques[counts != 1])
    # assert X.shape[0] == Y.shape
    labels = Y[idx]
    X = X[idx]
    dissimilarities = euclidean_distances(X)
    rdfs = benchmarking._discr_rdf(dissimilarities, labels)
    assert isinstance(rdfs, np.ndarray)
    assert labels.shape[0] == rdfs.shape[0] == dissimilarities.shape[0]
    assert np.sum(rdfs >= 0) <= rdfs.shape[0]*rdfs.shape[1] and np.sum(rdfs <= 1) <= rdfs.shape[0]*rdfs.shape[1]
