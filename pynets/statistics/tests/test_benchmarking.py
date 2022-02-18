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

    def get_recursively(search_dict, field):
        """
        Takes a dict with nested lists and dicts,
        and searches all dicts for a key of the field
        provided.
        """
        fields_found = []

        for key, value in search_dict.items():

            if key == field:
                fields_found.append(value)

            elif isinstance(value, dict):
                results = get_recursively(value, field)
                for result in results:
                    fields_found.append(result)

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        more_results = get_recursively(item, field)
                        for another_result in more_results:
                            fields_found.append(another_result)

        return fields_found

    d1 = np.array(get_recursively(mega_dict, 'n_samples_100'))
    d2 = np.array(get_recursively(mega_dict, 'n_samples_1000'))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.4162766457698946

    # Test for mean difference based on n informative
    d1 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_informative_49')[0])))
    d2 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_informative_25')[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.7244168201421846

    # Test for mean difference based on n informative
    d1 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_informative_25')[0])))
    d2 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_informative_5')[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.9999143618449223

    # Test for mean difference based on n informative
    d1 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_informative_49')[0])))
    d2 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_informative_5')[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.9999959561894431

    # Test for mean difference based on n redundant
    d1 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_redundant_49')[0])))
    d2 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_redundant_25')[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.6177770457546709

    # Test for mean difference based on n redundant
    d1 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_redundant_25')[0])))
    d2 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_redundant_49')[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.3822229542453291

    # Test for mean difference based on n redundant
    d1 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_redundant_49')[0])))
    d2 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_redundant_5')[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.04145449595382497

    # Test for mean difference based on n features
    d1 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_features_100')[0])))
    d2 = np.array(list(NestedDictValues(get_recursively(mega_dict, 'n_features_1000')[0])))
    assert ttost_paired(d1, d2, 0, 1)
    assert ttost_paired(d1, d2, 0, 1)[0] == 0.016759998812705155


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
