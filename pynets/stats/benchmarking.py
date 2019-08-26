#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@authors: Neurodata, Alex Loftus, Derek Pisner
"""
import warnings
import numpy as np
import re
import networkx as nx
from pathlib import Path
from math import sqrt, ceil
from sklearn.metrics import euclidean_distances
from collections import namedtuple
from sklearn.utils import check_X_y
from matplotlib import pyplot as plt
from graspy.utils import pass_to_ranks, import_edgelist
from graspy.plot import heatmap
from functools import reduce

KEYWORDS = ["sub", "ses"]


def is_graph(filename, atlas="", suffix=""):
    """
    Check if `filename` is a pynets graph file.
    
    Parameters
    ----------
    filename : str or Path
        location of the file.
    
    Returns
    -------
    bool
        True if the file has the pynets naming convention, else False.
    """

    if atlas:
        atlas = atlas.lower()
        KEYWORDS.append(atlas)

    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix

    correct_suffix = Path(filename).suffix == suffix
    correct_filename = all(i in str(filename) for i in KEYWORDS)
    return correct_suffix and correct_filename


def filter_graph_files(file_list, return_bool=False, **kwargs):
    """
    Generator. 
    Check if each file in `file_list` is a pynets edgelist,
    yield it if it is.
    
    Parameters
    ----------
    return_bool : bool
        if True, return a boolean that says whether graph files exist in the 
        directory.
    file_list : iterator
        iterator of inputs to the `is_graph` function.
    """
    if return_bool:
        has_graphs = any(is_graph(x, **kwargs) for x in file_list)
        return has_graphs

    for filename in file_list:
        if is_graph(filename, **kwargs):
            yield (filename)


def discr_stat(X, Y, dissimilarity="euclidean", remove_isolates=True, return_rdfs=False):
    """
    Computes the discriminability statistic.
    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        Input data. If dissimilarity=='precomputed', the input should be the 
        dissimilarity matrix.
    Y : 1d-array, shape (n_samples)
        Input labels.
    dissimilarity : str, {"euclidean" (default), "precomputed"}
        Dissimilarity measure to use:
        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.
        - 'precomputed':
            Pre-computed dissimilarities.
    remove_isolates : bool, optional, default=True
        Whether to remove data that have single label.
    return_rdfs : bool, optional, default=False
        Whether to return rdf for all data points.
    Returns
    -------
    stat : float
        Discriminability statistic. 
    rdfs : array, shape (n_samples, max{len(id)})
        Rdfs for each sample. Only returned if ``return_rdfs==True``.
    """
    check_X_y(X, Y, accept_sparse=True)

    uniques, counts = np.unique(Y, return_counts=True)
    if (counts != 1).sum() <= 1:
        msg = "You have passed a vector containing only a single unique sample id."
        raise ValueError(msg)
    if remove_isolates:
        idx = np.isin(Y, uniques[counts != 1])
        labels = Y[idx]

        if dissimilarity == "euclidean":
            X = X[idx]
        else:
            X = X[np.ix_(idx, idx)]
    else:
        labels = Y

    if dissimilarity == "euclidean":
        dissimilarities = euclidean_distances(X)
    else:
        dissimilarities = X

    rdfs = _discr_rdf(dissimilarities, labels)
    stat = np.nanmean(rdfs)

    if return_rdfs:
        return stat, rdfs
    else:
        return stat


def _discr_rdf(dissimilarities, labels):
    """
    A function for computing the reliability density function of a dataset.
    Parameters
    ----------
    dissimilarities : array, shape (n_samples, n_features) or (n_samples, n_samples)
        Input data. If dissimilarity=='precomputed', the input should be the 
        dissimilarity matrix.
    labels : 1d-array, shape (n_samples)
        Input labels.
    Returns
    -------
    out : array, shape (n_samples, max{len(id)})
        Rdfs for each sample. Only returned if ``return_rdfs==True``.
    """
    check_X_y(dissimilarities, labels, accept_sparse=True)

    rdfs = []
    for i, label in enumerate(labels):
        di = dissimilarities[i]

        # All other samples except its own label
        idx = labels == label
        Dij = di[~idx]

        # All samples except itself
        idx[i] = False
        Dii = di[idx]

        rdf = [1 - ((Dij < d).sum() + 0.5 * (Dij == d).sum()) / Dij.size for d in Dii]
        rdfs.append(rdf)

    out = np.full((len(rdfs), max(map(len, rdfs))), np.nan)
    for i, rdf in enumerate(rdfs):
        out[i, : len(rdf)] = rdf

    return out


def replace_doc(value):
    """
    Decorator for changing docstring of a function.
    
    Parameters
    ----------
    value : str
        docstring to change to.
    
    Returns
    -------
    func
        wrapper function.
    """

    def _doc(func):
        func.__doc__ = value
        return func

    return _doc


def nearest_square(num):
    """ 
    Return the smallest square number greater than `num`.
    For use in visualize_biggraph(), to make the correct number of axes.
    
    Parameters:
    -----------
        num: int
    Returns: 
    --------
        int. Square number.
    """
    return ceil(sqrt(num)) ** 2


class PyNetsGraphs(object):

    """    
    PyNetsDirectory which contains graph objects.

    Parameters
    ----------
    delimiter : str
        The delimiter used in edgelists    
    Attributes
    ----------
    delimiter : str
        The delimiter used in edgelists    
    vertices : np.ndarray
        sorted union of all nodes across edgelists.
    graphs : np.ndarray, shape (n, v, v), 3D
        Volumetric numpy array, n vxv adjacency matrices corresponding to each edgelist.
        graphs[0, :, :] corresponds to files[0].
    subjects : np.ndarray, shape n, 1D
        subject IDs, sorted set of all subject IDs in `dir`.
        Y[0] corresponds to files[0].
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.vertices = self._vertices()
        self.graphs = self._graphs()
        self.subjects = self._parse()[0]
        self.sessions = self._parse()[0]

    def __repr__(self):
        return f"PyNetsGraphs : {str(self.directory)}"

    def _nx_graphs(self):
        """
        List of networkx graph objects. Hidden property, mainly for use to calculate vertices.
        Returns
        -------
        nx_graphs : List[nx.Graph]
            List of networkX graphs corresponding to subjects.
        """
        nx_graphs = [
            nx.read_weighted_edgelist(f, nodetype=int, delimiter=self.delimiter)
            for f in self.files
        ]
        return nx_graphs

    def _vertices(self):
        nx_graphs = self._nx_graphs()
        return np.sort(reduce(np.union1d, [G.nodes for G in nx_graphs]))

    def _graphs(self):
        """
        volumetric numpy array, shape (n, v, v),
        accounting for isolate nodes by unioning the vertices of all component edgelists,
        sorted in the same order as `self.files`.
        Returns
        -------
        graphs : np.ndarray, shape (n, v, v), 3D
            Volumetric numpy array, n vxv adjacency matrices corresponding to each edgelist.
            graphs[0, :, :] corresponds to files[0].D
        """
        list_of_arrays = import_edgelist(self.files, delimiter=self.delimiter)
        if not isinstance(list_of_arrays, list):
            list_of_arrays = [list_of_arrays]
        return np.atleast_3d(list_of_arrays)

    def _parse(self):
        """
        Get subject IDs
        
        Returns
        -------
        out : np.ndarray 
            Array of strings. Each element is a subject ID.
        """
        pattern = r"(?<=sub-|ses-)(\w*)(?=_ses|_dwi)"
        subjects = [re.findall(pattern, str(edgelist))[0] for edgelist in self.files]
        sessions = [re.findall(pattern, str(edgelist))[1] for edgelist in self.files]
        return np.array(subjects), np.array(sessions)


class PyNetsStats(object):
    """Compute statistics from a pynets directory.
    Parameters
    ----------
    X : np.ndarray, shape (n, v*v), 2D
        numpy array, created by vectorizing each adjacency matrix and stacking.
    Methods
    -------
    pass_to_ranks : returns None 
        change state of object.
        calls pass to ranks on `self.graphs`, `self.X`, or both.
    save_X_and_Y : returns None
        Saves `self.X` and `self.Y` into a directory.
    discriminability : return float
        discriminability statistic for this dataset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = self._X()
        self.Y = self.subjects

    def __repr__(self):
        return f"PyNetsStats : {str(self.directory)}"

    def _X(self, graphs=None):
        """
        this will be a single matrix,
        created by vectorizing each array in `self.graphs`,
        and then appending that array as a row to X.
        Parameters
        ----------
        graphs : None or np.ndarray
            if None, graphs will be `self.graphs`.
        Returns
        -------
        X : np.ndarray, shape (n, v*v), 2D
            numpy array, created by vectorizing each adjacency matrix and stacking.
        """
        if graphs is None:
            graphs = self.graphs
        if graphs.ndim == 3:
            n, v1, v2 = np.shape(graphs)
            return np.reshape(graphs, (n, v1 * v2))
        elif len(self.files) == 1:
            warnings.warn("Only one graph in directory.")
            return graphs
        else:
            raise ValueError("Dimensionality of input must be 3.")

    def save_X_and_Y(self, output_directory="cwd", output_name=""):
        """
        Save `self.X` and `self.subjects` into an output directory.
        Parameters
        ----------
        output_directory : str, default current working directory
            Directory in which to save the output.
        Returns
        -------
        namedtuple with str
            namedtuple of `name.X, name.Y`. Paths to X and Y.
        """
        if not output_name:
            output_name = self.name

        if output_directory == "cwd":
            output_directory = Path.cwd()
        p = Path(output_directory)
        p.mkdir(parents=True, exist_ok=True)

        X_name = f"{str(p)}/{output_name}_X.csv"
        Y_name = f"{str(p)}/{output_name}_Y.csv"

        np.savetxt(X_name, self.X, fmt="%f", delimiter=",")
        np.savetxt(Y_name, self.subjects, fmt="%s")

        name = namedtuple("name", ["X", "Y"])
        return name(X_name, Y_name)

    @replace_doc(discr_stat.__doc__)
    def discriminability(self, PTR=True, **kwargs):
        """
        Attach discriminability functionality to the object.
        See `discr_stat` for full documentation.
        
        Returns
        -------
        stat : float
            Discriminability statistic.
        """
        if PTR:
            graphs = np.copy(self.graphs)
            graphs = np.array([pass_to_ranks(graph) for graph in graphs])
            X = self._X(graphs)
            return discr_stat(X, self.Y, **kwargs)

        return discr_stat(self.X, self.Y, **kwargs)


    def visualize(self, i, savedir=""):
        """
        Visualize the ith graph of self.graphs, passed-to-ranks.
        
        Parameters
        ----------
        i : int
            Graph to visualize.
        savedir : str, optional
            Directory to save graph into.
            If left empty, do not save.
        """

        nmax = np.max(self.graphs)

        if isinstance(i, int):
            graph = pass_to_ranks(self.graphs[i])
            sub = self.subjects[i]
            sesh = ""  # TODO
        
        elif isinstance(i, np.ndarray):
            graph = pass_to_ranks(i)
            sub = ""
            sesh = ""
        
        else:
            raise TypeError("Passed value must be integer or np.ndarray.")

        viz = heatmap(graph, title = f"sub-{sub}_session-{sesh}", xticklabels=True, yticklabels=True, vmin=0, vmax=1)

        # set color of title
        viz.set_title(viz.get_title(), color="black")

        # set color of colorbar ticks
        viz.collections[0].colorbar.ax.yaxis.set_tick_params(color="black")

        # set font size and color of heatmap ticks
        for item in (viz.get_xticklabels() + viz.get_yticklabels()):
            item.set_color("black")
            item.set_fontsize(7)

        if savedir:
            p = Path(savedir).resolve()
            if not p.is_dir():
                p.mkdir()
            plt.savefig(p / f"sub-{sub}_sesh-{sesh}.png", facecolor="white", bbox_inches="tight", dpi=300)
        else:
            plt.show()

        plt.cla()


def url_to_pynets_dir(urls):
    """
    take a list of urls or filepaths,
    get a dict of PyNetsGraphs objects
    
    Parameters
    ----------
    urls : list
        list of urls or filepaths. 
        Each element should be of the same form as the input to a `PyNetsGraphs` object.
    
    Returns
    -------
    dict
        dict of {dataset:PyNetsGraphs} objects.
    
    Raises
    ------
    TypeError
        Raises error if input is not a list.
    """

    # checks for type
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list):
        raise TypeError("urls must be a list of URLs.")

    # appends each object
    return_value = {}
    for url in urls:
        try:
            val = PyNetsStats(url)
            key = val.name
            return_value[key] = val
        except ValueError:
            warnings.warn(f"Graphs for {url} not found. Skipping ...")
            continue

    return return_value
