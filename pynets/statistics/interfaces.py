#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:44:46 2017
Copyright (C) 2017
"""
import matplotlib
import numpy as np
import networkx as nx
import warnings
import sys
import time
import gc
import os
import os.path as op
import yaml
if sys.platform.startswith('win') is False:
    import indexed_gzip
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
)
from pathlib import Path
from pynets.core import utils, thresholding

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


class CleanGraphs(object):
    """
    A Class for cleaning graphs in preparation for subnet analysis.

    Parameters
    ----------
    est_path : str
        File path to the thresholded graph, conn_matrix_thr, saved as a numpy
        array in .npy format.
    prune : int
        Indicates whether to prune final graph of disconnected nodes/isolates.
    norm : int
        Indicates method of normalizing resulting graph.

    Returns
    -------
    out_path : str
        Path to .csv file where graph analysis results are saved.

    References
    ----------
    .. [1] Qin, Tai, and Karl Rohe. "Regularized spectral clustering
      under the degree-corrected stochastic blockmodel." In Advances
      in Neural Information Processing Systems, pp. 3120-3128. 2013
    .. [2] Rohe, Karl, Tai Qin, and Bin Yu. "Co-clustering directed graphs to
      discover asymmetries and directional communities." Proceedings of the
      National Academy of Sciences 113.45 (2016): 12679-12684.

    """

    def __init__(
            self,
            est_path,
            prune,
            norm,
            out_fmt="gpickle",
            remove_self_loops=True):
        import graspologic.utils as gu

        self.est_path = est_path
        self.prune = prune
        self.norm = norm
        self.out_fmt = out_fmt
        self.in_mat = None

        # Load and threshold matrix
        self.in_mat_raw = utils.load_mat(self.est_path)

        # De-diagnal and remove nan's and inf's, ensure edge weights are
        # positive
        self.in_mat = np.array(
            np.array(
                thresholding.autofix(
                    np.array(np.abs(
                        self.in_mat_raw)))))

        # Remove self-loops and ensure symmetry
        if remove_self_loops is True:
            self.in_mat = gu.remove_loops(gu.symmetrize(self.in_mat))
        else:
            self.in_mat = gu.symmetrize(self.in_mat)

        self.in_mat[np.where(np.isnan(self.in_mat) |
                             np.isinf(self.in_mat))] = 0

        # Create nx graph
        self.G = nx.from_numpy_array(self.in_mat)

    # Normalize connectivity matrix
    def normalize_graph(self):
        import graspologic.utils as gu

        # By maximum edge weight
        if self.norm == 1:
            self.in_mat = thresholding.normalize(np.nan_to_num(self.in_mat))
        # Apply log10
        elif self.norm == 2:
            self.in_mat = np.log10(np.nan_to_num(self.in_mat))
        # Apply PTR simple-nonzero
        elif self.norm == 3:
            self.in_mat = gu.ptr.pass_to_ranks(
                np.nan_to_num(self.in_mat), method="simple-nonzero"
            )
        # Apply PTR simple-all
        elif self.norm == 4:
            self.in_mat = gu.ptr.pass_to_ranks(
                np.nan_to_num(
                    self.in_mat),
                method="simple-all")
        # Apply PTR zero-boost
        elif self.norm == 5:
            self.in_mat = gu.ptr.pass_to_ranks(
                np.nan_to_num(
                    self.in_mat),
                method="zero-boost")
        # Apply standardization [0, 1]
        elif self.norm == 6:
            self.in_mat = thresholding.standardize(np.nan_to_num(self.in_mat))
        elif self.norm == 7:
            # Get hyperbolic tangent (i.e. fischer r-to-z transform) of matrix
            # if non-covariance
            self.in_mat = np.arctanh(self.in_mat)
        else:
            pass

        self.in_mat = thresholding.autofix(self.in_mat)
        self.G = nx.from_numpy_array(self.in_mat)

        return self.G

    def prune_graph(self):
        import graspologic.utils as gu
        from pynets.statistics.individual.algorithms import defragment, \
            prune_small_components, most_important

        hardcoded_params = utils.load_runconfig()

        if int(self.prune) not in range(0, 4):
            raise ValueError(f"Pruning option {self.prune} invalid!")

        if self.prune != 0:
            # Remove isolates
            G_tmp = self.G.copy()
            self.G = defragment(G_tmp)[0]
            del G_tmp

        if int(self.prune) == 1:
            try:
                self.G = prune_small_components(self.G,
                                                min_nodes=hardcoded_params[
                                                    "min_nodes"][0])
            except BaseException:
                print(UserWarning(f"Warning: pruning {self.est_path} "
                                  f"failed..."))
        elif int(self.prune) == 2:
            try:
                hub_detection_method = \
                hardcoded_params["hub_detection_method"][0]
                print(f"Filtering for hubs on the basis of "
                      f"{hub_detection_method}...\n")
                self.G = most_important(self.G,
                                        method=hub_detection_method)[0]
            except FileNotFoundError as e:
                import sys
                print(e, "Failed to parse advanced.yaml")

        elif int(self.prune) == 3:
            print("Pruning all but the largest connected "
                  "component subgraph...")
            self.G = gu.largest_connected_component(self.G)
        else:
            print("No graph defragmentation applied...")

        self.G = nx.from_numpy_array(self.in_mat)

        if nx.is_empty(self.G) is True or \
            (np.abs(self.in_mat) < 0.0000001).all() or \
                self.G.number_of_edges() == 0:
            print(UserWarning(f"Warning: {self.est_path} "
                              f"empty after pruning!"))
            return self.in_mat, None

        # Saved pruned
        if (self.prune != 0) and (self.prune is not None):
            final_mat_path = f"{self.est_path.split('.npy')[0]}{'_pruned'}"
            utils.save_mat(self.in_mat, final_mat_path, self.out_fmt)
            print(f"{'Source File: '}{final_mat_path}")

        return self.in_mat, final_mat_path

    def print_summary(self):
        for i in list(nx.info(self.G).split("\n"))[2:]:
            print(i)
        return

    def binarize_graph(self):
        in_mat_bin = thresholding.binarize(self.in_mat)

        # Load numpy matrix as networkx graph
        G_bin = nx.from_numpy_array(in_mat_bin)
        return in_mat_bin, G_bin

    def create_length_matrix(self):
        in_mat_len = thresholding.weight_conversion(self.in_mat, "lengths")

        # Load numpy matrix as networkx graph
        G_len = nx.from_numpy_array(in_mat_len)
        return in_mat_len, G_len


class NetworkAnalysisInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for NetworkAnalysis"""

    ID = traits.Any(mandatory=True)
    est_path = File(exists=True, mandatory=True)
    prune = traits.Any(mandatory=False)
    norm = traits.Any(mandatory=False)
    binary = traits.Bool(False, usedefault=True)


class NetworkAnalysisOutputSpec(TraitedSpec):
    """Output interface wrapper for NetworkAnalysis"""

    out_path_neat = File(exists=True, mandatory=True)


class NetworkAnalysis(BaseInterface):
    """Interface wrapper for NetworkAnalysis"""

    input_spec = NetworkAnalysisInputSpec
    output_spec = NetworkAnalysisOutputSpec

    def _run_interface(self, runtime):
        # import random
        import pkg_resources
        from pynets.statistics.individual import algorithms
        from pynets.statistics.utils import save_netmets

        # Load netstats config and parse graph algorithms as objects
        with open(
            pkg_resources.resource_filename("pynets",
                                            "statistics/individual/"
                                            "global.yaml"),
            "r",
        ) as stream:
            try:
                nx_algs = [
                    "degree_assortativity_coefficient",
                    "average_clustering",
                    "average_shortest_path_length",
                    "graph_number_of_cliques",
                ]
                pynets_algs = [
                    "average_local_efficiency",
                    "global_efficiency",
                    "smallworldness",
                    "weighted_transitivity",
                ]
                metric_dict_global = yaml.load(stream, Loader=yaml.FullLoader)
                metric_list_global = metric_dict_global["metric_list_global"]
                if metric_list_global is not None:
                    metric_list_global = [
                                             getattr(nx.algorithms, i)
                                             for i in metric_list_global
                                             if i in nx_algs
                                         ] + [
                                             getattr(algorithms, i)
                                             for i in metric_list_global
                                             if i in pynets_algs
                                         ]
                    metric_list_global_names = [
                        str(i).split("<function ")[1].split(" at")[0]
                        for i in metric_list_global
                    ]
                    if self.inputs.binary is False:
                        from functools import partial

                        metric_list_global = [
                            partial(i, weight="weight")
                            if "weight" in i.__code__.co_varnames
                            else i for i in metric_list_global
                        ]
                    print(
                        f"\n\nGlobal Topographic Metrics:"
                        f"\n{metric_list_global_names}\n")
                else:
                    print("No global topographic metrics selected!")
                    metric_list_global = []
                    metric_list_global_names = []
            except FileNotFoundError as e:
                import sys
                print(e, "Failed to parse global.yaml")

        with open(
            pkg_resources.resource_filename("pynets",
                                            "statistics/individual/"
                                            "local.yaml"),
            "r",
        ) as stream:
            try:
                metric_dict_nodal = yaml.load(stream, Loader=yaml.FullLoader)
                metric_list_nodal = metric_dict_nodal["metric_list_nodal"]
                if metric_list_nodal is not None:
                    print(
                        f"\nNodal Topographic Metrics:"
                        f"\n{metric_list_nodal}\n\n")
                else:
                    print("No nodal topographic metrics selected!")
                    metric_list_nodal = []
            except FileNotFoundError as e:
                import sys
                print(e, "Failed to parse local.yaml")

        if os.path.isfile(self.inputs.est_path):

            cg = CleanGraphs(self.inputs.est_path, self.inputs.prune,
                             self.inputs.norm)

            tmp_graph_path = None
            if float(self.inputs.prune) >= 1:
                tmp_graph_path = cg.prune_graph()[1]

            if float(self.inputs.norm) >= 1:
                try:
                    cg.normalize_graph()
                except ValueError as e:
                    print(e, f"Graph normalization failed for "
                             f"{self.inputs.est_path}.")

            if self.inputs.binary is True:
                try:
                    in_mat, G = cg.binarize_graph()
                except ValueError as e:
                    print(e, f"Graph binarization failed for "
                             f"{self.inputs.est_path}.")
                    in_mat = np.zeros(1, 1)
                    G = nx.Graph()
            else:
                in_mat, G = cg.in_mat, cg.G

            cg.print_summary()

            dir_path = op.dirname(op.realpath(self.inputs.est_path))

            # Deal with empty graphs
            if nx.is_empty(G) is True or (np.abs(in_mat) < 0.0000001).all() \
                or G.number_of_edges() == 0 or len(G) < 3:
                out_path_neat = save_netmets(
                    dir_path, self.inputs.est_path, [""], [np.nan])
                print(UserWarning(f"Warning: Empty graph detected for "
                                  f"{self.inputs.ID}: "
                                  f"{self.inputs.est_path}..."))
            else:
                try:
                    in_mat_len, G_len = cg.create_length_matrix()
                except ValueError as e:
                    print(e, f"Failed to create length matrix for "
                             f"{self.inputs.est_path}.")
                    G_len = None

                if len(metric_list_global) > 0:
                    # Iteratively run functions from above metric list that
                    # generate single scalar output
                    net_met_val_list_final, metric_list_names = \
                        algorithms.iterate_nx_global_measures(
                            G, metric_list_global)

                    # Run miscellaneous functions that generate multiple
                    # outputs Calculate modularity using the Louvain algorithm
                    if "louvain_modularity" in metric_list_global:
                        try:
                            start_time = time.time()
                            net_met_val_list_final, metric_list_names, ci = \
                                algorithms.get_community(
                                    G, net_met_val_list_final,
                                              metric_list_names)
                            print(
                                f"{np.round(time.time() - start_time, 3)}"
                                f"{'s'}")
                        except BaseException:
                            print(
                                "Louvain modularity calculation is undefined "
                                "for G")
                            # np.save("%s%s%s" % ('/tmp/community_failure',
                            # random.randint(1, 400), '.npy'),
                            #         np.array(nx.to_numpy_matrix(G)))
                            ci = None
                            pass
                    else:
                        ci = None
                else:
                    metric_list_names = []
                    net_met_val_list_final = []
                    ci = None

                if len(metric_list_nodal) > 0:
                    # Participation Coefficient by louvain community
                    if ci and "participation_coefficient" in metric_list_nodal:
                        try:
                            if not ci:
                                raise KeyError(
                                    "Participation coefficient cannot be "
                                    "calculated for G in the absence of a "
                                    "community affiliation vector")
                            start_time = time.time()
                            metric_list_names, net_met_val_list_final = \
                                algorithms.get_participation(in_mat, ci,
                                                  metric_list_names,
                                                  net_met_val_list_final)
                            print(
                                f"{np.round(time.time() - start_time, 3)}"
                                f"{'s'}")
                        except BaseException:
                            print(
                                "Participation coefficient cannot be "
                                "calculated for G")
                            # np.save("%s%s%s" % ('/tmp/partic_coeff_failure',
                            # random.randint(1, 400), '.npy'), in_mat)
                            pass
                    else:
                        if not ci and "participation_coefficient" in \
                            metric_list_nodal:
                            print(UserWarning(
                                "Skipping participation coefficient "
                                "because community affiliation is "
                                "empty for G..."))

                    # Diversity Coefficient by louvain community
                    if ci and "diversity_coefficient" in metric_list_nodal:
                        try:
                            if not ci:
                                raise KeyError(
                                    "Diversity coefficient cannot be "
                                    "calculated for G in the absence of a "
                                    "community affiliation vector")
                            start_time = time.time()
                            metric_list_names, net_met_val_list_final = \
                                algorithms.get_diversity(in_mat, ci,
                                                         metric_list_names,
                                              net_met_val_list_final
                                              )
                            print(
                                f"{np.round(time.time() - start_time, 3)}"
                                f"{'s'}")
                        except BaseException:
                            print(
                                "Diversity coefficient cannot be calculated "
                                "for G")
                            # np.save("%s%s%s" % ('/tmp/div_coeff_failure',
                            # random.randint(1, 400), '.npy'), in_mat)
                            pass
                    else:
                        if not ci and "diversity_coefficient" in \
                            metric_list_nodal:
                            print(UserWarning("Skipping diversity coefficient "
                                              "because community affiliation"
                                              " is empty for G..."))

                    # # Link communities
                    # if "link_communities" in metric_list_nodal:
                    #     try:
                    #         if ci is None:
                    #             raise KeyError(
                    #                 "Link communities cannot be calculated for
                    #                 G in the absence of a community affiliation
                    #                 vector")
                    #         start_time = time.time()
                    #         metric_list_names, net_met_val_list_final =
                    #         get_link_communities(
                    #             in_mat, ci, metric_list_names,
                    #             net_met_val_list_final
                    #         )
                    #         print(f"{np.round(time.time() - start_time, 3)}s")
                    #     except BaseException:
                    #         print("Link communities cannot be calculated for G")
                    #         # np.save("%s%s%s" % ('/tmp/link_comms_failure',
                    #         random.randint(1, 400), '.npy'), in_mat)
                    #         pass

                    # Betweenness Centrality
                    if "betweenness_centrality" in metric_list_nodal and \
                        G_len is not None:
                        try:
                            start_time = time.time()
                            metric_list_names, net_met_val_list_final = \
                                algorithms.get_betweenness_centrality(
                                    G_len, metric_list_names,
                                    net_met_val_list_final)
                            print(
                                f"{np.round(time.time() - start_time, 3)}"
                                f"{'s'}")
                        except BaseException:
                            print(
                                "Betweenness centrality cannot be calculated "
                                "for G")
                            # np.save("%s%s%s" % ('/tmp/betw_cent_failure',
                            # random.randint(1, 400), '.npy'),
                            #         np.array(nx.to_numpy_matrix(G_len)))
                            pass
                    else:
                        if G_len is None and "betweenness_centrality" in \
                            metric_list_nodal:
                            print(
                                UserWarning("Skipping betweenness centrality "
                                            "because length matrix is empty "
                                            "for G..."))

                    for i in ["local_efficiency", "local_clustering",
                              "degree_centrality",
                              "eigen_centrality",
                              "communicability_centrality",
                              "rich_club_coefficient"]:
                        if i in metric_list_nodal:
                            routine = getattr(algorithms, f"get_{i}")
                            try:
                                start_time = time.time()
                                metric_list_names, net_met_val_list_final = \
                                    routine(G, metric_list_names,
                                        net_met_val_list_final)
                                print(
                                    f"{np.round(time.time() - start_time, 3)}"
                                    f"{'s'}")
                            except BaseException:
                                print(f"{i} cannot be calculated for G")
                                # np.save("%s%s%s" % (f"/tmp/local_{i}_failure",
                                # random.randint(1, 400), '.npy'),
                                #         np.array(nx.to_numpy_matrix(G)))
                                pass

                if len(metric_list_nodal) > 0 or len(metric_list_global) > 0:
                    out_path_neat = save_netmets(
                        dir_path, self.inputs.est_path, metric_list_names,
                        net_met_val_list_final
                    )
                    # Cleanup
                    if tmp_graph_path:
                        if os.path.isfile(tmp_graph_path):
                            os.remove(tmp_graph_path)

                    del net_met_val_list_final, metric_list_names, \
                        metric_list_global
                    gc.collect()
                else:
                    out_path_neat = save_netmets(
                        dir_path, self.inputs.est_path, [""], [np.nan],
                    )
        else:
            print(UserWarning(f"Warning: Empty graph detected for "
                              f"{self.inputs.ID}: "
                              f"{self.inputs.est_path}..."))
            dir_path = op.dirname(op.realpath(self.inputs.est_path))
            out_path_neat = save_netmets(
                dir_path, self.inputs.est_path, [""], [np.nan],
            )

        setattr(self, "_outpath", out_path_neat)
        return runtime

    def _list_outputs(self):
        import os.path as op

        return {"out_path_neat": op.abspath(getattr(self, "_outpath"))}
