"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner
"""
import gc
import os
import os.path as op
import pandas as pd
import time
import warnings

import matplotlib
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import yaml
from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    File,
    TraitedSpec,
    traits,
    SimpleInterface,
)

from pynets.core import thresholding, utils
from pynets.statistics.utils import (
    variance_inflation_factor,
    slice_by_corr,
    make_x_y,
)
from pynets.statistics.group.prediction import bootstrapped_nested_cv
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone

matplotlib.use("Agg")
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

    Attributes
    ----------
    out_fmt : str
        Format of output graph.
    in_mat : numpy.ndarray
        Numpy array of connectivity matrix.
    G : networkx.Graph
        Networkx graph of connectivity matrix.

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

    __slots__ = [
        "est_path",
        "prune",
        "norm",
        "out_fmt",
        "in_mat",
        "in_mat_raw",
        "G",
    ]

    def __init__(
        self,
        est_path: str,
        prune: int,
        norm: int,
        out_fmt: str = "gpickle",
        remove_self_loops: bool = True,
    ):
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
            np.array(thresholding.autofix(np.array(np.abs(self.in_mat_raw))))
        )

        # Remove self-loops and ensure symmetry
        if remove_self_loops is True:
            self.in_mat = gu.remove_loops(gu.symmetrize(self.in_mat))
        else:
            self.in_mat = gu.symmetrize(self.in_mat)

        self.in_mat[np.where(np.isnan(self.in_mat) | np.isinf(self.in_mat))] = 0

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
                np.nan_to_num(self.in_mat), method="simple-all"
            )
        # Apply PTR zero-boost
        elif self.norm == 5:
            self.in_mat = gu.ptr.pass_to_ranks(
                np.nan_to_num(self.in_mat), method="zero-boost"
            )
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

        from pynets.statistics.individual.algorithms import (
            defragment,
            most_important,
            prune_small_components,
        )

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
                self.G = prune_small_components(
                    self.G, min_nodes=hardcoded_params["min_nodes"][0]
                )
            except BaseException:
                print(
                    UserWarning(
                        f"Warning: pruning {self.est_path} " f"failed..."
                    )
                )
        elif int(self.prune) == 2:
            try:
                hub_detection_method = hardcoded_params["hub_detection_method"][
                    0
                ]
                print(
                    f"Filtering for hubs on the basis of "
                    f"{hub_detection_method}...\n"
                )
                self.G = most_important(self.G, method=hub_detection_method)[0]
            except FileNotFoundError as e:
                print(e, "Failed to parse advanced.yaml")

        elif int(self.prune) == 3:
            print(
                "Pruning all but the largest connected " "component subgraph..."
            )
            self.G = gu.largest_connected_component(self.G)
        else:
            print("No graph defragmentation applied...")

        self.G = nx.from_numpy_array(self.in_mat)

        if (
            nx.is_empty(self.G) is True
            or (np.abs(self.in_mat) < 0.0000001).all()
            or self.G.number_of_edges() == 0
        ):
            print(
                UserWarning(
                    f"Warning: {self.est_path} " f"empty after pruning!"
                )
            )
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
            pkg_resources.resource_filename(
                "pynets", "statistics/individual/" "global.yaml"
            ),
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
                            else i
                            for i in metric_list_global
                        ]
                    if len(metric_list_global) > 0:
                        print(
                            f"\n\nGlobal Topographic Metrics:"
                            f"\n{metric_list_global_names}\n"
                        )
                    else:
                        print("No global topographic metrics selected!")
                else:
                    print("No global topographic metrics selected!")
                    metric_list_global = []
            except FileNotFoundError as e:
                print(e, "Failed to parse global.yaml")

        with open(
            pkg_resources.resource_filename(
                "pynets", "statistics/individual/" "local.yaml"
            ),
            "r",
        ) as stream:
            try:
                metric_dict_nodal = yaml.load(stream, Loader=yaml.FullLoader)
                metric_list_nodal = metric_dict_nodal["metric_list_nodal"]
                if metric_list_nodal is not None:
                    if len(metric_list_nodal) > 0:
                        print(
                            f"\nNodal Topographic Metrics:"
                            f"\n{metric_list_nodal}\n\n"
                        )
                    else:
                        print("No local topographic metrics selected!")
                else:
                    print("No nodal topographic metrics selected!")
                    metric_list_nodal = []
            except FileNotFoundError as e:
                print(e, "Failed to parse local.yaml")

        if os.path.isfile(self.inputs.est_path):

            cg = CleanGraphs(
                self.inputs.est_path, self.inputs.prune, self.inputs.norm
            )

            tmp_graph_path = None
            if float(self.inputs.prune) >= 1:
                tmp_graph_path = cg.prune_graph()[1]

            if float(self.inputs.norm) >= 1:
                try:
                    cg.normalize_graph()
                except ValueError as e:
                    print(
                        e,
                        f"Graph normalization failed for "
                        f"{self.inputs.est_path}.",
                    )

            if self.inputs.binary is True:
                try:
                    in_mat, G = cg.binarize_graph()
                except ValueError as e:
                    print(
                        e,
                        f"Graph binarization failed for "
                        f"{self.inputs.est_path}.",
                    )
                    in_mat = np.zeros(1, 1)
                    G = nx.Graph()
            else:
                in_mat, G = cg.in_mat, cg.G

            cg.print_summary()

            dir_path = op.dirname(op.realpath(self.inputs.est_path))

            # Deal with empty graphs
            if (
                nx.is_empty(G) is True
                or (np.abs(in_mat) < 0.0000001).all()
                or G.number_of_edges() == 0
                or len(G) < 3
            ):
                out_path_neat = save_netmets(
                    dir_path, self.inputs.est_path, [""], [np.nan]
                )
                print(
                    UserWarning(
                        f"Warning: Empty graph detected for "
                        f"{self.inputs.ID}: "
                        f"{self.inputs.est_path}..."
                    )
                )
            else:
                try:
                    in_mat_len, G_len = cg.create_length_matrix()
                except ValueError as e:
                    print(
                        e,
                        f"Failed to create length matrix for "
                        f"{self.inputs.est_path}.",
                    )
                    G_len = None

                if len(metric_list_global) > 0:
                    # Iteratively run functions from above metric list that
                    # generate single scalar output
                    (
                        net_met_val_list_final,
                        metric_list_names,
                    ) = algorithms.iterate_nx_global_measures(
                        G, metric_list_global
                    )

                    # Run miscellaneous functions that generate multiple
                    # outputs Calculate modularity using the Louvain algorithm
                    if "louvain_modularity" in metric_list_global:
                        try:
                            start_time = time.time()
                            (
                                net_met_val_list_final,
                                metric_list_names,
                                ci,
                            ) = algorithms.get_community(
                                G, net_met_val_list_final, metric_list_names
                            )
                            print(
                                f"{np.round(time.time() - start_time, 3)}"
                                f"{'s'}"
                            )
                        except BaseException:
                            print(
                                "Louvain modularity calculation is undefined "
                                "for G"
                            )
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
                                    "community affiliation vector"
                                )
                            start_time = time.time()
                            (
                                metric_list_names,
                                net_met_val_list_final,
                            ) = algorithms.get_participation(
                                in_mat,
                                ci,
                                metric_list_names,
                                net_met_val_list_final,
                            )
                            print(
                                f"{np.round(time.time() - start_time, 3)}"
                                f"{'s'}"
                            )
                        except BaseException:
                            print(
                                "Participation coefficient cannot be "
                                "calculated for G"
                            )
                            # np.save("%s%s%s" % ('/tmp/partic_coeff_failure',
                            # random.randint(1, 400), '.npy'), in_mat)
                            pass
                    else:
                        if (
                            not ci
                            and "participation_coefficient" in metric_list_nodal
                        ):
                            print(
                                UserWarning(
                                    "Skipping participation coefficient "
                                    "because community affiliation is "
                                    "empty for G..."
                                )
                            )

                    # Diversity Coefficient by louvain community
                    if ci and "diversity_coefficient" in metric_list_nodal:
                        try:
                            if not ci:
                                raise KeyError(
                                    "Diversity coefficient cannot be "
                                    "calculated for G in the absence of a "
                                    "community affiliation vector"
                                )
                            start_time = time.time()
                            (
                                metric_list_names,
                                net_met_val_list_final,
                            ) = algorithms.get_diversity(
                                in_mat,
                                ci,
                                metric_list_names,
                                net_met_val_list_final,
                            )
                            print(
                                f"{np.round(time.time() - start_time, 3)}"
                                f"{'s'}"
                            )
                        except BaseException:
                            print(
                                "Diversity coefficient cannot be calculated "
                                "for G"
                            )
                            # np.save("%s%s%s" % ('/tmp/div_coeff_failure',
                            # random.randint(1, 400), '.npy'), in_mat)
                            pass
                    else:
                        if (
                            not ci
                            and "diversity_coefficient" in metric_list_nodal
                        ):
                            print(
                                UserWarning(
                                    "Skipping diversity coefficient "
                                    "because community affiliation"
                                    " is empty for G..."
                                )
                            )

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
                    if (
                        "betweenness_centrality" in metric_list_nodal
                        and G_len is not None
                    ):
                        try:
                            start_time = time.time()
                            (
                                metric_list_names,
                                net_met_val_list_final,
                            ) = algorithms.get_betweenness_centrality(
                                G_len,
                                metric_list_names,
                                net_met_val_list_final,
                            )
                            print(
                                f"{np.round(time.time() - start_time, 3)}"
                                f"{'s'}"
                            )
                        except BaseException:
                            print(
                                "Betweenness centrality cannot be calculated "
                                "for G"
                            )
                            # np.save("%s%s%s" % ('/tmp/betw_cent_failure',
                            # random.randint(1, 400), '.npy'),
                            #         np.array(nx.to_numpy_matrix(G_len)))
                            pass
                    else:
                        if (
                            G_len is None
                            and "betweenness_centrality" in metric_list_nodal
                        ):
                            print(
                                UserWarning(
                                    "Skipping betweenness centrality "
                                    "because length matrix is empty "
                                    "for G..."
                                )
                            )

                    for i in [
                        "local_efficiency",
                        "local_clustering",
                        "degree_centrality",
                        "eigen_centrality",
                        "communicability_centrality",
                        "rich_club_coefficient",
                    ]:
                        if i in metric_list_nodal:
                            routine = getattr(algorithms, f"get_{i}")
                            try:
                                start_time = time.time()
                                (
                                    metric_list_names,
                                    net_met_val_list_final,
                                ) = routine(
                                    G,
                                    metric_list_names,
                                    net_met_val_list_final,
                                )
                                print(
                                    f"{np.round(time.time() - start_time, 3)}"
                                    f"{'s'}"
                                )
                            except BaseException:
                                print(f"{i} cannot be calculated for G")
                                # np.save("%s%s%s" % (f"/tmp/local_{i}_failure",
                                # random.randint(1, 400), '.npy'),
                                #         np.array(nx.to_numpy_matrix(G)))
                                pass

                if len(metric_list_nodal) > 0 or len(metric_list_global) > 0:
                    out_path_neat = save_netmets(
                        dir_path,
                        self.inputs.est_path,
                        metric_list_names,
                        net_met_val_list_final,
                    )
                    # Cleanup
                    if tmp_graph_path:
                        if os.path.isfile(tmp_graph_path):
                            os.remove(tmp_graph_path)

                    del (
                        net_met_val_list_final,
                        metric_list_names,
                        metric_list_global,
                    )
                    gc.collect()
                else:
                    out_path_neat = save_netmets(
                        dir_path,
                        self.inputs.est_path,
                        [""],
                        [np.nan],
                    )
        else:
            print(
                UserWarning(
                    f"Warning: Empty graph detected for "
                    f"{self.inputs.ID}: "
                    f"{self.inputs.est_path}..."
                )
            )
            dir_path = op.dirname(op.realpath(self.inputs.est_path))
            out_path_neat = save_netmets(
                dir_path,
                self.inputs.est_path,
                [""],
                [np.nan],
            )

        setattr(self, "_outpath", out_path_neat)
        return runtime

    def _list_outputs(self):
        import os.path as op

        return {"out_path_neat": op.abspath(getattr(self, "_outpath"))}


class _MakeDFInputSpec(BaseInterfaceInputSpec):
    """Input Spec for MakeDF."""

    grand_mean_best_estimator = traits.Dict(mandatory=True)
    grand_mean_best_score = traits.Dict(mandatory=True)
    grand_mean_y_predicted = traits.Dict(mandatory=True)
    grand_mean_best_error = traits.Dict(mandatory=True)
    mega_feat_imp_dict = traits.Dict(mandatory=True)
    target_var = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)


class _MakeDFOutputSpec(TraitedSpec):
    """Output Spec for MakeDF."""

    df_summary = traits.Any(mandatory=False)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    target_var = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)


class MakeDF(SimpleInterface):
    """Make a dataframe from the outputs of the Estimate class."""

    input_spec = _MakeDFInputSpec
    output_spec = _MakeDFOutputSpec

    def _run_interface(self, runtime):
        import gc
        from ast import literal_eval
        import pandas as pd
        import numpy as np

        def get_CI(stats, alpha=0.95):
            p = ((1.0 - alpha) / 2.0) * 100
            lower = max(0.0, np.nanpercentile(stats, p))
            p = (alpha + ((1.0 - alpha) / 2.0)) * 100
            upper = min(1.0, np.nanpercentile(stats, p))
            # print('%.1f confidence interval %.1f%% and %.1f%%' % (
            #     alpha * 100, lower * 100, upper * 100))
            return lower, upper

        df_summary = pd.DataFrame(
            columns=[
                "modality",
                "grid",
                "embedding_type",
                "best_estimator",
                "Score",
                "Error",
                "Score_95CI_upper",
                "Score_95CI_lower",
                "Error_95CI_upper",
                "Error_95CI_lower",
                "Score_90CI_upper",
                "Score_90CI_lower",
                "Error_90CI_upper",
                "Error_90CI_lower",
                "target_variable",
                "lp_importance",
                "Predicted_y",
            ]
        )

        df_summary.at[0, "target_variable"] = self.inputs.target_var
        df_summary.at[0, "modality"] = self.inputs.modality
        df_summary.at[0, "embedding_type"] = self.inputs.embedding_type
        df_summary.at[0, "grid"] = tuple(literal_eval(self.inputs.grid_param))

        if (
            bool(self.inputs.grand_mean_best_score) is True
            and len(self.inputs.mega_feat_imp_dict.keys()) > 1
        ):
            y_pred_boots = [
                i
                for i in list(self.inputs.grand_mean_y_predicted.values())
                if not np.isnan(i).all()
            ]
            if len(y_pred_boots) > 0:
                max_row_len = max([len(ll) for ll in y_pred_boots])
                y_pred_vals = np.nanmean(
                    [
                        [el for el in row]
                        + [np.NaN] * max(0, max_row_len - len(row))
                        for row in y_pred_boots
                    ],
                    axis=0,
                )
            else:
                y_pred_vals = np.nan
        else:
            y_pred_vals = np.nan

        if bool(self.inputs.grand_mean_best_estimator) is True:
            df_summary.at[0, "best_estimator"] = list(
                self.inputs.grand_mean_best_estimator.values()
            )
            df_summary.at[0, "Score"] = list(
                self.inputs.grand_mean_best_score.values()
            )
            df_summary.at[0, "Predicted_y"] = y_pred_vals
            df_summary.at[0, "Error"] = list(
                self.inputs.grand_mean_best_error.values()
            )
            df_summary.at[0, "Score_95CI_upper"] = get_CI(
                list(self.inputs.grand_mean_best_score.values()), alpha=0.95
            )[1]
            df_summary.at[0, "Score_95CI_lower"] = get_CI(
                list(self.inputs.grand_mean_best_score.values()), alpha=0.95
            )[0]
            df_summary.at[0, "Score_90CI_upper"] = get_CI(
                list(self.inputs.grand_mean_best_score.values()), alpha=0.90
            )[1]
            df_summary.at[0, "Score_90CI_lower"] = get_CI(
                list(self.inputs.grand_mean_best_score.values()), alpha=0.90
            )[0]
            df_summary.at[0, "Error_95CI_upper"] = get_CI(
                list(self.inputs.grand_mean_best_error.values()), alpha=0.95
            )[1]
            df_summary.at[0, "Error_95CI_lower"] = get_CI(
                list(self.inputs.grand_mean_best_error.values()), alpha=0.95
            )[0]
            df_summary.at[0, "Error_90CI_upper"] = get_CI(
                list(self.inputs.grand_mean_best_error.values()), alpha=0.90
            )[1]
            df_summary.at[0, "Error_90CI_lower"] = get_CI(
                list(self.inputs.grand_mean_best_error.values()), alpha=0.90
            )[0]
            df_summary.at[0, "lp_importance"] = np.array(
                list(self.inputs.mega_feat_imp_dict.keys())
            )
        else:
            df_summary.at[0, "best_estimator"] = np.nan
            df_summary.at[0, "Score"] = np.nan
            df_summary.at[0, "Predicted_y"] = np.nan
            df_summary.at[0, "Error"] = np.nan
            df_summary.at[0, "Score_95CI_upper"] = np.nan
            df_summary.at[0, "Score_95CI_lower"] = np.nan
            df_summary.at[0, "Score_90CI_upper"] = np.nan
            df_summary.at[0, "Score_90CI_lower"] = np.nan
            df_summary.at[0, "Error_95CI_upper"] = np.nan
            df_summary.at[0, "Error_95CI_lower"] = np.nan
            df_summary.at[0, "Error_90CI_upper"] = np.nan
            df_summary.at[0, "Error_90CI_lower"] = np.nan
            df_summary.at[0, "lp_importance"] = np.nan

        out_df_summary = (
            f"{runtime.cwd}/df_summary_"
            f"{self.inputs.target_var}_"
            f"{self.inputs.modality}_"
            f"{self.inputs.embedding_type}_"
            f"{self.inputs.grid_param.replace(', ','_')}.csv"
        )

        df_summary.to_csv(out_df_summary, index=False)
        print(f"Writing dataframe to file {out_df_summary}...")
        self._results["df_summary"] = out_df_summary
        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["modality"] = self.inputs.modality
        self._results["grid_param"] = self.inputs.grid_param
        gc.collect()

        return runtime


class ReduceVIF(BaseEstimator, TransformerMixin):
    """Reduce Variable Inflation Factor (VIF)"""

    def __init__(self, thresh=10.0, nthreads=4, r_min=0, obs=250):
        self.thresh = thresh
        self.nthreads = nthreads
        self.r_min = r_min
        self.obs = obs

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : DataFrame
            DataFrame to be reduced
        """
        self.X = X
        return self

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : DataFrame
            DataFrame to be reduced

        Returns
        -------
        X:  DataFrame
            Transformed DataFrame
        vif_cols: list
            List of columns with VIF greater than threshold
        """
        return self.calculate_vif(
            X, self.thresh, self.nthreads, self.r_min, self.obs
        )

    @staticmethod
    def calculate_vif(X, thresh=10.0, nthreads=16, r_min=0, obs=250):
        dropped = True
        vif_cols = []
        X_vif_candidates = slice_by_corr(X, r_min)
        X_vif_candidates = X_vif_candidates.sample(n=obs)
        while dropped:
            variables = X_vif_candidates.columns
            dropped = False
            with Parallel(n_jobs=nthreads, backend="threading") as parallel:
                vif = parallel(
                    delayed(variance_inflation_factor)(
                        np.asarray(X_vif_candidates[variables].values),
                        X_vif_candidates.columns.get_loc(var),
                    )
                    for var in X_vif_candidates.columns
                )
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(
                    f"Dropping {X_vif_candidates.columns[maxloc]} with vif={max_vif}"
                )
                vif_cols.append(X_vif_candidates.columns.tolist()[maxloc])
                X_vif_candidates = X_vif_candidates.drop(
                    [X_vif_candidates.columns.tolist()[maxloc]], axis=1
                )
                dropped = True

        if len(vif_cols) > 0:
            return X.drop(columns=vif_cols), vif_cols
        else:
            return X, vif_cols


class Razors(object):
    """
    Razors is a callable refit option for `GridSearchCV` whose aim is to
    balance model complexity and cross-validated score in the spirit of the
    "one standard error" rule of Breiman et al. (1984), which showed that
    the tuning hyperparameter associated with the best performing model may be
    prone to overfit. To help mitigate this risk, we can instead instruct
    gridsearch to refit the highest performing 'parsimonious' model, as defined
    using simple statistical rules (e.g. standard error (`sigma`),
    percentile (`eta`), or significance level (`alpha`)) to compare
    distributions of model performance across folds. Importantly, this
    strategy assumes that the grid of multiple cross-validated models
    can be principly ordered from simplest to most complex with respect to some
    target hyperparameter of interest. To use the razors suite, supply
    the `simplify` function partial of the `Razors` class as a callable
    directly to the `refit` argument of `GridSearchCV`.

    Parameters
    ----------
    cv_results : dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.
    scoring : str
        Refit scoring metric.
    param : str
        Parameter whose complexity will be optimized.
    rule : str
        Rule for balancing model complexity with performance.
        Options are 'se', 'percentile', and 'ranksum'. Default is 'se'.
    sigma : int
        Number of standard errors tolerance in the case that a standard error
        threshold is used to filter outlying scores across folds. Required if
        `rule`=='se'. Default is 1.
    eta : float
        Percentile tolerance in the case that a percentile threshold
        is used to filter outlier scores across folds. Required if
        `rule`=='percentile'. Default is 0.68.
    alpha : float
        An alpha significance level in the case that wilcoxon rank sum
        hypothesis testing is used to filter outlying scores across folds.
        Required if `rule`=='ranksum'. Default is 0.05.

    References
    ----------
    Breiman, Friedman, Olshen, and Stone. (1984) Classification and Regression
    Trees. Wadsworth.

    Notes
    -----
    Here, 'simplest' is defined by the complexity of the model as influenced by
    some user-defined target parameter (e.g. number of components, number of
    estimators, polynomial degree, cost, scale, number hidden units, weight
    decay, number of nearest neighbors, L1/L2 penalty, etc.).

    The callable API accordingly assumes that the `params` attribute of
    `cv_results_` 1) contains the indicated hyperparameter (`param`) of
    interest, and 2) contains a sequence of values (numeric, boolean, or
    categorical) that are ordered from least to most complex.
    """

    __slots__ = (
        "cv_results",
        "param",
        "param_complexity",
        "scoring",
        "rule",
        "greater_is_better",
        "_scoring_funcs",
        "_scoring_dict",
        "_n_folds",
        "_splits",
        "_score_grid",
        "_cv_means",
        "_sigma",
        "_eta",
        "_alpha",
    )

    def __init__(
        self,
        cv_results_,
        param,
        scoring,
        rule,
        sigma=1,
        eta=0.95,
        alpha=0.01,
    ):
        import sklearn.metrics

        self.cv_results = cv_results_
        self.param = param
        self.scoring = scoring
        self.rule = rule
        self._scoring_funcs = [
            met
            for met in sklearn.metrics.__all__
            if (met.endswith("_score")) or (met.endswith("_error"))
        ]
        # Set _score metrics to True and _error metrics to False
        self._scoring_dict = dict(
            zip(
                self._scoring_funcs,
                [met.endswith("_score") for met in self._scoring_funcs],
            )
        )
        self.greater_is_better = self._check_scorer()
        self._n_folds = len(
            list(
                set(
                    [
                        i.split("_")[0]
                        for i in list(self.cv_results.keys())
                        if i.startswith("split")
                    ]
                )
            )
        )
        # Extract subgrid corresponding to the scoring metric of interest
        self._splits = [
            i
            for i in list(self.cv_results.keys())
            if i.endswith(f"test_{self.scoring}") and i.startswith("split")
        ]
        self._score_grid = np.vstack(
            [self.cv_results[cv] for cv in self._splits]
        ).T
        self._cv_means = np.array(np.nanmean(self._score_grid, axis=1))
        self._sigma = sigma
        self._eta = eta
        self._alpha = alpha

    def _check_scorer(self):
        """
        Check whether the target refit scorer is negated. If so, adjust
        greater_is_better accordingly.
        """

        if (
            self.scoring not in self._scoring_dict.keys()
            and f"{self.scoring}_score" not in self._scoring_dict.keys()
        ):
            if self.scoring.startswith("neg_"):
                self.greater_is_better = True
            else:
                raise NotImplementedError(
                    f"Scoring metric {self.scoring} not " f"recognized."
                )
        else:
            self.greater_is_better = [
                value
                for key, value in self._scoring_dict.items()
                if self.scoring in key
            ][0]
        return self.greater_is_better

    def _best_low_complexity(self):
        """
        Balance model complexity with cross-validated score.

        Return
        ------
        int
            Index of a model that has the lowest complexity but its test score
            is the highest on average across folds as compared to other models
            that are equally likely to occur.
        """

        # Check parameter(s) whose complexity we seek to restrict
        if not any(
            self.param in x for x in self.cv_results["params"][0].keys()
        ):
            raise KeyError(f"Parameter {self.param} not found in cv grid.")
        else:
            hyperparam = [
                i
                for i in self.cv_results["params"][0].keys()
                if i.endswith(self.param)
            ][0]

        # Select low complexity threshold based on specified evaluation rule
        if self.rule == "se":
            if not self._sigma:
                raise ValueError(
                    "For `se` rule, the tolerance "
                    "(i.e. `_sigma`) parameter cannot be null."
                )
            l_cutoff, h_cutoff = self.call_standard_error()
        elif self.rule == "percentile":
            if not self._eta:
                raise ValueError(
                    "For `percentile` rule, the tolerance "
                    "(i.e. `_eta`) parameter cannot be null."
                )
            l_cutoff, h_cutoff = self.call_percentile()
        elif self.rule == "ranksum":
            if not self._alpha:
                raise ValueError(
                    "For `ranksum` rule, the alpha-level "
                    "(i.e. `_alpha`) parameter cannot be null."
                )
            l_cutoff, h_cutoff = self.call_rank_sum_test()
        else:
            raise NotImplementedError(
                f"{self.rule} is not a valid " f"rule of RazorCV."
            )

        self.cv_results[f"param_{hyperparam}"].mask = np.where(
            (self._cv_means >= float(l_cutoff))
            & (self._cv_means <= float(h_cutoff)),
            True,
            False,
        )

        if np.sum(self.cv_results[f"param_{hyperparam}"].mask) == 0:
            print(f"\nLow: {l_cutoff}")
            print(f"High: {h_cutoff}")
            print(f"{self._cv_means}")
            print(f"hyperparam: {hyperparam}\n")
            raise ValueError(
                "No valid grid columns remain within the "
                "boundaries of the specified razor"
            )

        highest_surviving_rank = np.nanmin(
            self.cv_results[f"rank_test_{self.scoring}"][
                self.cv_results[f"param_{hyperparam}"].mask
            ]
        )

        # print(f"Highest surviving rank: {highest_surviving_rank}\n")

        return np.flatnonzero(
            np.isin(
                self.cv_results[f"rank_test_{self.scoring}"],
                highest_surviving_rank,
            )
        )[0]

    def call_standard_error(self):
        """
        Returns the simplest model whose performance is within `sigma`
        standard errors of the average highest performing model.
        """

        # Estimate the standard error across folds for each column of the grid
        cv_se = np.array(
            np.nanstd(self._score_grid, axis=1) / np.sqrt(self._n_folds)
        )

        # Determine confidence interval
        if self.greater_is_better:
            best_score_idx = np.nanargmax(self._cv_means)
            h_cutoff = self._cv_means[best_score_idx] + cv_se[best_score_idx]
            l_cutoff = self._cv_means[best_score_idx] - cv_se[best_score_idx]
        else:
            best_score_idx = np.nanargmin(self._cv_means)
            h_cutoff = self._cv_means[best_score_idx] - cv_se[best_score_idx]
            l_cutoff = self._cv_means[best_score_idx] + cv_se[best_score_idx]

        return l_cutoff, h_cutoff

    def call_rank_sum_test(self):
        """
        Returns the simplest model whose paired performance across folds is
        insignificantly different from the average highest performing,
        at a predefined `alpha` level of significance.
        """

        from scipy.stats import wilcoxon
        import itertools

        if self.greater_is_better:
            best_score_idx = np.nanargmax(self._cv_means)
        else:
            best_score_idx = np.nanargmin(self._cv_means)

        # Perform signed Wilcoxon rank sum test for each pair combination of
        # columns against the best average score column
        tests = [
            pair
            for pair in list(
                itertools.combinations(range(self._score_grid.shape[0]), 2)
            )
            if best_score_idx in pair
        ]

        p_dict = {}
        for i, test in enumerate(tests):
            p_dict[i] = wilcoxon(
                self._score_grid[test[0], :], self._score_grid[test[1], :]
            )[1]

        # Sort and prune away significant tests
        p_dict = {
            k: v
            for k, v in sorted(p_dict.items(), key=lambda item: item[1])
            if v > self._alpha
        }

        # Flatten list of tuples, remove best score index, and take the
        # lowest and highest remaining bounds
        tests = [
            j
            for j in list(
                set(list(sum([tests[i] for i in list(p_dict.keys())], ())))
            )
            if j != best_score_idx
        ]
        if self.greater_is_better:
            h_cutoff = self._cv_means[
                np.nanargmin(
                    self.cv_results[f"rank_test_{self.scoring}"][tests]
                )
            ]
            l_cutoff = self._cv_means[
                np.nanargmax(
                    self.cv_results[f"rank_test_{self.scoring}"][tests]
                )
            ]
        else:
            h_cutoff = self._cv_means[
                np.nanargmax(
                    self.cv_results[f"rank_test_{self.scoring}"][tests]
                )
            ]
            l_cutoff = self._cv_means[
                np.nanargmin(
                    self.cv_results[f"rank_test_{self.scoring}"][tests]
                )
            ]

        return l_cutoff, h_cutoff

    def call_percentile(self):
        """
        Returns the simplest model whose performance is within the `eta`
        percentile of the average highest performing model.
        """

        # Estimate the indicated percentile, and its inverse, across folds for
        # each column of the grid
        perc_cutoff = np.nanpercentile(
            self._score_grid, [100 * self._eta, 100 - 100 * self._eta], axis=1
        )

        # Determine bounds of the percentile interval
        if self.greater_is_better:
            best_score_idx = np.nanargmax(self._cv_means)
            h_cutoff = perc_cutoff[0, best_score_idx]
            l_cutoff = perc_cutoff[1, best_score_idx]
        else:
            best_score_idx = np.nanargmin(self._cv_means)
            h_cutoff = perc_cutoff[0, best_score_idx]
            l_cutoff = perc_cutoff[1, best_score_idx]

        return l_cutoff, h_cutoff

    @staticmethod
    def simplify(param, scoring, rule="se", sigma=1, eta=0.68, alpha=0.01):
        """
        Callable to be run as `refit` argument of `GridsearchCV`.

        Parameters
        ----------
        param : str
            Parameter with the largest influence on model complexity.
        scoring : str
            Refit scoring metric.
        sigma : int
            Number of standard errors tolerance in the case that a standard
            error threshold is used to filter outlying scores across folds.
            Only applicable if `rule`=='se'. Default is 1.
        eta : float
            Acceptable percent tolerance in the case that a percentile
            threshold is used. Only applicable if `rule`=='percentile'.
            Default is 0.68.
        alpha : float
            Alpha-level to use for signed wilcoxon rank sum testing.
            Only applicable if `rule`=='ranksum'. Default is 0.01.
        """
        from functools import partial

        def razor_pass(cv_results_, param, scoring, rule, sigma, alpha, eta):
            rcv = Razors(
                cv_results_,
                param,
                scoring,
                rule=rule,
                sigma=sigma,
                alpha=alpha,
                eta=eta,
            )
            return rcv._best_low_complexity()

        return partial(
            razor_pass,
            param=param,
            scoring=scoring,
            rule=rule,
            sigma=sigma,
            alpha=alpha,
            eta=eta,
        )


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for Scikit-learn estimators.
    """

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def fit(self, X, y=None):
        from sklearn.utils.validation import check_X_y
        from sklearn.utils.multiclass import unique_labels
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        X_out, y = check_X_y(X, y)

        X = X if hasattr(X, "columns") else X_out

        if hasattr(X, "columns"):
            self.expected_ = list(X.columns)
            self.expected_n_ = X.shape[1]
        else:
            self.expected_ = None
            self.expected_n_ = X.shape[1]
            warnings.warn(
                "Input does not have a columns attribute, "
                "only number of columns will be validated"
            )

        self.classes_ = unique_labels(y)

        self.ensemble_fitted = self.ensemble.fit(X, y)

        return self

    def predict(self, X):
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = check_array(X)

        return self.ensemble_fitted.predict(X)


class _CopyJsonDictInputSpec(BaseInterfaceInputSpec):
    feature_spaces = traits.Any(mandatory=True)
    modality = traits.Any(mandatory=True)
    embedding_type = traits.Any(mandatory=True)
    target_var = traits.Any(mandatory=True)


class _CopyJsonDictOutputSpec(TraitedSpec):
    json_dict = traits.Any(mandatory=True)
    modality = traits.Any(mandatory=True)
    embedding_type = traits.Any(mandatory=True)
    target_var = traits.Any(mandatory=True)


class CopyJsonDict(SimpleInterface):
    """Interface wrapper for CopyJsonDict"""

    input_spec = _CopyJsonDictInputSpec
    output_spec = _CopyJsonDictOutputSpec

    def _run_interface(self, runtime):
        import uuid
        import os

        # import time
        # import random
        from time import strftime

        run_uuid = f"{strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
        from nipype.utils.filemanip import fname_presuffix, copyfile

        # time.sleep(random.randint(1, 30))
        if (
            self.inputs.feature_spaces is not None
            and self.inputs.modality is not None
            and self.inputs.embedding_type is not None
            and self.inputs.target_var is not None
        ):
            input_dict_tmp = self.inputs.feature_spaces[
                f"{self.inputs.modality}_{self.inputs.embedding_type}"
            ]
            json_dict = fname_presuffix(
                input_dict_tmp,
                suffix=f"_{run_uuid}_{self.inputs.modality}_"
                f"{self.inputs.embedding_type}_"
                f"{self.inputs.target_var}.json",
                newpath=runtime.cwd,
            )
            copyfile(input_dict_tmp, json_dict, use_hardlink=False)
        else:
            json_dict = (
                f"{runtime.cwd}/{run_uuid}_{self.inputs.modality}_"
                f"{self.inputs.embedding_type}_"
                f"{self.inputs.target_var}.json"
            )
            os.mknod(json_dict)

        # time.sleep(random.randint(1, 30))

        self._results["json_dict"] = json_dict
        self._results["modality"] = self.inputs.modality
        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["target_var"] = self.inputs.target_var

        return runtime


class _MakeXYInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper MakeXY"""

    json_dict = traits.Any(mandatory=True)
    target_var = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)


class _MakeXYOutputSpec(TraitedSpec):
    """Output interface wrapper MakeXY"""

    X = traits.Any(mandatory=False)
    Y = traits.Any(mandatory=False)
    embedding_type = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)
    target_var = traits.Str(mandatory=True)


class MakeXY(SimpleInterface):
    """Interface wrapper for MakeXY"""

    input_spec = _MakeXYInputSpec
    output_spec = _MakeXYOutputSpec

    def _run_interface(self, runtime):
        import os
        from ast import literal_eval

        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["grid_param"] = self.inputs.grid_param
        self._results["modality"] = self.inputs.modality
        self._results["target_var"] = self.inputs.target_var

        def prefix_df_columns(df, cols, prefix):
            new_names = [(i, f"{prefix}_{i}") for i in df[cols].columns.values]
            df.rename(columns=dict(new_names), inplace=True)
            return df

        if self.inputs.json_dict is not None:
            if (
                os.path.isfile(self.inputs.json_dict)
                and self.inputs.json_dict.endswith(".json")
                and os.stat(self.inputs.json_dict).st_size != 0
            ):
                drop_cols = [self.inputs.target_var, "id", "participant_id"]

                [X, Y] = make_x_y(
                    self.inputs.json_dict,
                    drop_cols,
                    self.inputs.target_var,
                    self.inputs.embedding_type,
                    tuple(literal_eval(self.inputs.grid_param)),
                )

                numeric_cols = [col for col in X if col[0].isdigit()]
                if len(numeric_cols) > 0:
                    X = prefix_df_columns(X, numeric_cols, self.inputs.modality)

                if X is not None:
                    self._results["X"] = X
                    self._results["Y"] = Y
                else:
                    self._results["X"] = None
                    self._results["Y"] = None
            else:
                self._results["X"] = None
                self._results["Y"] = None
        else:
            self._results["X"] = None
            self._results["Y"] = None

        return runtime


class _BSNestedCVInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for BSNestedCV"""

    X = traits.Any(mandatory=False)
    y = traits.Any(mandatory=False)
    target_var = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)
    n_boots = traits.Int()
    dummy_run = traits.Bool()
    search_method = traits.Str(mandatory=True)
    stack = traits.Bool()
    stack_prefix_list = traits.Any([], mandatory=True, usedefault=True)
    nuisance_cols = traits.Any()
    predict_type = traits.Str("classifier", mandatory=True, usedefault=True)


class _BSNestedCVOutputSpec(TraitedSpec):
    """Output interface wrapper for BSNestedCV"""

    grand_mean_best_estimator = traits.Dict(
        {0: "None"}, mandatory=True, usedefault=True
    )
    grand_mean_best_score = traits.Dict(
        {0: np.nan}, mandatory=True, usedefault=True
    )
    grand_mean_y_predicted = traits.Dict(
        {0: np.nan}, mandatory=True, usedefault=True
    )
    grand_mean_best_error = traits.Dict(
        {0: np.nan}, mandatory=True, usedefault=True
    )
    mega_feat_imp_dict = traits.Dict(
        {0: "None"}, mandatory=True, usedefault=True
    )
    target_var = traits.Str(mandatory=True)
    modality = traits.Str(mandatory=True)
    embedding_type = traits.Str(mandatory=True)
    grid_param = traits.Str(mandatory=True)
    out_path_est = traits.Any()


class BSNestedCV(SimpleInterface):
    """Interface wrapper for BSNestedCV"""

    input_spec = _BSNestedCVInputSpec
    output_spec = _BSNestedCVOutputSpec

    def _run_interface(self, runtime):
        import gc
        from colorama import Fore, Style
        from joblib import dump

        self._results["target_var"] = self.inputs.target_var
        self._results["modality"] = self.inputs.modality
        self._results["embedding_type"] = self.inputs.embedding_type
        self._results["grid_param"] = self.inputs.grid_param

        if self.inputs.X is None:
            return runtime

        if not self.inputs.X.empty and not np.isnan(self.inputs.y).all():
            if isinstance(self.inputs.X, pd.DataFrame):
                [
                    grand_mean_best_estimator,
                    grand_mean_best_score,
                    grand_mean_best_error,
                    mega_feat_imp_dict,
                    grand_mean_y_predicted,
                    final_est,
                ] = bootstrapped_nested_cv(
                    self.inputs.X,
                    self.inputs.y,
                    nuisance_cols=self.inputs.nuisance_cols,
                    predict_type=self.inputs.predict_type,
                    n_boots=self.inputs.n_boots,
                    dummy_run=self.inputs.dummy_run,
                    search_method=self.inputs.search_method,
                    stack=self.inputs.stack,
                    stack_prefix_list=self.inputs.stack_prefix_list,
                )
                if final_est:
                    grid_param_name = self.inputs.grid_param.replace(", ", "_")
                    out_path_est = (
                        f"{runtime.cwd}/estimator_"
                        f"{self.inputs.target_var}_"
                        f"{self.inputs.modality}_"
                        f"{self.inputs.embedding_type}_"
                        f"{grid_param_name}.joblib"
                    )

                    dump(final_est, out_path_est)
                else:
                    out_path_est = None

                if len(grand_mean_best_estimator.keys()) > 1:
                    print(
                        f"\n\n{Fore.BLUE}Target Outcome: "
                        f"{Fore.GREEN}{self.inputs.target_var}"
                        f"{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Embedding type: "
                        f"{Fore.RED}{self.inputs.embedding_type}"
                        f"{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Grid Params: "
                        f"{Fore.RED}{self.inputs.grid_param}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Best Estimator: "
                        f"{Fore.RED}{grand_mean_best_estimator}"
                        f"{Style.RESET_ALL}"
                    )
                    print(
                        f"\n{Fore.BLUE}Variance: "
                        f"{Fore.RED}{grand_mean_best_score}{Style.RESET_ALL}"
                    )
                    print(
                        f"{Fore.BLUE}Error: "
                        f"{Fore.RED}{grand_mean_best_error}{Style.RESET_ALL}\n"
                    )
                    # print(f"y_actual: {self.inputs.y}")
                    # print(f"y_predicted: {grand_mean_y_predicted}")
                    if self.inputs.stack is False:
                        print(
                            f"{Fore.BLUE}Feature Importance: "
                            f"{Fore.RED}{list(mega_feat_imp_dict.keys())}"
                            f"{Style.RESET_ALL} "
                            f"with {Fore.RED}{len(mega_feat_imp_dict.keys())} "
                            f"features...{Style.RESET_ALL}\n\n"
                        )
                        print(
                            f"{Fore.BLUE}Modality: "
                            f"{Fore.RED}{self.inputs.modality}"
                            f"{Style.RESET_ALL}"
                        )
                else:
                    print(
                        f"{Fore.RED}Empty feature-space for "
                        f"{self.inputs.grid_param}, "
                        f"{self.inputs.target_var}, "
                        f"{self.inputs.embedding_type}, "
                        f"{self.inputs.modality}{Style.RESET_ALL}"
                    )
                    grand_mean_best_estimator = {0: "None"}
                    grand_mean_best_score = {0: np.nan}
                    grand_mean_y_predicted = {0: np.nan}
                    grand_mean_best_error = {0: np.nan}
                    mega_feat_imp_dict = {0: "None"}
            else:
                print(
                    f"{Fore.RED}{self.inputs.X} is not pd.DataFrame for"
                    f" {self.inputs.grid_param}, {self.inputs.target_var},"
                    f" {self.inputs.embedding_type}, "
                    f"{self.inputs.modality}{Style.RESET_ALL}"
                )
                grand_mean_best_estimator = {0: "None"}
                grand_mean_best_score = {0: np.nan}
                grand_mean_y_predicted = {0: np.nan}
                grand_mean_best_error = {0: np.nan}
                mega_feat_imp_dict = {0: "None"}
                out_path_est = None
        else:
            print(
                f"{Fore.RED}Empty feature-space for {self.inputs.grid_param},"
                f" {self.inputs.target_var}, {self.inputs.embedding_type},"
                f" {self.inputs.modality}{Style.RESET_ALL}"
            )
            grand_mean_best_estimator = {0: "None"}
            grand_mean_best_score = {0: np.nan}
            grand_mean_y_predicted = {0: np.nan}
            grand_mean_best_error = {0: np.nan}
            mega_feat_imp_dict = {0: "None"}
            out_path_est = None
        gc.collect()

        self._results["grand_mean_best_estimator"] = grand_mean_best_estimator
        self._results["grand_mean_best_score"] = grand_mean_best_score
        self._results["grand_mean_y_predicted"] = grand_mean_y_predicted
        self._results["grand_mean_best_error"] = grand_mean_best_error
        self._results["mega_feat_imp_dict"] = mega_feat_imp_dict
        self._results["out_path_est"] = out_path_est

        return runtime


class DeConfounder(BaseEstimator, TransformerMixin):
    """A transformer removing the effect of y on X using
    sklearn.linear_model.LinearRegression.

    References
    ----------
    D. Chyzhyk, G. Varoquaux, B. Thirion and M. Milham,
        "Controlling a confound in predictive models with a test set minimizing
        its effect," 2018 International Workshop on Pattern Recognition in
        Neuroimaging (PRNI), Singapore, 2018,
        pp. 1-4. doi: 10.1109/PRNI.2018.8423961
    """

    from sklearn.linear_model import LinearRegression

    def __init__(self, confound_model=LinearRegression()):
        self.confound_model = confound_model

    def fit(self, X, z):
        if z.ndim == 1:
            z = z[:, np.newaxis]
        confound_model = clone(self.confound_model)
        confound_model.fit(z, X)
        self.confound_model_ = confound_model

        return self

    def transform(self, X, z):
        if z.ndim == 1:
            z = z[:, np.newaxis]
        X_confounds = self.confound_model_.predict(z)
        return X - X_confounds
