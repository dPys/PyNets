import pandas as pd
import os
import re
import glob
import numpy as np
import pickle
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
import itertools
import warnings
import psutil
warnings.simplefilter("ignore")
from collections import OrderedDict
from pynets.stats.benchmarking import graph_theory_prep, build_grid, \
    create_feature_space, get_coords_labels
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    traits,
    SimpleInterface,
)

import_list = ["import pandas as pd",
               "import os",
               "import re",
               "import glob",
               "import numpy as np",
               "from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_validate",
               "from sklearn.dummy import DummyRegressor",
               "from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression",
               "from sklearn.pipeline import Pipeline",
               "from sklearn.impute import SimpleImputer",
               "from sklearn.preprocessing import StandardScaler",
               "from sklearn import linear_model, decomposition",
               "from pynets.stats.benchmarking import build_hp_dict",
               "import seaborn as sns",
               "import matplotlib.pyplot as plt",
               "from sklearn.base import BaseEstimator, TransformerMixin",
               "from pynets.stats.embeddings import build_asetomes, _omni_embed",
               "from joblib import Parallel, delayed",
               "from pynets.core import utils",
               "from itertools import groupby",
               "import shutil",
               "from pathlib import Path",
               "from collections import OrderedDict",
               "from operator import itemgetter",
               "from statsmodels.stats.outliers_influence import variance_inflation_factor"]

#target_vars = ['rum_persist', 'dep_1', 'age']
target_vars = ['rum_persist']
thr_type = 'MST'
drop_cols = ['rum_persist', 'dep_1', 'age', 'sex']
#embedding_types = ['OMNI', 'ASE']
embedding_types = ['OMNI']
modalities = ['func', 'dwi']
template = 'MNI152_T1'
mets = ["global_efficiency", "average_clustering", "average_shortest_path_length", "average_betweenness_centrality",
        "average_eigenvector_centrality", "average_degree_centrality", "average_diversity_coefficient",
        "average_participation_coefficient"]
df = pd.read_csv('/Users/derekpisner/Documents/Dissertation/Analysis/TuningSet/df_rum_persist_all.csv', index_col=False)
hyperparams_func = ["rsn", "res", "model", 'hpass', 'extract', 'smooth']
hyperparams_dwi = ["rsn", "res", "model", 'directget', 'minlength']
base_dir = '/Users/derekpisner/Downloads'


def get_ensembles_ase(modality, alg, base_dir):
    ensembles = list(set([os.path.basename(i).split(alg + '_')[
                              1].split('_all_nodes_rawgraph')[
                              0] + '_' +
                          os.path.basename(i).split(modality + '_')[
                              1].replace('.npy', '') for i in
                          glob.glob(
                              f"{base_dir}/embeddings_all_{modality}/*/*/*{alg}*.npy")]))
    return ensembles


def get_ensembles_omni(modality, alg, base_dir):
    ensembles = list(set(['rsn-' +
                          os.path.basename(i).split(alg + '_')[
                              1].split('_')[1] + '_res-' +
                          os.path.basename(i).split(alg + '_')[
                              1].split('_')[0] + '_' +
                          os.path.basename(i).split(modality + '_')[
                              1].replace('.npy', '') for i in
                          glob.glob(
                              f"{base_dir}/embeddings_all_{modality}/*/*/*{alg}*.npy")]))
    return ensembles


def get_ensembles_top(modality, thr_type):
    df_top = pd.read_csv(
        f"{base_dir}/all_subs_neat_{modality}.csv")
    df_top = df_top.dropna(subset=["id"])
    df_top['id'] = df_top['id'].str.replace('topology_auc_sub-', '')
    df_top = df_top.rename(
        columns=lambda x: re.sub("partcorr", "model-partcorr", x))
    df_top = df_top.rename(
        columns=lambda x: re.sub("_corr", "_model-corr", x))
    df_top = df_top.rename(
        columns=lambda x: re.sub("_cov", "_model-cov", x))
    df_top['participant_id'] = df_top['id'].str.replace("_ses-ses-",
                                                        "_ses-")
    df_top['participant_id'] = df_top['participant_id'].str.replace(
        ".csv", "")
    [df_top, ensembles] = graph_theory_prep(df_top, thr_type)
    ensembles = [i for i in ensembles if
                 i != 'id' and i != 'participant_id']
    return ensembles, df_top


def make_feature_space_dict(df, subject_dict, ses, base_dir):
    ml_dfs = {}
    for modality in modalities:
        print(modality)
        if modality not in ml_dfs.keys():
            ml_dfs[modality] = {}
        for alg in embedding_types:
            if alg not in ml_dfs[modality].keys():
                ml_dfs[modality][alg] = {}
            for grid_param in modality_grids[modality]:
                print(grid_param)
                # Skip any invalid combinations (e.g. the covariance of the
                # variance is singular...)
                if 'cov' in grid_param and 'variance' in grid_param:
                    continue

                # save feature space to dict
                ml_dfs[modality][alg][grid_param] = create_feature_space(df, grid_param, subject_dict, ses, modality, alg)

    dict_file_path = f"{base_dir}/pynets_ml_dict.pkl"
    with open(dict_file_path, 'wb') as f:
        pickle.dump(ml_dfs, f, protocol=2)
    f.close()

    return dict_file_path


subject_dict = {}
modality_grids = {}
for modality in modalities:
    hyperparams = eval(f"hyperparams_{modality}")
    ids = [os.path.basename(i) + '_ses-1' for i in glob.glob(
        f"{base_dir}/embeddings_all_{modality}/*") if
           os.path.basename(i).startswith('sub')]

    for alg in embedding_types:
        if alg == 'ASE':
            ensembles = get_ensembles_ase(modality, alg, base_dir)
        elif alg == 'OMNI':
            ensembles = get_ensembles_omni(modality, alg, base_dir)
        elif alg == 'topology':
            ensembles, df_top = get_ensembles_top(modality, thr_type)

        hyperparam_dict = {}

        [hyperparam_dict, grid] = build_grid(modality, hyperparam_dict,
                                             sorted(list(set(hyperparams))),
                                             ensembles)

        # Since we are using all of the 3 RSN connectomes (pDMN, coSN, and fECN) in the feature-space,
        # rather than varying them as hyperparameters (i.e. we assume they each add distinct variance
        # from one another) Create an abridged grid, where
        if modality == 'func':
            modality_grids[modality] = list(
                set([i[:-2] + tuple(i[-1]) for i in grid]))
        else:
            modality_grids[modality] = list(set([i[:-1] for i in grid]))

        for id in ids:
            ID = id.split("_")[0].split("sub-")[1]
            ses = id.split("_")[1].split("ses-")[1]

            if ID not in subject_dict.keys():
                subject_dict[ID] = {}

            if ses not in subject_dict[ID].keys():
                subject_dict[ID][ses] = {}

            if modality not in subject_dict[ID][ses].keys():
                subject_dict[ID][ses][modality] = {}

            subject_dict[ID][ses][modality] = dict.fromkeys(grid, np.nan)

            # Functional case
            if modality == 'func':
                for comb in grid:
                    extract, hpass, model, res, atlas, smooth = comb
                    comb_tuple = (atlas, extract, hpass, model, res, smooth)
                    subject_dict[ID][ses][modality][comb_tuple] = {}
                    if alg == 'ASE' or alg == 'OMNI':
                        if alg == 'ASE':
                            embedding = f"{base_dir}/embeddings_all" \
                                        f"_{modality}/sub-{ID}/rsn-{atlas}_res-{res}/" \
                                        f"gradients_embedding-{alg}_rsn-{atlas}_res-{res}" \
                                        f"_all_nodes_rawgraph_sub-{ID}_1_modality-{modality}_model" \
                                        f"-{model}_template-{template}_nodetype-parc_smooth" \
                                        f"-{smooth}fwhm_hpass-{hpass}Hz_extract-{extract}.npy"
                        elif alg == 'OMNI':
                            embedding = f"{base_dir}/embeddings_all_" \
                                        f"{modality}/sub-{ID}/rsn-{atlas}_res-{res}/" \
                                        f"gradients-{alg}_{res}_{atlas}_graph_sub-{ID}" \
                                        f"_1_modality-{modality}_model-{model}_template-{template}" \
                                        f"_nodetype-parc_smooth-{smooth}fwhm_hpass-{hpass}Hz" \
                                        f"_extract-{extract}.npy"
                        if os.path.isfile(embedding):
                            print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
                            data = np.load(embedding)
                            coords, labels = get_coords_labels(embedding)
                            if alg not in subject_dict[ID][ses][modality][comb_tuple].keys():
                                subject_dict[ID][ses][modality][comb_tuple][alg] = {}
                            subject_dict[ID][ses][modality][comb_tuple][alg]['coords'] = coords
                            subject_dict[ID][ses][modality][comb_tuple][alg]['labels'] = labels
                            subject_dict[ID][ses][modality][comb_tuple][alg]['data'] = data
                        else:
                            print(
                                f"Functional embedding not found for {id} and"
                                f" recipe {comb_tuple}...")
                            continue
                    elif alg == 'topology':
                        data = np.empty([len(mets), 1], dtype=np.float32)
                        data[:] = np.nan
                        i = 0
                        for met in mets:
                            col = (
                                'rsn-'
                                + atlas
                                + "_res-"
                                + res
                                + "_model-"
                                + model
                                + f"_template-{template}_nodetype-parc_"
                                + "smooth-"
                                + str(smooth)
                                + "fwhm_hpass-"
                                + str(hpass)
                                + "Hz_extract-"
                                + extract
                                + "_thrtype-"
                                + thr_type
                                + "_auc_"
                                + met
                                + "_auc"
                            )
                            if col in df_top.columns:
                                try:
                                    data[i] = df_top[df_top[
                                                         "participant_id"]
                                                     == "sub-" + ID + "_ses-"
                                                     + ses][
                                        col].values[0]
                                except BaseException:
                                    data[i] = np.nan
                            else:
                                data[i] = np.nan
                            del col
                            i += 1
                        subject_dict[ID][ses][modality][comb_tuple][alg] = data

            # Structural case
            elif modality == 'dwi':
                for comb in grid:
                    directget, minlength, model, atlas, res = comb
                    comb_tuple = (atlas, directget, minlength, model, res)
                    subject_dict[ID][ses][modality][comb_tuple] = {}
                    if alg == 'ASE' or alg == 'OMNI':
                        if alg == 'ASE':
                            embedding = f"{base_dir}/embeddings_all" \
                                        f"_{modality}/sub-{ID}/rsn-{atlas}_res-{res}/" \
                                        f"gradients_embedding-{alg}_rsn-{atlas}_res-{res}" \
                                        f"_all_nodes_rawgraph_sub-{ID}_1_modality-{modality}_model" \
                                        f"-{model}_template-{template}_nodetype-parc_samples-" \
                                        f"20000streams_tracktype-local_directget-{directget}_" \
                                        f"minlength-{minlength}.npy"
                            if os.path.isfile(embedding):
                                print(f"Found {ID}, {ses}, {modality}, {comb_tuple}...")
                                data = np.load(embedding)
                                coords, labels = get_coords_labels(embedding)
                                if alg not in subject_dict[ID][ses][modality][comb_tuple].keys():
                                    subject_dict[ID][ses][modality][comb_tuple][alg] = {}
                                subject_dict[ID][ses][modality][comb_tuple][alg]['coords'] = coords
                                subject_dict[ID][ses][modality][comb_tuple][alg]['labels'] = labels
                                subject_dict[ID][ses][modality][comb_tuple][alg]['data'] = data
                            else:
                                continue
                        elif alg == 'OMNI':
                            embedding = f"{base_dir}/embeddings_all" \
                                        f"_{modality}/sub-{ID}/rsn-{atlas}_res-{res}/" \
                                        f"gradients-{alg}_{res}_{atlas}" \
                                        f"_graph_sub-{ID}_1_modality-{modality}_model" \
                                        f"-{model}_template-{template}_nodetype-parc_samples-" \
                                        f"20000streams_tracktype-local_directget-{directget}_" \
                                        f"minlength-{minlength}.npy"
                        if os.path.isfile(embedding):
                            print('FOUND STRUCT...')
                            data = np.load(embedding)
                            coords, labels = get_coords_labels(embedding)
                            if alg not in subject_dict[ID][ses][modality][comb_tuple].keys():
                                subject_dict[ID][ses][modality][comb_tuple][alg] = {}
                            subject_dict[ID][ses][modality][comb_tuple][alg]['coords'] = coords
                            subject_dict[ID][ses][modality][comb_tuple][alg]['labels'] = labels
                            subject_dict[ID][ses][modality][comb_tuple][alg]['data'] = data
                        else:
                            continue
                    elif alg == 'topology':
                        data = np.empty([len(mets), 1], dtype=np.float32)
                        data[:] = np.nan
                        i = 0
                        for met in mets:
                            col = (
                                'rsn-'
                                + atlas
                                + "_res-"
                                + res
                                + "_model-"
                                + model
                                + f"_template-{template}_"
                                + "_nodetype-parc_samples-20000streams"
                                  "_tracktype-"
                                + "local"
                                + "_directget-"
                                + directget
                                + "_minlength-"
                                + minlength
                                + "_thrtype-"
                                + thr_type
                                + "_topology_"
                                + met
                                + "_auc"
                            )
                            if col in df_top.columns:
                                try:
                                    data[i] = df_top[df_top[
                                                         "participant_id"] ==
                                                     "sub-" + ID + "_ses-" +
                                                     ses][
                                        col].values[0]
                                except BaseException:
                                    data[i] = np.nan
                            else:
                                data[i] = np.nan
                            del col
                            i += 1
                        subject_dict[ID][ses][modality][comb_tuple][alg] = data


def cleanNullTerms(d):
   clean = {}
   for k, v in d.items():
      if isinstance(v, dict):
         nested = cleanNullTerms(v)
         if len(nested.keys()) > 0:
            clean[k] = nested
      elif v is not None and v is not np.nan:
         clean[k] = v
   return clean


sub_dict_clean = cleanNullTerms(subject_dict)

# Subset only those participants which have usable data
df = df[df['participant_id'].isin(list(subject_dict.keys()))]
df = df[['participant_id', 'rum_persist', 'dep_1', 'age', 'sex']]

dict_file_path = make_feature_space_dict(df, sub_dict_clean, ses, base_dir)

dict_file_path = '/Users/derekpisner/Downloads/pynets_ml_dict.pkl'


def make_x_y(input_dict, drop_cols, target_var, alg, grid_param, modality):
    import pandas as pd
    import pickle

    print(target_var)
    print(alg)
    print(grid_param)
    with open(input_dict, 'rb') as f:
        ml_dfs = pickle.load(f)
    f.close()

    if grid_param in ml_dfs[modality][alg].keys():
        df_all = ml_dfs[modality][alg][grid_param]
        if df_all is None:
            df_all = pd.Series()
    else:
        df_all = pd.Series()

    if len(df_all) < 30:
        X = None
        Y = None
        print('Feature-space NA')
    else:
        Y = df_all[target_var].values
        X = df_all.drop(columns=drop_cols)
        print(X)
    return X, Y


class _MakeXYInputSpec(BaseInterfaceInputSpec):
    input_dict = traits.Str()
    drop_cols = traits.List()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class _MakeXYOutputSpec(TraitedSpec):
    X = traits.Any()
    Y = traits.Any()
    alg = traits.Str()
    modality = traits.Str()
    grid_param = traits.List()
    target_var = traits.Str()


class MakeXY(SimpleInterface):
    """Interface wrapper for MakeXY"""

    input_spec = _MakeXYInputSpec
    output_spec = _MakeXYOutputSpec

    def _run_interface(self, runtime):
        import gc
        import pandas as pd
        from nipype.utils.filemanip import fname_presuffix, copyfile

        input_dict_tmp = fname_presuffix(
            self.inputs.input_dict, suffix="_tmp", newpath=runtime.cwd
        )
        copyfile(
            self.inputs.input_dict,
            input_dict_tmp,
            copy=True,
            use_hardlink=False)

        [X, Y] = \
            make_x_y(input_dict_tmp, self.inputs.drop_cols,
                     self.inputs.target_var, self.inputs.alg,
                     tuple(self.inputs.grid_param), self.inputs.modality)

        if isinstance(X, pd.DataFrame):
            out_X = f"{runtime.cwd}/X_{self.inputs.target_var}_" \
                    f"{self.inputs.modality}_{self.inputs.alg}_" \
                    f"{'_'.join(self.inputs.grid_param)}.csv"

            X.to_csv(out_X, index=False)
        else:
            out_X = None

        self._results["X"] = out_X
        self._results["Y"] = Y
        self._results["alg"] = self.inputs.alg
        self._results["grid_param"] = list(self.inputs.grid_param)
        self._results["modality"] = self.inputs.modality
        self._results["target_var"] = self.inputs.target_var

        gc.collect()

        return runtime

# Create wf
ml_wf = pe.Workflow(name="ensemble_connectometry")
ml_wf.base_dir = f"{base_dir}/pynets_ml"

inputnode = pe.Node(
    niu.IdentityInterface(
        fields=[
            "input_dict",
            "drop_cols",
            "out_dir"
        ]
    ),
    name="inputnode",
)


os.makedirs(f"{base_dir}/pynets_ml", exist_ok=True)
inputnode.inputs.out_dir = f"{base_dir}/pynets_ml"
inputnode.inputs.input_dict = dict_file_path
inputnode.inputs.drop_cols = drop_cols

make_x_y_func_node = pe.Node(
    MakeXY(),
    name="make_x_y_func_node",
    nested=True
)

combos = list(itertools.product(target_vars,
                                embedding_types,
                                [list(i) for i in
                                 modality_grids['func'][:10]]))

x_y_iters = []
target_vars_list = [i[0] for i in combos]
embedding_types_list = [i[1] for i in combos]
grid_param_list = [i[2] for i in combos]
x_y_iters.append(("target_var", target_vars_list[1:]))
x_y_iters.append(("alg", embedding_types_list[1:]))
x_y_iters.append(("grid_param", grid_param_list[1:]))

make_x_y_func_node.iterables = x_y_iters
make_x_y_func_node.interface.n_procs = 1
make_x_y_func_node._mem_gb = 4
make_x_y_func_node.inputs.modality = 'func'
make_x_y_func_node.inputs.target_var = target_vars_list[0]
make_x_y_func_node.inputs.alg = embedding_types_list[0]
make_x_y_func_node.inputs.grid_param = grid_param_list[0]


class _BSNestedCVInputSpec(BaseInterfaceInputSpec):
    """Input interface wrapper for BSNestedCV"""
    X = traits.Any()
    y = traits.Any()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class _BSNestedCVOutputSpec(TraitedSpec):
    """Output interface wrapper for BSNestedCV"""

    grand_mean_best_estimator = traits.Dict()
    grand_mean_best_Rsquared = traits.Dict()
    grand_mean_best_MSE = traits.Dict()
    mega_feat_imp_dict = traits.Dict()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class BSNestedCV(SimpleInterface):
    """Interface wrapper for BSNestedCV"""

    input_spec = _BSNestedCVInputSpec
    output_spec = _BSNestedCVOutputSpec

    def _run_interface(self, runtime):
        import gc
        from pynets.stats.benchmarking import bootstrapped_nested_cv

        if self.inputs.X:
            X = pd.read_csv(self.inputs.X, index_col=False)
            [grand_mean_best_estimator, grand_mean_best_Rsquared,
             grand_mean_best_MSE, mega_feat_imp_dict] = bootstrapped_nested_cv(
                X, self.inputs.y)
            print(f"Target Outcome: {self.inputs.target_var}")
            print(f"Modality: {self.inputs.modality}")
            print(f"Embedding type: {self.inputs.alg}")
            print(f"Grid Params: {self.inputs.grid_param}")
            print(f"Best Estimator: {grand_mean_best_estimator}")
            print(f"R2: {grand_mean_best_Rsquared}")
            print(f"MSE: {grand_mean_best_MSE}")
            print(
                f"Most important latent positions: "
                f"{list(mega_feat_imp_dict.keys())}")
        else:
            print('Empty feature-space!')
            grand_mean_best_estimator = dict()
            grand_mean_best_Rsquared = dict()
            grand_mean_best_MSE = dict()
            mega_feat_imp_dict = OrderedDict()

        self._results["grand_mean_best_estimator"] = grand_mean_best_estimator
        self._results["grand_mean_best_Rsquared"] = grand_mean_best_Rsquared
        self._results["grand_mean_best_MSE"] = grand_mean_best_MSE
        self._results["mega_feat_imp_dict"] = mega_feat_imp_dict
        self._results["target_var"] = self.inputs.target_var
        self._results["modality"] = self.inputs.modality
        self._results["alg"] = self.inputs.alg
        self._results["grid_param"] = list(self.inputs.grid_param)

        gc.collect()

        return runtime


bootstrapped_nested_cv_node = pe.Node(
    BSNestedCV(),
    name="bootstrapped_nested_cv_node",
    nested=True
)

bootstrapped_nested_cv_node.interface.n_procs = 1
bootstrapped_nested_cv_node.interface._mem_gb = 1


class _MakeDFInputSpec(BaseInterfaceInputSpec):
    grand_mean_best_estimator = traits.Dict()
    grand_mean_best_Rsquared = traits.Dict()
    grand_mean_best_MSE = traits.Dict()
    mega_feat_imp_dict = traits.Dict()
    target_var = traits.Str()
    modality = traits.Str()
    alg = traits.Str()
    grid_param = traits.List()


class _MakeDFOutputSpec(TraitedSpec):

    df_summary = traits.Str()


class MakeDF(SimpleInterface):

    input_spec = _MakeDFInputSpec
    output_spec = _MakeDFOutputSpec

    def _run_interface(self, runtime):
        import gc
        import pandas as pd
        import numpy as np

        df_summary = pd.DataFrame(
            columns=["modality", "grid", "alg", "best_estimator", "Rsquared",
                     "MSE", "target_variable", "lp_importance"])

        df_summary.at[0, "target_variable"] = self.inputs.target_var
        df_summary.at[0, "modality"] = self.inputs.modality
        df_summary.at[0, "alg"] = self.inputs.alg
        df_summary.at[0, "grid"] = tuple(self.inputs.grid_param)

        if self.inputs.grand_mean_best_estimator:
            df_summary.at[0, "best_estimator"] = max(
                set(list(self.inputs.grand_mean_best_estimator.values())),
                key=list(self.inputs.grand_mean_best_estimator.values()).count)
            df_summary.at[0, "Rsquared"] = np.mean(
                list(self.inputs.grand_mean_best_Rsquared.values()))
            df_summary.at[0, "MSE"] = \
                np.mean(list(self.inputs.grand_mean_best_MSE.values()))
            df_summary.at[0, "lp_importance"] = \
                np.array(list(self.inputs.mega_feat_imp_dict.keys()))

        else:
            df_summary.at[0, "best_estimator"] = np.nan
            df_summary.at[0, "Rsquared"] = np.nan
            df_summary.at[0, "MSE"] = np.nan
            df_summary.at[0, "lp_importance"] = np.nan

        out_df_summary = f"{runtime.cwd}/df_summary_" \
                         f"{self.inputs.target_var}_" \
                f"{self.inputs.modality}_{self.inputs.alg}_" \
                f"{'_'.join(self.inputs.grid_param)}.csv"
        df_summary.to_csv(out_df_summary, index=False)

        self._results["df_summary"] = out_df_summary

        gc.collect()

        return runtime

make_df_node = pe.Node(
    MakeDF(),
    name="make_df_node"
)

make_df_node.interface.n_procs = 1
make_df_node.interface._mem_gb = 1

df_join_node = pe.JoinNode(
    niu.IdentityInterface(fields=["df_summary"]),
    name="df_join_node",
    joinfield=["df_summary"],
    joinsource=make_x_y_func_node,
)


def concatenate_frames(out_dir, files_):
    import pandas as pd

    dfs = []
    for file_ in files_:
        df = pd.read_csv(file_, chunksize=100000).read()
        try:
            df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
        except BaseException:
            pass
        dfs.append(df)
    frame = pd.concat(dfs, axis=0, join="outer", sort=True,
                      ignore_index=False)
    out_path = f"{out_dir}/final_df.csv"
    frame.to_csv(out_path, index=False)

    return out_path


concatenate_frames_node = pe.Node(
    niu.Function(
        input_names=["out_dir", "files_"],
        output_names=["out_path"],
        function=concatenate_frames,
    ),
    name="concatenate_frames_node",
)

outputnode = pe.Node(
    niu.IdentityInterface(
        fields=["df_summary"]),
    name="outputnode")

ml_wf.connect(
    [
        (
            inputnode, make_x_y_func_node,
            [("input_dict", "input_dict"), ("drop_cols", "drop_cols")]
        ),
        (
            make_x_y_func_node, bootstrapped_nested_cv_node,
            [("X", "X"), ("Y", "y"), ("target_var", "target_var"),
             ("modality", "modality"),
             ("alg", "alg"), ("grid_param", "grid_param")]
        ),
        (
            bootstrapped_nested_cv_node, make_df_node,
            [("grand_mean_best_estimator", "grand_mean_best_estimator"),
             ("grand_mean_best_Rsquared", "grand_mean_best_Rsquared"),
             ("grand_mean_best_MSE", "grand_mean_best_MSE"),
             ("mega_feat_imp_dict", "mega_feat_imp_dict"),
             ("target_var", "target_var"), ("modality", "modality"),
             ("alg", "alg"), ("grid_param", "grid_param")]
        ),
        (
            make_df_node, df_join_node,
            [("df_summary", "df_summary")]
        ),
        (
            df_join_node, concatenate_frames_node,
            [("df_summary", "files_")]
        ),
        (
            inputnode, concatenate_frames_node,
            [("out_dir", "out_dir")]
        ),
        (
            concatenate_frames_node, outputnode,
            [("out_path", "df_summary")]
        )
    ]
)

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_" \
               "frozen_importlib.BuiltinImporter'>)"
    execution_dict = {}
    execution_dict["crashdump_dir"] = str(ml_wf.base_dir)
    execution_dict["poll_sleep_duration"] = 0.1
    execution_dict["crashfile_format"] = 'txt'
    execution_dict['local_hash_check'] = False
    execution_dict['hash_method'] = 'timestamp'

    cfg = dict(execution=execution_dict)

    for key in cfg.keys():
        for setting, value in cfg[key].items():
            ml_wf.config[key][setting] = value

    nthreads = psutil.cpu_count(logical=False)
    procmem = [int(nthreads),
               int(list(psutil.virtual_memory())[4]/1000000000) - 2]
    plugin_args = {
        "n_procs": int(procmem[0]),
        "memory_gb": int(procmem[1]),
        "scheduler": "mem_thread",
    }
    out = ml_wf.run(plugin='MultiProc', plugin_args=plugin_args)


# import seaborn as sns
# from pynets.core.utils import flatten
# frame = pd.read_csv('/private/tmp/pynets_ml/final_df.csv', index_col=False)
#
# df_summary = frame[np.abs(frame['Rsquared'])>0].sort_values(
#     "Rsquared", ascending=True)
#
# df_summary['Rsquared'] = np.abs(df_summary['Rsqu'lp_importance'ared'].astype('float64'))
# df_summary['MSE'] = df_summary['MSE'].astype('float64')
# df_summary['alg'] = df_summary['alg'].astype('str')
# df_summary['best_estimator'] = df_summary['best_estimator'].astype('str')
#
# all_nodes = list(set(list(flatten([eval(i) for i in df_summary['lp_importance']]))))
#
# sns.set(style="whitegrid")
# ax = sns.violinplot(x='alg', y="Rsquared", data=df_summary, palette="Pastel1")
# ax = sns.violinplot(x='alg', y="MSE", data=df_summary, palette="Pastel1")
#
# df_grid = df_summary.copy()
# df_grid[['extract', 'hpass', 'model', 'res', 'smooth']] = \
#     pd.DataFrame([eval(i) for i in df_summary['grid'].tolist()], index=df_summary.index)
# df_grid = df_grid[['extract', 'hpass', 'model', 'res', 'smooth', 'Rsquared']]
#
# fig = px.parallel_categories(df_grid, color="Rsquared",
#                              labels={"res": "Node Resolution",
#                                      "model": "Estimator",
#                                      "extract": "Extraction Method",
#                                      "hpass": "High-Pass (Hz)",
#                                      "smooth": "Smooth (FWHM)"},
#                              dimensions=['res', 'model', 'extract', 'hpass',
#                                          'smooth'],
#                              color_continuous_scale=px.colors.sequential.Inferno)
# fig.update_layout(
#     autosize=False,
#     height=1000,
#     width=2000,
#     margin=dict(r=200, l=200, b=75, t=75),
# )
# fig.update_traces(labelfont=dict(size=24, color='black'),
#                   tickfont=dict(family="Arial, sans-serif",
#                                 size=20, color='black'))
# fig.write_image('func_rsquared_parallel_all.png')
