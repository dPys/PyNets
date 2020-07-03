PyNets™
=======
[![CircleCI](https://circleci.com/gh/dPys/PyNets.svg?style=svg)](https://circleci.com/gh/dPys/PyNets)
[![codecov](https://codecov.io/gh/dPys/PyNets/branch/master/graph/badge.svg)](https://codecov.io/gh/dPys/PyNets?branch=master)
[![PyPI - Version](https://img.shields.io/pypi/v/omniduct.svg)](https://pypi.org/project/pynets/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pynets.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![](docs/_static/logo.png)

About
-----
PyNets is a tool for sampling and analyzing varieties of individual structural and functional connectomes. PyNets enables the user to specify any of a several of methodological choices known to impact brain network node and/or edge definition, and then sample the prescribed connectome estimates in a massively parallel framework that in turn becomes conducive to grid-search and multi-view learning. PyNets is a post-processing workflow, which means that it can be run on virtually any preprocessed fMRI or dMRI data. It draws from Dipy, Nilearn, GrasPy, and Networkx libraries, but is powered primarily through the Nipype workflow engine. PyNets can now also be deployed as a BIDS application, where it takes BIDS derivatives and makes BIDS derivatives.

Install
-------
Dockerhub (preferred):
```
docker pull dpys/pynets:latest
```

Manual (Requires a local dependency install of FSL version >=5.0.9 and forked version of Nilearn 0.6.0):
```
pip install pynets --user
```
or
```
git clone https://github.com/dpys/pynets
cd PyNets
pip install -r requirements.txt --user
python setup.py install
```

Documentation
-------------
Explore official installation instruction, user-guide, API, and examples: <https://pynets.readthedocs.io/en/latest/>

Citing
------
A manuscript is in preparation, but for now, please cite ALL uses with the following entry:
```
@CONFERENCE{
    title = {PyNets: A Reproducible Workflow for Structural and Functional Connectome Ensemble Learning},
    author = {Pisner, D., Hammonds R.}
    publisher = {Poster session presented at: Annual Meeting of the Organization for Human Brain Mapping}
    url = {https://github.com/dPys/PyNets},
    year = {2020},
    month = {June}
}
```

The `pynets_bids` CLI
---------------------
```
usage: pynets_bids [-h]
                   [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                   [--session_label SESSION_LABEL [SESSION_LABEL ...]]
                   [--push_location PUSH_LOCATION]
                   [-ua Path to parcellation file in MNI-space [Path to parcellation file in MNI-space ...]]
                   [-cm Cluster mask [Cluster mask ...]]
                   [-roi Path to binarized Region-of-Interest ROI) Nifti1Image [Path to binarized Region-of-Interest (ROI Nifti1Image ...]]
                   [-ref Atlas reference file path]
                   [-way Path to binarized Nifti1Image to constrain tractography [Path to binarized Nifti1Image to constrain tractography ...]]
                   [-config Optional path to a config.json file with runtime settings.]
                   [-pm Cores,memory] [-plug Scheduler type] [-v]
                   [-work Working directory]
                   bids_dir output_dir {participant,group} {dwi,func}
                   [{dwi,func} ...]
pynets_bids: the following arguments are required: bids_dir, output_dir, analysis_level, modality
```

where the `-config` flag specifies that path to a .json configuration spec that includes at least one of many possible connectome recipes to apply to your data. Pre-built configuration files are included in the pynets/config directory, and an example is shown here (with commented descriptions):

```
{
    "dwi": {
            "dg": "['prob', 'det']",  # Indicates the direction-getting method(s) of tractography.
            "ml": "['10', '40']",  # Indicates the minimum streamline length(s) for tractographic filtering.
            "mod": "['csd', 'csa', 'tensor']"  # Indicates the type(s) of diffusion model estimators for fixel reconstruction. At least 1 is required for structural connectometry.
        },
    "func": {
            "ct": "['rena', 'ward', 'kmeans']", # Indicates the type(s) of clustering that will be used to generate a clustering-based parcellation. This should be left as "None" if no clustering will be performed, but can be included simultaneously with the `-a` and `-ua` parcellation options.
            "k": "['200', '400', '600']", # Indicates the number of clusters to generate in a clustering-based parcellation. This should be left as "None" if no clustering will be performed.
            "hp": "['0', '0.028', '0.080']", # Indicates the high-pass frequenc(ies) to apply to signal extraction from nodes.
            "mod": "['partcorr', 'sps']", # Indicates the functional connectivity estimator(s) to use. At least 1 is required for functional connectometry.
            "sm": "['0', '2', '4']", # Indicates the smoothing FWHM value(s) to apply during the nodal time-series signal extraction.
            "es": "['mean', 'median']" # Indicates the method(s) of nodal time-series signal extraction.
        },
    "gen": {
            "a":  "DesikanKlein2012", # Anatomical atlases to define nodes.
            "bin":  "False", # Binarize the resulting connectome graph before analyzing it. Note that undirected weighted graphs are analyzed by default.
            "embed":  "False", # Activate omnibus and single-graph adjacency spectral embedding of connectome estimates sampled.
            "mplx":  0, # If both functional and structural data is provided, the type of multiplex connectome modeling to perform. See `pynets -h` for more information.
            "n":  "['Cont', 'Default']", # Which, if any, Yeo-7/17 resting-state sub-networks to select from the given parcellation. If multiple are specified, all other options will iterate across each.
            "norm": "['6']", # Level of normalization to apply to graph (e.g. standardize betwee 0-1, Pass-to-Ranks (PTR), log10).
            "spheres":  "False", # Use spheres as nodes (vs. parcel labels, the default).
            "ns":  "None", # If `spheres` is True, this indicates integer radius size(s) of spherical centroid nodes.
            "p":  "['1']", # Apply anti-fragmentation, largest connected-component subgraph selection, or any of a variety of hub-detection methods to graph(s).
            "plt":  "False", # Activate plotting (adjancency matrix and glass-brain included by default).
            "thr":  1.0, # A threshold (0.0-1.0). This can be left as "None" if multi-thresholding is used.
            "max_thr":  0.80, # If performing multi-thresholding, a minimum threshold.
            "min_thr":  0.20, # If performing multi-thresholding, a maximum threshold.
            "step_thr":  0.10, # If performing multi-thresholding, a threshold interval size.
            "dt":  "False", # Global thresholding to achieve a target density. (Only one of `mst`, `dt`, and `df` can be used).
            "mst":  "True", # Local thresholding using the Minimum-Spanning Tree approach. (Only one of `mst`, `dt`, and `df` can be used).
            "df":  "False", # Local thresholding using a disparity filter. (Only one of `mst`, `dt`, and `df` can be used).
            "vox":  "'2mm'" # Voxel size (1mm or 2mm). 2mm is the default.
        }
}
```

![Multiplex Layers](docs/_static/structural_functional_multiplex.png)
![Multiplex Glass](docs/_static/glassbrain_mplx.png)
![Yeo7](docs/_static/yeo7_mosaic.png)
![Workflow DAG](docs/_static/graph.png)
