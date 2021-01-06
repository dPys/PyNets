PyNets®
=======
[![CircleCI](https://circleci.com/gh/dPys/PyNets.svg?style=svg)](https://circleci.com/gh/dPys/PyNets)
[![codecov](https://codecov.io/gh/dPys/PyNets/branch/master/graph/badge.svg)](https://codecov.io/gh/dPys/PyNets?branch=master)
[![PyPI - Version](https://img.shields.io/pypi/v/omniduct.svg)](https://pypi.org/project/pynets/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pynets.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

About
-----
PyNets is a tool for sampling and analyzing varieties of individual structural and functional connectomes. Using decision-tree learning, along with extensive bagging and boosting, PyNets is the first application of its kind to facilitate fully-reproducible, parametric sampling of connectome ensembles from neuroimaging data. As a post-processing workflow, PyNets is intended for any preprocessed fMRI or dMRI data in native anatomical space such that it supports normative-referenced connectotyping at the individual-level. Towards these ends, it comprehensively integrates best-practice tractography and functional connectivity analysis methods based open-source libraries such as Dipy and Nilearn, though it is powered primarily through NetworkX and the Nipype workflow engine. PyNets can now also be deployed as a BIDS application, where it takes BIDS derivatives and makes BIDS derivatives.

Install
-------
## Dockerhub (preferred):
```
docker pull dpys/pynets
```

## Manual
(REQUIRES a local dependency install of [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) version >=5.0.9, as well as an
installation of [Python3.6+](https://www.python.org/download/releases/3.0/) with GUI programming enabled (See [tkinter](https://docs.python.org/3/library/tkinter.html#module-tkinter))
```
[sudo] pip install pynets [--user]
```
or
```
# Install git-lfs
brew install git-lfs (macOS) or [sudo] apt-get install git-lfs (linux)
git lfs install --skip-repo

# Clone the repository and install
git clone https://github.com/dpys/pynets
cd PyNets
[sudo] python setup.py install [--user]
```

Hardware Requirements
---------------------
4 vCPUs, 8+ GB free RAM, and at least 8-16 GB of free disk space.

Operating Systems
-----------------
UNIX/MacOS 64-bit platforms

Windows 10 with [WSL2](https://docs.microsoft.com/en-us/windows/wsl/compare-versions#whats-new-in-wsl-2)


Documentation
-------------
Explore official installation instructions, user-guide, API, and examples: <https://pynets.readthedocs.io/en/latest/>

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

Data already preprocessed with BIDS apps like fmriprep, CPAC, dmriprep? If your BIDS derivatives can be queried with pybids, then you should be able to run them with the user-friendly `pynets_bids` CLI!
```
   pynets_bids '/hnu/fMRIprep/fmriprep' '/Users/dPys/outputs/pynets' participant func --participant_label 0025427 0025428 --session_label 1 2 -config pynets/config/bids_config.json

```
*Note: If you preprocessed your BOLD data using fMRIprep, then you will need to have specified either `T1w` or `anat` in the list of fmriprep `--output-spaces`. Similarly, if you preprocessed your data using CPAC, then you will want to be sure that an ALFF image exists. PyNets does NOT currently accept template-normalized BOLD or DWI data. See the usage docs for more information on compatible file types.


where the `-config` flag specifies that path to a .json configuration spec that includes at least one of many possible connectome recipes to apply to your data. Pre-built configuration files are available (see: <https://github.com/dPys/PyNets/tree/master/pynets/config>), and an example is shown here (with commented descriptions):

```
{
    "func": { # fMRI options. If you only have functional (i.e. BOLD) data, set each of the `dwi` options to "None"
            "ct": "None", # Indicates the type(s) of clustering that will be used to generate a clustering-based parcellation. This should be left as "None" if no clustering will be performed, but can be included simultaneously with `-a`.
            "k": "None", # Indicates the number of clusters to generate in a clustering-based parcellation. This should be left as "None" if no clustering will be performed.
            "hp": "['0', '0.028', '0.080']", # Indicates the high-pass frequenc(ies) to apply to signal extraction from nodes.
            "mod": "['partcorr', 'cov']", # Indicates the functional connectivity estimator(s) to use. At least 1 is required for functional connectometry.
            "sm": "['0', '4']", # Indicates the smoothing FWHM value(s) to apply during the nodal time-series signal extraction.
            "es": "['mean', 'median']" # Indicates the method(s) of nodal time-series signal extraction.
        },
    "dwi": { # dMRI options. If you only have structural (i.e. DWI) data, set each of the `func` options to "None"
            "dg": "det", # The directional assumptions of tractography (e.g. deterministic, probabilistic)
            "ml": "40", # The minimum length criterion for streamlines in tractography
            "mod": "csd", # The diffusion model type
            "tol": "8" # The tolerance distance (in the units of the streamlines, usually mm). If any node in the streamline is within this distance from the center of any voxel in the ROI, then the connection is counted as an edge"
        },
    "gen": { # These are general options that apply to all modalities
            "a":  "['BrainnetomeAtlasFan2016', 'atlas_harvard_oxford', 'destrieux2009_rois']", # Anatomical atlases to define nodes.
            "bin":  "False", # Binarize the resulting connectome graph before analyzing it. Note that undirected weighted graphs are analyzed by default.
            "embed":  "False", # Activate omnibus and single-graph adjacency spectral embedding of connectome estimates sampled.
            "mplx":  0, # If both functional and structural data are provided, this parameter [0-3] indicates the type of multiplex connectome modeling to perform. See `pynets -h` for more details on multiplex modes.
            "n":  "['Cont', 'Default']", # Which, if any, Yeo-7/17 resting-state sub-networks to select from the given parcellation. If multiple are specified, all other options will iterate across each.
            "norm": "['6']", # Level of normalization to apply to graph (e.g. standardize betwee 0-1, Pass-to-Ranks (PTR), log10).
            "spheres":  "False", # Use spheres as nodes (vs. parcel labels, the default).
            "ns":  "None", # If `spheres` is True, this indicates integer radius size(s) of spherical centroid nodes.
            "p":  "['1']", # Apply anti-fragmentation, largest connected-component subgraph selection, or any of a variety of hub-detection methods to graph(s).
            "plt":  "False", # Activate plotting (adjacency matrix and glass-brain included by default).
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

Data not in BIDS format and/or preprocessed using in-house tools?
No problem-- you can still run pynets manually:
```
    pynets -id '002_1' '/Users/dPys/outputs/pynets' \ # where `-id` is an arbitrary subject identifier and the first path is an arbitrary output directory to store derivatives of the workflow.
    -func '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/func/BOLD_PREPROCESSED_IN_ANAT_NATIVE.nii.gz' \ # The fMRI BOLD image data.
    -anat '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/anat/ANAT_PREPROCESSED_NATIVE.nii.gz' \ # The T1w anatomical image. This is mandatory -- PyNets requires a T1/T2-weighted anatomical image unless you are analyzing raw graphs that ahve already been produced.
    -a 'BrainnetomeAtlasFan2016' \ # An anatomical atlas name. Note that if were to omit the `-a` flag, a custom parcellation file would need to be specified using the `-a` flag instead or a valid clustering mask (`-cm`) would be needed to generate an individual parcellation. For a complete catalogue of anatomical atlases available in PyNets, see the `Usage` section of the documentation.
    -mod 'partcorr' \ # The connectivity model. In the case of structural connectometry, this becomes the diffusion model type.
    -thr 0.20 \ # Optionally apply a single proportional threshold to the generated graph.
```

```
    pynets -id '002_1' '/Users/dPys/outputs/pynets' \ # where `-id` is an arbitrary subject identifier and the first path is an arbitrary output directory to store derivatives of the workflow.
    -dwi '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/dwi/DWI_PREPROCESSED_NATIVE.nii.gz' \ # The dMRI diffusion-weighted image data.
    -bval '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/dwi/BVAL.bval' \ # The b-values.
    -bvec '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/dwi/BVEC.bvec' \ # The b-vectors.
    -anat '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/anat/ANAT_PREPROCESSED_NATIVE.nii.gz' \ # The T1w anatomical image.
    -a '/Users/dPys/.atlases/MyCustomParcellation-scale1.nii.gz' '/Users/dPys/.atlases/MyCustomParcellation-scale2.nii.gz' \ # The parcellations.
    -mod 'csd' 'csa' 'sfm' \ # The (diffusion) connectivity model(s).
    -dg 'prob' 'det'  \ # The tractography direction-getting method.
    -mst -min_thr 0.20 -max_thr 0.80 -step_thr 0.10 # Multi-thresholding from the Minimum-Spanning Tree, with AUC graph analysis.
    -n 'Default' # The resting-state network definition to restrict node-making.
```

![Multiplex Layers](docs/_static/structural_functional_multiplex.png)
![Multiplex Glass](docs/_static/glassbrain_mplx.png)
![Yeo7](docs/_static/yeo7_mosaic.png)
![Workflow DAG](docs/_static/graph.png)
