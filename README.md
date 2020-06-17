PyNets™
=======
[![Build Status](https://travis-ci.org/dPys/PyNets.svg?branch=master)](https://travis-ci.org/dPys/PyNets)
[![CircleCI](https://circleci.com/gh/dPys/PyNets.svg?style=svg)](https://circleci.com/gh/dPys/PyNets)
[![codecov](https://codecov.io/gh/dPys/PyNets/branch/master/graph/badge.svg)](https://codecov.io/gh/dPys/PyNets?branch=master)
[![PyPI - Version](https://img.shields.io/pypi/v/omniduct.svg)](https://pypi.org/project/pynets/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pynets.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![](docs/_static/logo.png)

About
-----
PyNets is a tool for sampling and analyzing varieties of individual structural and functional connectomes. PyNets enables the user to specify any of a variety of methodological choices known to impact node and/or edge definition, and then sample the prescribed connectome estimates, in a massively parallel framework, conducive to grid-search. PyNets is a post-processing workflow, which means that it can be run on virtually any preprocessed fMRI or dMRI data. It draws from Dipy, Nilearn, GrasPy, and Networkx libraries, but is powered primarily through the Nipype workflow engine. PyNets can now also be deployed as a BIDS application, where it takes BIDS derivatives and makes BIDS derivatives. 

Documentation
-------------
Explore official installation instruction, user-guide, API, and examples: <https://pynets.readthedocs.io/en/latest/>

Citing
------
A manuscript is in preparation, but for now, please cite all uses with the following entry:
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
pynets_bids: error: the following arguments are required: bids_dir, output_dir, analysis_level, modality
```

where the `-config` flag specifies that path to a .json configuration spec that includes at least one of many possible connectome recipes to apply to your data. Pre-built configuration files are included in the pynets/config directory, and an example is shown here:

```
{
    "dwi": {
            "dg": "['prob', 'det']",
            "ml": "['10', '40']",
            "mod": "['csd', 'csa', 'tensor']",
            "s": 10000,
            "tc": "['bin']",
            "tt": "['local']"
        },
    "func": {
            "cc": "['allcorr']",
            "ct": "['rena', 'ward', 'kmeans']",
            "hp": "['0', '0.028', '0.080']",
            "k": "['200', '400', '600']",
            "mod": "['partcorr', 'sps']",
            "sm": "['0', '2', '4']",
            "es": "['mean', 'median']"
        },
    "gen": {
            "a":  "DesikanKlein2012",
            "bin":  "False",
            "df":  "False",
            "dt":  "False",
            "embed":  "False",
            "max_thr":  0.80,
            "min_thr":  0.20,
            "mplx":  0,
            "mst":  "True",
            "n":  "['Cont', 'Default']",
            "names":  "True",
            "norm": "['6']",
            "ns":  "None",
            "p":  "['1']",
            "plt":  "False",
            "spheres":  "False",
            "step_thr":  0.10,
            "thr":  1.0,
            "vox":  "'2mm'"
        }
}
```

![Multiplex Layers](docs/_static/structural_functional_multiplex.png)
![Multiplex Glass](docs/_static/glassbrain_mplx.png)
![Yeo7](docs/_static/yeo7_mosaic.png)
![Workflow DAG](docs/_static/graph.png)
