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
PyNets is a tool for sampling and analyzing varieties of individual structural and functional connectomes. PyNets enables the user to specify any of a variety of methodological choices  impacting node and/or edge definition, and then sample the prescribed connectome estimates in a massively parallel framework that is conducive to predictive optimization (i.e. grid-search). PyNets is a post-processing workflow, which means that it can be run on virtually any preprocessed fMRI or dMRI data. It relies on Dipy, Nilearn, Networkx, and the Nipype workflow engine under-the-hood. It can now also be deployed as a BIDS application, where it takes BIDS derivatives and makes BIDS derivatives. 

Documentation
-------------
Explore official installation instruction, user-guide, API, and examples: <https://pynets.readthedocs.io/en/latest/>

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

Citing
------
A manuscript is in preparation, but for now, please cite all uses with the following enty:
@CONFERENCE{
    title = {PyNets: A Reproducible Workflow for Structural and Functional Connectome Ensemble Learning},
    author = {Pisner, D., Hammonds R.}
    publisher = {Poster session presented at: Annual Meeting of the Organization for Human Brain Mapping}
    url = {https://github.com/dPys/PyNets},
    year = {2020},
    month = {June}
}

![Multiplex Layers](docs/_static/structural_functional_multiplex.png)
![Multiplex Glass](docs/_static/glassbrain_mplx.png)
![Yeo7](docs/_static/yeo7_mosaic.png)
![Workflow DAG](docs/_static/graph.png)
