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
PyNets leverages the Nipype workflow engine, along with Nilearn and Dipy fMRI and dMRI libraries, to sample individual structural and functional connectomes. Uniquely, PyNets enables the user to specify any of a variety of methodological choices (i.e. that impact node and/or edge definitions) and sampling the resulting connectome estimates in a massively scalable and parallel framework. PyNets is a post-processing workflow, which means that it can be run manually on virtually any preprocessed fMRI or dMRI data. Further, it can be deployed as a BIDS application that takes BIDS derivatives and makes BIDS derivatives. Docker and Singularity containers are further available to facilitate reproducibility of executions. Cloud computing with AWS batch and S3 is also supported.

Documentation
-------------
Explore official installation instruction, user-guide, API, and examples: <https://pynets.readthedocs.io/en/latest/>

Citing
------
A manuscript is in preparation, but for now, please cite all uses with reference
to the github repository: <https://github.com/dPys/PyNets>

![Multiplex Layers](docs/_static/structural_functional_multiplex.png)
![Multiplex Glass](docs/_static/glassbrain_mplx.png)
![Yeo7](docs/_static/yeo7_mosaic.png)
![Workflow DAG](docs/_static/graph.png)
