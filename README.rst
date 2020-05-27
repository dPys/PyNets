PyNets™
=======
.. image:: https://travis-ci.org/dPys/PyNets.svg?branch=master
.. image:: https://circleci.com/gh/dPys/PyNets.svg?branch=master
.. image:: https://codecov.io/gh/dPys/PyNets/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/dPys/PyNets

About
-----
A Reproducible Workflow for Structural-Functional Connectome Sampling, Ensembles, Optimization, and Analysis.

PyNets leverages the Nipype workflow engine, along with Nilearn and Dipy fMRI and dMRI libraries, to sample individual structural and functional connectomes. Uniquely, PyNets enables the user to specify any of a variety of methodological choices (i.e. that impact node and/or edge definitions) and sampling the resulting connectome estimates in a massively scalable and parallel framework. PyNets is a post-processing workflow, which means that it can be run manually on virtually any preprocessed fMRI or dMRI data. Further, it can be deployed as a BIDS application that takes BIDS derivatives and makes BIDS derivatives. Docker and Singularity containers are further available to facilitate reproducibility of executions. Cloud computing with AWS batch and S3 is also supported.

Documentation
-------------
Official installation, user-guide, and API docs now live here: https://pynets.readthedocs.io/en/latest/

Citing
------
A manuscript is in preparation, but for now, please cite all uses with reference
to the github repository: https://github.com/dPys/PyNets

.. image:: /_static/multimodalconnectome.png
