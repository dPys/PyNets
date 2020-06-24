PyNets™
=======

About
-----
A Reproducible Workflow for Structural and Functional Connectome Ensemble Learning

PyNets leverages the Nipype workflow engine, along with Nilearn and Dipy fMRI and dMRI libraries, to sample individual structural and functional connectomes. Uniquely, PyNets enables the user to specify any of a variety of methodological choices (i.e. that impact node and/or edge definitions) and sampling the resulting connectome estimates in a massively scalable and parallel framework. PyNets is a post-processing workflow, which means that it can be run manually on virtually any preprocessed fMRI or dMRI data. Further, it can be deployed as a BIDS application that takes BIDS derivatives and makes BIDS derivatives. Docker and Singularity containers are further available to facilitate reproducibility of executions. Cloud computing with AWS batch and S3 is also supported.

Documentation
-------------
Official installation, user-guide, and API docs now live here: https://pynets.readthedocs.io/en/latest/

Citing
------
A manuscript is in preparation, but for now, please cite all uses with the following entry:

Pisner, D., Hammonds R. (2020) PyNets: A Reproducible Workflow for Structural and Functional Connectome
    Ensemble Learning. Poster session presented at the 26th Annual Meeting of the Organization for
    Human Brain Mapping. https://github.com/dPys/PyNets.
