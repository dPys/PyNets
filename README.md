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
usage: pynets [-h] -id A subject id or other unique identifier
              [A subject id or other unique identifier ...]
              [-func Path to input functional file required for functional connectomes) [Path to input functional file (required for functional connectomes) ...]]
              [-dwi Path to diffusion-weighted imaging data file (required for dmri connectomes) [Path to diffusion-weighted imaging data file (required for dmri connectomes) ...]]
              [-bval Path to b-values file (required for dmri connectomes) [Path to b-values file (required for dmri connectomes) ...]]
              [-bvec Path to b-vectors file (required for dmri connectomes) [Path to b-vectors file (required for dmri connectomes) ...]]
              [-anat Path to a skull-stripped anatomical Nifti1Image [Path to a skull-stripped anatomical Nifti1Image ...]]
              [-m Path to binarized mask Nifti1Image to apply to regions before extracting signals [Path to binarized mask Nifti1Image to apply to regions before extracting signals ...]]
              [-conf Confound regressor file (.tsv/.csv format) [Confound regressor file (.tsv/.csv format) ...]]
              [-g Path to graph file input. [Path to graph file input. ...]]
              [-roi Path to binarized Region-of-Interest (ROI) Nifti1Image [Path to binarized Region-of-Interest (ROI) Nifti1Image ...]]
              [-ref Atlas reference file path]
              [-way Path to binarized Nifti1Image to constrain tractography [Path to binarized Nifti1Image to constrain tractography ...]]
              [-mod Connectivity estimation/reconstruction method [Connectivity estimation/reconstruction method ...]]
              [-ua Path to parcellation file in MNI-space [Path to parcellation file in MNI-space ...]]
              [-a Atlas [Atlas ...]]
              [-ns Spherical centroid node size [Spherical centroid node size ...]]
              [-thr Graph threshold]
              [-min_thr Multi-thresholding minimum threshold]
              [-max_thr Multi-thresholding maximum threshold]
              [-step_thr Multi-thresholding step size]
              [-sm Smoothing value (mm fwhm) [Smoothing value (mm fwhm) ...]]
              [-hp High-pass filter (Hz) [High-pass filter (Hz) ...]]
              [-es Node extraction strategy [Node extraction strategy ...]]
              [-k Number of k clusters [Number of k clusters ...]]
              [-ct Clustering type [Clustering type ...]]
              [-cc Clustering connectivity type]
              [-cm Cluster mask [Cluster mask ...]]
              [-ml Minimum fiber length for tracking [Minimum fiber length for tracking ...]]
              [-dg Direction getter [Direction getter ...]]
              [-tt Tracking algorithm] [-tc Tissue classification method]
              [-s Number of samples]
              [-norm Normalization strategy for resulting graph(s] [-bin]
              [-dt] [-mst] [-p Pruning strategy] [-df]
              [-mplx Perform various levels of multiplex graph analysis ONLY IF both structural and diffusion connectomes are provided.]
              [-embed] [-spheres]
              [-n Resting-state network [Resting-state network ...]] [-names]
              [-vox {1mm,2mm}] [-plt] [-pm Cores,memory]
              [-plug Scheduler type] [-v] [-clean] [-work Working directory]
              [--version]
              output_dir

PyNets: A Reproducible Workflow for Structural and Functional Connectome
Ensemble Learning

positional arguments:
  output_dir            The directory to store pynets derivatives.

optional arguments:
  -h, --help            show this help message and exit
  -id A subject id or other unique identifier [A subject id or other unique identifier ...]
                        An subject identifier OR list of subject identifiers,
                        separated by space and of equivalent length to the
                        list of input files indicated with the -func flag.
                        This parameter must be an alphanumeric string and can
                        be arbitrarily chosen. If functional and dmri
                        connectomes are being generated simultaneously, then
                        space-separated id's need to be repeated to match the
                        total input file count.
  -func Path to input functional file (required for functional connectomes) [Path to input functional file (required for functional connectomes) ...]
                        Specify either a path to a preprocessed functional
                        Nifti1Image in MNI152 space OR multiple space-
                        separated paths to multiple preprocessed functional
                        Nifti1Image files in MNI152 space and in .nii or
                        .nii.gz format, OR the path to a text file containing
                        a list of paths to subject files.
  -dwi Path to diffusion-weighted imaging data file (required for dmri connectomes) [Path to diffusion-weighted imaging data file (required for dmri connectomes) ...]
                        Specify either a path to a preprocessed dmri diffusion
                        Nifti1Image in native diffusion space and in .nii or
                        .nii.gz format OR multiple space-separated paths to
                        multiple preprocessed dmri diffusion Nifti1Image files
                        in native diffusion space and in .nii or .nii.gz
                        format.
  -bval Path to b-values file (required for dmri connectomes) [Path to b-values file (required for dmri connectomes) ...]
                        Specify either a path to a b-values text file
                        containing gradient shell values per diffusion
                        direction OR multiple space-separated paths to
                        multiple b-values text files in the order of
                        accompanying b-vectors and dwi files.
  -bvec Path to b-vectors file (required for dmri connectomes) [Path to b-vectors file (required for dmri connectomes) ...]
                        Specify either a path to a b-vectors text file
                        containing gradient directions (x,y,z) per diffusion
                        direction OR multiple space-separated paths to
                        multiple b-vectors text files in the order of
                        accompanying b-values and dwi files.
  -anat Path to a skull-stripped anatomical Nifti1Image [Path to a skull-stripped anatomical Nifti1Image ...]
                        Required for dmri and/or functional connectomes.
                        Multiple paths to multiple anatomical files should be
                        specified by space in the order of accompanying
                        functional and/or dmri files. If functional and dmri
                        connectomes are both being generated simultaneously,
                        then anatomical Nifti1Image file paths need to be
                        repeated, but separated by comma.
  -m Path to binarized mask Nifti1Image to apply to regions before extracting signals [Path to binarized mask Nifti1Image to apply to regions before extracting signals ...]
                        Specify either a path to a binarized brain mask
                        Nifti1Image in MNI152 space OR multiple paths to
                        multiple brain mask Nifti1Image files in the case of
                        running multiple participants, in which case paths
                        should be separated by a space. If no brain mask is
                        supplied, the template mask will be used (see
                        runconfig.yaml).
  -conf Confound regressor file (.tsv/.csv format) [Confound regressor file (.tsv/.csv format) ...]
                        Optionally specify a path to a confound regressor file
                        to reduce noise in the time-series estimation for the
                        graph. This can also be a list of paths in the case of
                        running multiple subjects, which requires separation
                        by space and of equivalent length to the list of input
                        files indicated with the -func flag.
  -g Path to graph file input. [Path to graph file input. ...]
                        In either .txt, .npy, .graphml, .csv, .ssv, .tsv, or
                        .gpickle format. This skips fMRI and dMRI graph
                        estimation workflows and begins at the thresholding
                        and graph analysis stage. Multiple graph files
                        corresponding to multiple subject ID's should be
                        separated by space, and multiple graph files
                        corresponding to the same subject ID should be
                        separated by comma. If the `-g` flag is used, then the
                        `-id` flag must also be used. Consider also including
                        `-thr` flag to activate thresholding only or the `-p`
                        and `-norm` flags if graph defragementation or
                        normalization is desired. The `-mod` flag can be used
                        for additional provenance/file-naming.
  -roi Path to binarized Region-of-Interest (ROI) Nifti1Image [Path to binarized Region-of-Interest (ROI) Nifti1Image ...]
                        Optionally specify a binarized ROI mask and retain
                        only those nodes of a parcellation contained within
                        that mask for connectome estimation.
  -ref Atlas reference file path
                        Specify the path to the atlas reference .txt file that
                        maps labels to intensities corresponding to the atlas
                        parcellation file specified with the -ua flag.
  -way Path to binarized Nifti1Image to constrain tractography [Path to binarized Nifti1Image to constrain tractography ...]
                        Optionally specify a binarized ROI mask in MNI-space
                        to constrain tractography in the case of dmri
                        connectome estimation.
  -mod Connectivity estimation/reconstruction method [Connectivity estimation/reconstruction method ...]
                        (Hyperparameter): Specify connectivity estimation
                        model. For fMRI, possible models include: corr for
                        correlation, cov for covariance, sps for precision
                        covariance, partcorr for partial correlation. sps type
                        is used by default. If skgmm is installed
                        (https://github.com/skggm/skggm), then
                        QuicGraphicalLasso, QuicGraphicalLassoCV,
                        QuicGraphicalLassoEBIC, and
                        AdaptiveQuicGraphicalLasso. Default is partcorr for
                        fMRI. For dMRI, models include csa and csd.
  -ua Path to parcellation file in MNI-space [Path to parcellation file in MNI-space ...]
                        (Hyperparameter): Optionally specify a path to a
                        parcellation/atlas Nifti1Image file in MNI152 space.
                        Labels should be spatially distinct across hemispheres
                        and ordered with consecutive integers with a value of
                        0 as the background label. If specifying a list of
                        paths to multiple user atlases, separate them by
                        space.
  -a Atlas [Atlas ...]  (Hyperparameter): Specify an atlas parcellation from
                        nilearn or local libraries. If you wish to iterate
                        your pynets run over multiple atlases, separate them
                        by space. Available nilearn atlases are: atlas_aal
                        atlas_talairach_gyrus atlas_talairach_ba
                        atlas_talairach_lobe atlas_harvard_oxford
                        atlas_destrieux_2009 atlas_msdl coords_dosenbach_2010
                        coords_power_2011 atlas_pauli_2017. Available local
                        atlases are: destrieux2009_rois
                        BrainnetomeAtlasFan2016
                        VoxelwiseParcellationt0515kLeadDBS
                        Juelichgmthr252mmEickhoff2005 CorticalAreaParcellation
                        fromRestingStateCorrelationsGordon2014
                        whole_brain_cluster_labels_PCA100
                        AICHAreorderedJoliot2015
                        HarvardOxfordThr252mmWholeBrainMakris2006
                        VoxelwiseParcellationt058kLeadDBS MICCAI2012MultiAtlas
                        LabelingWorkshopandChallengeNeuromorphometrics
                        Hammers_mithAtlasn30r83Hammers2003Gousias2008
                        AALTzourioMazoyer2002 DesikanKlein2012
                        AAL2zourioMazoyer2002
                        VoxelwiseParcellationt0435kLeadDBS AICHAJoliot2015
                        whole_brain_cluster_labels_PCA200
                        RandomParcellationsc05meanalll43Craddock2011
  -ns Spherical centroid node size [Spherical centroid node size ...]
                        (Hyperparameter): Optionally specify coordinate-based
                        node radius size(s). Default is 4 mm for fMRI and 8mm
                        for dMRI. If you wish to iterate the pipeline across
                        multiple node sizes, separate the list by space (e.g.
                        2 4 6).
  -thr Graph threshold  Optionally specify a threshold indicating a proportion
                        of weights to preserve in the graph. Default is
                        proportional thresholding. If omitted, no thresholding
                        will be applied.
  -min_thr Multi-thresholding minimum threshold
                        (Hyperparameter): Minimum threshold for multi-
                        thresholding.
  -max_thr Multi-thresholding maximum threshold
                        (Hyperparameter): Maximum threshold for multi-
                        thresholding.
  -step_thr Multi-thresholding step size
                        (Hyperparameter): Threshold step value for multi-
                        thresholding. Default is 0.01.
  -sm Smoothing value (mm fwhm) [Smoothing value (mm fwhm) ...]
                        (Hyperparameter): Optionally specify smoothing
                        width(s). Default is 0 / no smoothing. If you wish to
                        iterate the pipeline across multiple smoothing
                        separate the list by space (e.g. 2 4 6).
  -hp High-pass filter (Hz) [High-pass filter (Hz) ...]
                        (Hyperparameter): Optionally specify high-pass filter
                        values to apply to node-extracted time-series for
                        fMRI. Default is None. If you wish to iterate the
                        pipeline across multiple high-pass filter thresholds,
                        values, separate the list by space (e.g. 0.008 0.01).
  -es Node extraction strategy [Node extraction strategy ...]
                        Include this flag if you are running functional
                        connectometry using parcel labels and wish to specify
                        the name of a specific function (i.e. other than the
                        mean) to reduce the region's time-series. Options are:
                        `sum`, `mean`, `median`, `mininum`, `maximum`,
                        `variance`, `standard_deviation`.
  -k Number of k clusters [Number of k clusters ...]
                        (Hyperparameter): Specify a number of clusters to
                        produce. If you wish to iterate the pipeline across
                        multiple values of k, separate the list by space (e.g.
                        100 150 200).
  -ct Clustering type [Clustering type ...]
                        (Hyperparameter): Specify the types of clustering to
                        use. Recommended options are: ward, rena or kmeans.
                        Note that imposing spatial constraints with a mask
                        consisting of disconnected components will leading to
                        clustering instability in the case of complete,
                        average, or single clustering. If specifying a list of
                        clustering types, separate them by space.
  -cc Clustering connectivity type
                        Include this flag if you are running agglomerative-
                        type clustering (e.g. ward, average, single, complete,
                        and wish to specify a spatially constrained
                        connectivity method based on tcorr or scorr. Default
                        is allcorr which has no constraints.
  -cm Cluster mask [Cluster mask ...]
                        (Hyperparameter): Specify the path to a Nifti1Image
                        mask file to constrained functional clustering. If
                        specifying a list of paths to multiple cluster masks,
                        separate them by space.
  -ml Minimum fiber length for tracking [Minimum fiber length for tracking ...]
                        (Hyperparameter): Include this flag to manually
                        specify a minimum tract length (mm) for dmri
                        connectome tracking. Default is 20. If you wish to
                        iterate the pipeline across multiple minimums,
                        separate the list by space (e.g. 10 30 50).
  -dg Direction getter [Direction getter ...]
                        (Hyperparameter): Include this flag to manually
                        specify the statistical approach to tracking for dmri
                        connectome estimation. Options are: det
                        (deterministic), closest (clos), and prob
                        (probabilistic). Default is det. If you wish to
                        iterate the pipeline across multiple direction-getting
                        methods, separate the list by space (e.g. 'det',
                        'prob', 'clos').
  -tt Tracking algorithm
                        Include this flag to manually specify a tracking
                        algorithm for dmri connectome estimation. Options are:
                        local and particle. Default is local. Iterable
                        tracking techniques not currently supported.
  -tc Tissue classification method
                        Include this flag to manually specify a tissue
                        classification method for dmri connectome estimation.
                        Options are: cmc (continuous), act (anatomically-
                        constrained), wb (whole-brain mask), and bin (binary
                        to white-matter only). Default is bin. Iterable
                        selection of tissue classification method is not
                        currently supported.
  -s Number of samples  Include this flag to manually specify a number of
                        cumulative streamline samples for tractography.
                        Default is 100000. Iterable number of samples not
                        currently supported.
  -norm Normalization strategy for resulting graph(s)
                        Include this flag to normalize the resulting graph by
                        (1) maximum edge weight; (2) using log10; (3) using
                        pass-to-ranks for all non-zero edges; (4) using pass-
                        to-ranks for all non-zero edges relative to the number
                        of nodes; (5) using pass-to-ranks with zero-edge
                        boost; and (6) which standardizes the matrix to values
                        [0, 1]. Default is (0) which is no normalization.
  -bin                  Include this flag to binarize the resulting graph such
                        that edges are boolean and not weighted.
  -dt                   Optionally use this flag if you wish to threshold to
                        achieve a given density or densities indicated by the
                        -thr and -min_thr, -max_thr, -step_thr flags,
                        respectively.
  -mst                  Optionally use this flag if you wish to apply local
                        thresholding via the Minimum Spanning Tree approach.
                        -thr values in this case correspond to a target
                        density (if the -dt flag is also included), otherwise
                        a target proportional threshold.
  -p Pruning strategy   Include this flag to prune the resulting graph of (1)
                        any isolated + fully disconnected nodes, (2) any
                        isolated + fully disconnected + non-important nodes,
                        or (3) the larged connected component subgraph Default
                        pruning=1. Include -p 0 to disable pruning.
  -df                   Optionally use this flag if you wish to apply local
                        thresholding via the disparity filter approach. -thr
                        values in this case correspond to α.
  -mplx Perform various levels of multiplex graph analysis ONLY IF both structural and diffusion connectomes are provided.
                        Include this flag to perform multiplex graph analysis
                        across structural-functional connectome modalities.
                        Options include level (1) Create multiplex graphs
                        using motif-matched adaptive thresholding; (2)
                        Additionally perform multiplex graph embedding and
                        analysis.Default is (0) which is no multiplex
                        analysis.
  -embed                Optionally use this flag if you wish to embed the
                        ensemble(s) produced into feature vector(s).
  -spheres              Include this flag to use spheres instead of parcels as
                        nodes.
  -n Resting-state network [Resting-state network ...]
                        Optionally specify the name of any of the 2017 Yeo-
                        Schaefer RSNs (7-network or 17-network): Vis, SomMot,
                        DorsAttn, SalVentAttn, Limbic, Cont, Default, VisCent,
                        VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB,
                        SalVentAttnA, SalVentAttnB, LimbicOFC, LimbicTempPole,
                        ContA, ContB, ContC, DefaultA, DefaultB, DefaultC,
                        TempPar. If listing multiple RSNs, separate them by
                        space. (e.g. -n 'Default' 'Cont' 'SalVentAttn')'.
  -names                Optionally use this flag if you wish to perform
                        automated anatomical labeling of nodes.
  -vox {1mm,2mm}        Optionally use this flag if you wish to change the
                        resolution of the images in the workflow. Default is
                        2mm.
  -plt                  Optionally use this flag if you wish to activate
                        plotting of adjacency matrices, connectomes, and time-
                        series.
  -pm Cores,memory      Number of cores to use, number of GB of memory to use
                        for single subject run, entered as two integers
                        seperated by comma. Otherwise, default is `auto`,
                        which uses all resources detected on the current
                        compute node.
  -plug Scheduler type  Include this flag to specify a workflow plugin other
                        than the default MultiProc.
  -v                    Verbose print for debugging.
  -clean                Clean up temporary runtime directory after workflow
                        termination.
  -work Working directory
                        Specify the path to a working directory for pynets to
                        run. Default is /tmp/work.
  --version             show program's version number and exit
```

Citing
------
A manuscript is in preparation, but for now, please cite all uses with reference
to the github repository: <https://github.com/dPys/PyNets>

![Multiplex Layers](docs/_static/structural_functional_multiplex.png)
![Multiplex Glass](docs/_static/glassbrain_mplx.png)
![Yeo7](docs/_static/yeo7_mosaic.png)
![Workflow DAG](docs/_static/graph.png)
