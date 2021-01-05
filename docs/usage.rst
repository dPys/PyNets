.. include:: links.rst

#####
Usage
#####

The exact command to run ``PyNets®`` depends on several factors:

:(1): The Installation_ method (i.e. pip, docker, singularity, git), along with the environment resources available for computing;

:(2): The types and modalities of available data inputs;

:(3): The execution objective (e.g. ensemble connectome sampling, unitary connectome sampling, plotting, graph-theory, embedding, optimization/benchmarking).

***************
Required Inputs
***************

Required
========

:(A): An alphanumeric subject identifier must be specified with the `-id` flag. It can be a pre-existing label or an arbitrarily selected one, but it will be used by PyNets for naming of output directories. In the case of BIDS data, this should be `PARTICIPANT`_`SESSION`_`RUN` from sub-PARTICIPANT, ses-SESSION, run-RUN.

:(B): A supported connectivity model specified with the `-mod` flag. If PyNets is executed in multimodal mode (i.e. with both fMRI and dMRI inputs in the same command-line call), multiple modality-applicable connectivity models should be specified (minimally providing at least one for either modality). PyNets will automatically parse which model is appropriate for which data.

:(C): If an atlas is not specified with the `-a` flag, then a parcellation file must be specified with the `-a` flag. The following curated list of atlases is currently supported:

:Atlas Library:
    - 'atlas_harvard_oxford'
    - 'atlas_aal'
    - 'atlas_destrieux_2009'
    - 'atlas_talairach_gyrus'
    - 'atlas_talairach_ba'
    - 'atlas_talairach_lobe'
    - 'coords_power_2011' (only valid when using the `-spheres` flag)
    - 'coords_dosenbach_2010' (only valid when using the `-spheres` flag)
    - 'atlas_msdl'
    - 'atlas_pauli_2017'
    - 'destrieux2009_rois'
    - 'BrainnetomeAtlasFan2016'
    - 'VoxelwiseParcellationt0515kLeadDBS'
    - 'Juelichgmthr252mmEickhoff2005'
    - 'CorticalAreaParcellationfromRestingStateCorrelationsGordon2014'
    - 'AICHAreorderedJoliot2015'
    - 'HarvardOxfordThr252mmWholeBrainMakris2006'
    - 'VoxelwiseParcellationt058kLeadDBS'
    - 'MICCAI2012MultiAtlasLabelingWorkshopandChallengeNeuromorphometrics'
    - 'Hammers_mithAtlasn30r83Hammers2003Gousias2008'
    - 'AALTzourioMazoyer2002'
    - 'DesikanKlein2012'
    - 'AAL2zourioMazoyer2002'
    - 'VoxelwiseParcellationt0435kLeadDBS'
    - 'AICHAJoliot2015'
    - 'whole_brain_cluster_labels_PCA100'
    - 'whole_brain_cluster_labels_PCA200'
    - 'RandomParcellationsc05meanalll43Craddock2011'

:(D): A set of brain image files. `PyNets` is a post-processing workflow which means that input files should already be preprocessed. Minimally, all DWI, BOLD, and T1W image inputs should be **motion-corrected** (and ideally also susceptibility-corrected + denoised).

    :`anat`: The T1w can be preprocessed using any method, but should be in its native scanner anatomical space.

    :`func`: A BOLD/EPI series can be preprocessed using any method, but should in the same scanner anatomical space as the T1w (i.e. coregistered to the T1w anat and not yet normalized to a standard-space template since PyNets must do this in order that it can accurately map parcellations to individual subject anatomy).

    :`dwi`: A DWI series should ideally be in its native diffusion MRI (dMRI) space (though can also be co-registered to the T1w image) and must contain at least one B0 for reference. If `-dwi` is specified, then `-bvec` and `-bval` must also be. Note that the choice of models specified with `-mod` also depends on the sampling scheme of your dwi data (e.g. CSD will likely overfit your data in the case of too few directional volumes).

    .. note::
        Native-space DWI images are preferred for several reasons. Even when rigidly applied, intermodal registration of the diffusion signal to T1-weighted space, for instance, which has considerably different white-matter/grey-matter signal contrast (and lower specificity for the former), will inevitably result in some degree of spatial misalignment and signal loss. Note that this is unlike the case of BOLD EPI -- an inherently noisy, temporal (i.e. non-structural) modality -- which benefits from being co-registered to T1w images of significantly higher spatial resolution, particularly in grey-matter tissue where BOLD signal is typically observed. To ensure minimal within-subject variance and maximal between-subject variance as a function of numerous hyperparameters used to sample connectome ensembles with PyNets, input DWI data should ideally carry maximal SNR and have undergone the least amount of resampling necessary (e.g. minimally eddy/motion correction).

    :`-g`: A path to a raw graph can alternatively be specified, in which case the initial stages of the pipeline will be skipped. In this case, the graph should be in .txt, .npy, .csv, .tsv, or .ssv format.

    .. note::
        Prior normalization of the `anat`, `func`, or `dwi` inputs to PyNets is not (yet) supported. This is because PyNets relies on the inverse transform from an MNI-template to conform a template-resampled version of the atlas(es) specified (i.e. to define nodes) into native T1w anatomical space. PyNets uses the MNI152 template by default to accomplish this, but you can specify alternative templates in the runconfig.yml advanced settings to override MNI152 (e.g. a Pediatric template), following the naming spec of `templateflow` (See: <https://github.com/templateflow/templateflow>).

    .. note::
        If you preprocessed your BOLD data using fMRIprep, then you will need to have specified either `T1w` or `anat` in the list of fmriprep `--output-spaces`.

    .. note::
        Input image orientation and voxel resolution are not relevant, as PyNets will create necessary working copies with standardized RAS+ orientations and either 1mm or 2mm voxel resolution reslicing, depending on the runconfig.yml default or resolution override using the `-vox` flag.

    .. note::
        All file formats are assumed to be Nifti1Image (i.e. .nii or .nii.gz file suffix), and **absolute** file paths should always be specified to the CLI's.

    .. note::
        Tissue segmentations are calculated automatically in PyNets using FAST, but if you are using the `pynets_bids` CLI on preprocessed BIDS derivatives containing existing segmentations, pynets will alternatively attempt to autodetect and use those.

Custom File Inputs
==================

:`-m`: (*fMRI + dMRI*) A binarized brain mask of the T1w image in its native anatomical space. Input images need not be skull-stripped. If brain masking has been applied already, `PyNets` will attempt to detect this, else it will attempt to extract automatically using a deep-learning classifier. See [deepbrain]<https://github.com/iitzco/deepbrain> for more information.

:`-roi`: (*fMRI + dMRI*) A binarized ROI mask used to constrain connectome node-making to restricted brain regions of the parcellation being used. ROI inputs should be in MNI space.

:`-a`: (*fMRI + dMRI*) A parcellation/atlas image (in MNI space) used to define nodes of a connectome. Labels should be spatially distinct across hemispheres and ordered with consecutive integers with a value of 0 as the background label. This flag can uniquely be listed with multiple, space-separated file inputs.

:`-ref`: (*fMRI + dMRI*) An atlas reference .txt file that indices intensities corresponding to atlas labels of the parcellation specified with the `-a` flag. This label map is used only to delineate node labels manually. Otherwise, PyNets will attempt to perform automated node labeling via AAL, else sequential numeric labels will be used.

:`-way`: (*dMRI*) A binarized white-matter ROI mask (in MNI template space) used to constrain tractography in native diffusion space such that streamlines are retained only if they pass within the vicinity of the mask. Like with ROI inputs, waymasks should be in MNI space.

:`-cm`: (*fMRI*) A binarized ROI mask used to spatially-constrained clustering during parcellation-making. Note that if this flag is used, `-k` and `-ct` must also be included. Like with ROI inputs, clustering masks should be in MNI space.

:`-conf`: (*fMRI*) An additional noise confound regressor file for extracting a cleaner time-series.


Multimodal Workflow Variations
==============================

In the case of running pynets on a single subject, several combinations of input files can be used:

:fMRI Connectometry: `-func`, `-anat`, (`-conf`), (`-roi`), (`-m`), (`-cm`)

:dMRI Connectometry: `-dwi`, `-bval`, `-bvec`, `-anat`, (`-roi`), (`-m`), (`-way`)

:dMRI + fMRI Multiplex Connectometry: All of the above required flags should be included simultaneously. Note that in this case, `-anat` only needs to be specified once.

:Raw Graph Connectometry (i.e. for graph analysis/embedding only): `-g`

**********************
Command-Line Arguments
**********************

.. argparse::
    :module: pynets.cli.pynets_run
    :func: get_bids_parser
    :prog: pynets

.. argparse::
    :module: pynets.cli.pynets_bids
    :func: get_parser
    :prog: pynets

**********
Quickstart
**********

Execution on BIDS derivative datasets using the `pynets_bids` CLI
=================================================================

PyNets now includes an API for running single-subject and group workflows on BIDS derivatives (e.g. produced using popular BIDS apps like fmriprep/cpac and dmriprep/qsiprep).
In this scenarioo, the input dataset should follow the derivatives specification of the `BIDS (Brain Imaging Data Structure)` format (<https://bids-specification.readthedocs.io/en/derivatives/05-derivatives/01-introduction.html>), which must include at least one subject's fMRI image or dMRI image (in T1w space), along with a T1w anatomical image.

The `runconfig.yml` file in the base directory includes parameter presets, but all file input options that are included with the `pynets` cli are also exposed to the `pynets_bids` cli.

The common parts of the command follow the `BIDS-Apps <https://github.com/BIDS-Apps>`_ definition.
Example: ::

    pynets_bids '/hnu/fMRIprep/fmriprep' '/Users/dPys/outputs/pynets' participant func --participant_label 0025427 0025428 --session_label 1 2 3 -config pynets/config/bids_config.json

A similar CLI, `pynets_cloud` has also been made available using AWS Batch and S3, which require a AWS credentials and configuration of job queues and definitions using cloud_config.json: ::

    pynets_cloud --bucket 'hnu' --dataset 'HNU' participant func --participant_label 0025427 --session_label 1 --push_location 's3://hnu/outputs' --jobdir '/Users/derekpisner/.pynets/jobs' -cm 's3://hnu/HNU/masks/MyClusteringROI.nii.gz' -pm '30,110'


Manual Execution Using the `pynets` CLI
=======================================

You have a preprocessed EPI bold dataset from the first session for subject 002, and you wish to analyze a whole-brain network using 'sub-colin27_label-L2018_desc-scale1_atlas', thresholding the connectivity graph proportionally to retain 0.20% of the strongest connections, and you wish to use partial correlation model estimation: ::

    pynets -id '002_1' '/Users/dPys/outputs/pynets' \
    -func '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/func/BOLD_PREPROCESSED_IN_ANAT_NATIVE.nii.gz' \ # The fMRI BOLD image data.
    -anat '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/anat/ANAT_PREPROCESSED_NATIVE.nii.gz' \ # The T1w anatomical image.
    -a 'sub-colin27_label-L2018_desc-scale1_atlas' \ # Lausanne parcellation at scale=1.
    -mod 'partcorr' \ # The connectivity model.
    -thr 0.20 \ # A single proportional threshold to apply post-hoc.

Building upon the previous example, let's say you now wish to analyze the Default network for this same subject's data, but based on the 95-node atlas parcellation scheme from Desikan-Klein 2012 called 'DesikanKlein2012' and the Brainnetome Atlas from Fan 2016 called 'BrainnetomeAtlasFan2016', you wish to threshold the graph to achieve a target density of 0.3, and you wish to fit a sparse inverse covariance model in addition to partial correlation, and you wish to plot the results: ::

    pynets -id '002_1' '/Users/dPys/outputs/pynets' \
    -func '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/func/BOLD_PREPROCESSED_IN_ANAT_NATIVE.nii.gz' \ # The fMRI BOLD image data.
    -anat '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/anat/ANAT_PREPROCESSED_NATIVE.nii.gz' \ # The T1w anatomical image.
    -a 'DesikanKlein2012' 'BrainnetomeAtlasFan2016' # Multiple spherical atlases.
    -mod 'partcorr' 'sps' \ # The connectivity models.
    -dt -thr 0.3 \ # The thresholding settings.
    -n 'Default' \ # The resting-state network definition to restrict node-making from each of the input atlas.
    -plt # Activate plotting.

Building upon the previous examples, let's say you now wish to analyze the Default and Executive Control Networks for this subject, but this time based on a custom atlas (DesikanKlein2012.nii.gz), this time defining your nodes as parcels (as opposed to spheres), you wish to fit a partial correlation model, you wish to iterate the pipeline over a range of densities (i.e. 0.05-0.10 with 1% step), and you wish to prune disconnected nodes: ::

    pynets -id '002_1' '/Users/dPys/outputs/pynets' \
    -func '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/func/BOLD_PREPROCESSED_IN_ANAT_NATIVE.nii.gz' \ # The fMRI BOLD image data.
    -anat '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/anat/ANAT_PREPROCESSED_NATIVE.nii.gz' \ # The T1w anatomical image.
    -a '/Users/dPys/PyNets/pynets/atlases/MyCustomAtlas.nii.gz' \ # A user-supplied atlas parcellation.
    -mod 'partcorr' \ # The connectivity model.
    -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 \ # The thresholding settings.
    -n 'Default' 'Cont' # The resting-state network definitions to restrict node-making from each of the input atlas.

.. note::
    In general, parcels are preferable to spheres as nodes because parcels more closely respect cortical topographgy.

Building upon the previous examples, let's say you now wish to create a subject-specific atlas based on the subject's unique spatial-temporal profile. In this case, you can specify the path to a binarized mask within which to performed spatially-constrained spectral clustering, and you want to try this at multiple resolutions of k clusters/nodes (i.e. k=50,100,150). You again also wish to define your nodes spherically with radii at both 2 and 4 mm, fitting a partial correlation and sparse inverse covariance model, you wish to iterate the pipeline over a range of densities (i.e. 0.05-0.10 with 1% step), you wish to prune disconnected nodes, and you wish to plot your results: ::

    pynets -id '002_1' '/Users/dPys/outputs/pynets' \
    -func '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/func/BOLD_PREPROCESSED_IN_ANAT_NATIVE.nii.gz' \ # The fMRI BOLD image data.
    -anat '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/anat/ANAT_PREPROCESSED_NATIVE.nii.gz' \ # The T1w anatomical image.
    -mod 'partcorr' 'sps' \ # The connectivity models.
    -cm '/Users/dPys/PyNets/tests/examples/MyClusteringROI.nii.gz' -k 50 100 150 -ct 'ward' \ # Node-making specification with spatially-constrained clustering.
    -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 \ # The thresholding settings.
    -plt # Activate plotting.

You wish to generate a structural connectome, using deterministic and probabilistic ensemble tractography, based on both constrained-spherical deconvolution (CSD), Constant Solid Angle (CSA), and Sparse Fascicle (SFM) models. You wish to use atlas parcels as defined by both DesikanKlein2012, and AALTzourioMazoyer2002, exploring only those nodes belonging to the Default Mode Network, iterate over a range of graph densities (i.e. 0.05-0.10 with 1% step), and prune disconnected nodes: ::

    pynets -id '002_1' '/Users/dPys/outputs/pynets' \
    -dwi '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/dwi/DWI_PREPROCESSED_NATIVE.nii.gz' \ # The dMRI diffusion-weighted image data.
    -bval '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/dwi/BVAL.bval' \ # The b-values.
    -bvec '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/dwi/BVEC.bvec' \ # The b-vectors.
    -anat '/Users/dPys/PyNets/tests/examples/sub-002/ses-1/anat/ANAT_PREPROCESSED_NATIVE.nii.gz' \ # The T1w anatomical image.
    -a '/Users/dPys/.atlases/DesikanKlein2012.nii.gz' '/Users/dPys/.atlases/AALTzourioMazoyer2002.nii.gz' \ # The atlases.
    -mod 'csd' 'csa' 'sfm' \ # The connectivity model.
    -dg 'prob' 'det'  \ # The tractography settings.
    -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 \ # The thresholding settings.
    -n 'Default' # The resting-state network definition to restrict node-making from each of the input atlases.

.. note::
    Spherical nodes can be used by triggering the `-spheres` flag, and for some coordinate-based atlases like coords_power_2011 or coords_dosenbach_2010, only spheres are possible, but in general parcel volumes should be used as the default.

.. note::
    Iterable sampling parameters specified at runtime should always be space-delimited.

There are many other runtime options than these examples demonstrate. To explore all of the possible hyper-parameter combinations that pynets has to offer, see `pynets -h`. A full set of tutorials and python notebooks are coming soon.

Docker and AWS
==============

PyNets includes an API for running `pynets_bids` or `pynets` in a Docker container as well as using AWS Batch. The latter assumes a dataset with BIDS derivatives is stored in an S3 bucket.
Docker Example: ::

    docker run -ti --rm --privileged -v '/home/dPys/.aws/credentials:/home/neuro/.aws/credentials' dpys/pynets:latest pynets_bids 's3://hnu/HNU' '/outputs' participant func --participant_label 0025427 --session_label 1 -plug 'MultiProc' -pm '8,12' -work '/working' -config pynets/config/bids_config.json

Running a Singularity Image
===========================

If the data to be preprocessed is also on an HPC server, you are ready to run pynets, either manually or as a BIDS application.
For example, where PARTICIPANT is a subject identifier and SESSION is a given scan session, we could sample an ensemble of connectomes manually as follows ::

    singularity exec -w \
     '/scratch/04171/dPys/pynets_singularity_latest-2020-02-07-eccf145ea766.img' \
     pynets /outputs \
     -p 1 -mod 'partcorr' 'corr' -min_thr 0.20 -max_thr 1.00 -step_thr 0.10 -sm 0 2 4 -hp 0 0.028 0.080
     -ct 'ward' -k 100 200 -cm '/working/MyClusteringROI.nii.gz' \
     -pm '24,48' \
     -norm 6 \
     -anat '/inputs/sub-PARTICIPANT/ses-SESSION/anat/sub-PARTICIPANT_space-anat_desc-preproc_T1w_brain.nii.gz' \
     -func '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_space-anat_desc-smoothAROMAnonaggr_bold_masked.nii.gz' \
     -conf '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_desc-confounds_regressors.tsv' \
     -m '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_space-anat_desc-brain_mask.nii.gz' \
     -id 'PARTICIPANT_SESSION' -plug 'MultiProc' -work '/tmp'

.. note::

   Singularity by default `exposes all environment variables from the host inside
   the container <https://github.com/singularityware/singularity/issues/445>`_.
   Because of this your host libraries (such as nipype) could be accidentally used
   instead of the ones inside the container - if they are included in ``PYTHONPATH``.
   To avoid such situation we sometimes recommend using the ``--cleanenv`` singularity flag
   in production use. For example: ::

      singularity exec --cleanenv --no-home_clust_est '/scratch/04171/dPys/pynets_latest-2016-12-04-5b74ad9a4c4d.img' \
        pynets /outputs \
        -p 1 -mod 'partcorr' 'corr' -min_thr 0.20 -max_thr 1.00 -step_thr 0.10 -sm 0 2 4 -hp 0 0.028 0.080
        -ct 'ward' -k 100 200 -cm '/working/MyClusteringROI.nii.gz' \
        -norm 6 \
        -anat '/inputs/sub-PARTICIPANT/ses-SESSION/anat/sub-PARTICIPANT_space-anat_desc-preproc_T1w_brain.nii.gz' \
        -func '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_space-anat_desc-smoothAROMAnonaggr_bold_masked.nii.gz' \
        -conf '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_desc-confounds_regressors.tsv' \
        -m '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_space-anat_desc-brain_mask.nii.gz' \
        -id 'PARTICIPANT_SESSION' -plug 'MultiProc' -work '/tmp' -pm '24,48'

   or, unset the ``PYTHONPATH`` variable before running: ::

      unset PYTHONPATH; singularity exec /scratch/04171/dPys/pynets_latest-2016-12-04-5b74ad9a4c4d.img \
        pynets /outputs \
        -p 1 -mod 'partcorr' 'corr' -min_thr 0.20 -max_thr 1.00 -step_thr 0.10 -sm 0 2 4 -hp 0 0.028 0.080
        -ct 'ward' -cm '/working/MyClusteringROI.nii.gz' -k 100 200 \
        -norm 6 \
        -anat '/inputs/sub-PARTICIPANT/ses-SESSION/anat/sub-PARTICIPANT_space-anat_desc-preproc_T1w_brain.nii.gz' \
        -func '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_space-anat_desc-smoothAROMAnonaggr_bold_masked.nii.gz' \
        -conf '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_desc-confounds_regressors.tsv' \
        -m '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_space-anat_desc-brain_mask.nii.gz' \
        -id 'PARTICIPANT_SESSION' -plug 'MultiProc' -work '/tmp' -pm '24,48'

.. note::

   Depending on how Singularity is configured on your cluster it might or might not
   automatically bind (mount or expose) host folders to the container.
   If this is not done automatically you will need to bind the necessary folders using
   the ``-B <host_folder>:<container_folder>`` Singularity argument.
   For example: ::

      singularity exec_clust_est -B /work:/work /scratch/04171/dPys/pynets_latest-2016-12-04-5b74ad9a4c4d.img \
        -B '/scratch/04171/dPys/pynets_out:/inputs,/scratch/04171/dPys/masks/PARTICIPANT_triple_network_masks_SESSION':'/outputs' \
        pynets /outputs \
        -p 1 -mod 'partcorr' 'corr' -min_thr 0.20 -max_thr 1.00 -step_thr 0.10 -sm 0 2 4 -hp 0 0.028 0.080 \
        -ct 'ward' -k 100 200 -cm '/working/MyClusteringROI.nii.gz' \
        -norm 6 \
        -anat '/inputs/sub-PARTICIPANT/ses-SESSION/anat/sub-PARTICIPANT_space-anat_desc-preproc_T1w_brain.nii.gz' \
        -func '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_space-anat_desc-smoothAROMAnonaggr_bold_masked.nii.gz' \
        -conf '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_desc-confounds_regressors.tsv' \
        -m '/inputs/sub-PARTICIPANT/ses-SESSION/func/sub-PARTICIPANT_ses-SESSION_task-rest_space-anat_desc-brain_mask.nii.gz' \
        -id 'PARTICIPANT_SESSION' -plug 'MultiProc' -work '/tmp'  -pm '24,48'

*********
Debugging
*********

Logs and crashfiles are outputted into the ``<working dir>/Wf_single_subject_<id>`` directory. To include verbose debugging and resource benchmarking, run pynets with the `-v` flag.


*************************
Support and communication
*************************

The documentation of this project is found here: http://pynets.readthedocs.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/dPys/PyNets/issues.

If you have a problem or would like to ask a question about how to use ``pynets``,
please submit a question to `NeuroStars.org <http://neurostars.org/tags/pynets>`_ with an ``pynets`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

All previous ``pynets`` questions are available here:
http://neurostars.org/tags/pynets/

To participate in the ``pynets`` development-related discussions please use the following mailing list: http://mail.python.org/mailman/listinfo/neuroimaging
Please add *[pynets]* to the subject line when posting on the mailing list.


***********************************************
Not running on a local machine? - Data transfer
***********************************************

If you intend to run ``pynets`` on a remote system, you will need to make your data available within that system first.

Alternatively, more comprehensive solutions such as `Datalad <http://www.datalad.org/>`_ will handle data transfers with the appropriate settings and commands. Datalad also performs version control over your data.


********************
Interpreting outputs
********************

To collect the graph topological outputs from one or more completed pynets
runs, you can use the `pynets_collect` CLI: ::

    pynets_collect -basedir '/Users/dPys/outputs/pynets' -modality 'func'

which will generate a group summary dataframe in `basedir`, all_subs_neat.csv, where each row is a
given subject session and/or run, and each column is a graph topological
metric the was calculated, with the prefix indicating correspondence to a
given connectome ensemble of interest.
