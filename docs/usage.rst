.. include:: links.rst

#####
Usage
#####

The exact command to run ``PyNets`` depends on several factors:

:(1): The Installation_ method (i.e. pip, docker, singularity, AWS), along with the environment resources available for computing;

:(2): The types and modalities of available data inputs;

:(3): The execution objective (e.g. ensemble connectome sampling, unitary connectome sampling, plotting, graph-analysis, embedding, optimization, or other, derivative applications).

****************
Data Assumptions
****************

`PyNets` is a post-processing workflow which means that input files should already be preprocessed. That is, for both dMRI and fMRI modalities of images, both dMRI and fMRI should be:

(1) Pre-scrubbed for common sources of noise (mandatory, albeit a noise confound regressor file can optionally be supplied via the `-conf` flag).

(2) **All** input image data should be skull-stripped, *except* in the case that a binary brain mask image is also supplied via the `-m` flag.

(3) Image space matters:

    :fMRI: Inputs should be normalized to Montreal Neurological Institute (MNI) space. In functional BOLD imaging, normalization helps to stabilize the underlying connectivity time-series and thereby morphologically standardize functional connectome estimates for optimal discriminability of individual differences.

    :dMRI: Inputs should be left in native diffusion/scanner space and **not** normalized using a T1w image or template. This is because tractography, upon which structural connectome estimates are based is most reliably performed in native space -- with limited resampling or geometric distortion of spatial information beyond that which is minimally needed in preprocessing (e.g. to correct for head motion or eddy currents). To nevertheless ensure comparability across individuals, PyNets will still perform normalization, but at a later stage in the connectome estimation process. That is, after reconstructio with tractography, resulting streamlines will be directly mapped into MNI-space via a rigid transformation of streamline points.

    :T1w: Should be in native anatomical space, as it will be segmented (a native-space operation) and subsequently re-normalized to MNI-space automatically.


***********
File Inputs
***********

Primary File Inputs
===================

In the case of a single subject, several combinations of input files can be used:

:fMRI: `-func` (required); `-conf`, `-m`, `-anat` (optional)

:dMRI: `-dwi`, `-bval`, `-bvec` (required); `-m`, `-anat` (optional)

:dMRI + fMRI: all of the above flags still apply, but should be used simultaneoously. `-m`, `-anat` only need to be specified once.

:Raw Graph: `-g`

.. note::
    All formats are assumed to be Nifti1Image (i.e. .nii or .nii.gz file suffix), except for a raw graph which can be in .txt, .npy, .csv, .tsv, or .ssv.

.. note::
    T1w input images should be skull-stripped and un-resampled (i.e. *not* normalized to MNI).

Secondary File Inputs
=====================

:`-way`: (*dMRI*) A binarized mask used to constrain tractography such that streamlines are retained only if they pass within the vicinity of the mask.

:`-roi`: (*fMRI + dMRI*) A binarized mask used to constrain connectome node-making to restricted brain regions of interest (ROI's).

:`-ua`: (*fMRI + dMRI*) A parcellation/atlas image used to define nodes of a connectome. Labels should be spatially distinct across hemispheres and ordered with consecutive integers with a value of 0 as the background label. This flag can uniquely be listed with multiple, space-separated file inputs.

:`-ref`: (*fMRI + dMRI*) An atlas reference .txt file that indices intensities corresponding to atlas labels of the parcellation specified with the `-ua` flag. This label map is used only to delineate node labels manually. Automated node labeling via AAL can alternatively be used by including the `-names` flag. Otherwise, sequential numeric labels will be used by default.

:`-templ`: (*fMRI + dMRI*) A template image to override normalization in place of the MNI152 template.

:`-templm`: (*fMRI + dMRI*) A template image mask to override mask normalization in place of the MNI152 mask template.

.. note::
    All general image inputs are assumed to be normalized to MNI space. Image orientation and voxel resolution are not relevant, as PyNets will create necessary working copies with standardized RAS+ orientations and either 1mm or 2mm voxel resolution reslicing, depending on that which is specified with the `-vox` flag.

BIDS Derivatives
================

PyNets now includes an API for running single-subject and group workflows on BIDS derivatives (e.g. produced using popular BIDS apps like fmriprep/cpac and dmriprep/qsiprep).
In this case, the input dataset should be in `BIDS (Brain Imaging Data Structure)` format, and it must include at least one fMRI image or dMRI image.

The `runconfig.yml` file in the base directory includes parameter presets, but all file input options that are included with the `pynets` cli are also exposed to the `pynets_bids` cli.

The common parts of the command follow the `BIDS-Apps <https://github.com/BIDS-Apps>`_ definition.
Example: ::

    pynets_bids 's3://hnu/HNU' '~/outputs' func --participant_label 0025427 --session_label 1 --push_location 's3://hnu/outputs' -cm 's3://hnu/HNU/masks/0025427_triple_network_masks_1/triple_net_ICA_overlap_9_sig_bin.nii.gz'

Docker and AWS
==============

PyNets now includes an API for running pynets_bids in a Docker container as well as using AWS Batch. The latter assumes a dataset with BIDS derivatives is stored in an S3 bucket.
Docker Example: ::

    docker run -ti --rm --privileged -v '~/.aws/credentials:/home/neuro/.aws/credentials' dpys/pynets:latest 's3://hnu/HNU' '/outputs' func --participant_label 0025427 --session_label 1 --push_location 's3://hnu/outputs' -cm 's3://hnu/HNU/masks/0025427_triple_network_masks_1/triple_net_ICA_overlap_9_sig_bin.nii.gz' -plug 'MultiProc' -pm '8,12' -work '/working'

AWS Batch Example: ::

    pynets_cloud --bucket 'hnu' --dataset 'HNU' func --participant_label 0025427 --session_label 1 --push_location 's3://hnu/outputs' --jobdir '/Users/derekpisner/.pynets/jobs' -cm 's3://hnu/HNU/masks/0025427_triple_network_masks_1/triple_net_ICA_overlap_9_sig_bin.nii.gz' -pm '30,110'

*****************
Parametric Inputs
*****************

Required
========

:(A): An alphanumeric subject identifier must be specified with the `-id` flag. It can be a pre-existing label or an arbitrarily selected one, but it will be used by PyNets for naming of output directories.

:(B): A supported connectivity model specified with the `-mod` flag. See `pynets --help` for supported fMRI and dMRI connectivity models. If PyNets is executed in multimodal mode (i.e. with both fMRI and dMRI inputs in the same command-line call), multiple modality-applicable connectivity models should be specified (minimally providing at least one for either modality). PyNets will automatically parse which model is appropriate for which data.

:(C): If a parcellation file is not specified with the `-ua` flag, then the `-a` flag must be included, followed by one or more supported atlases. See `pynets --help` to view a list of available atlases.


**********************
Command-Line Arguments
**********************
.. argparse::
   :ref: cli.pynets_run.get_parser
   :prog: pynets
   :nodefault:
   :nodefaultconst:


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


**********
Quickstart
**********

Examples
========


You have a preprocessed (minimally -- normalized and skull stripped) functional fMRI dataset called "002.nii.gz" where you assign an arbitrary subject id of 002, you wish to analyze a whole-brain network, using the nilearn atlas 'coords_dosenbach_2010', thresholding the connectivity graph proportionally to retain 0.20% of the strongest connections, and you wish to use partial correlation model estimation: ::

    pynets -id '002' '/Users/dPys/outputs' \
    -func '/Users/dPys/PyNets/tests/examples/002/fmri/002.nii.gz' \ # The fMRI BOLD image data.
    -a 'coords_dosenbach_2010' \ # A spherical atlas.
    -mod 'partcorr' \ # The connectivity model.
    -spheres # Node-making specification.
    -thr 0.20 \ # A single proportional threshold to apply post-hoc.
    -m '/Users/dPys/PyNets/tests/examples/002/fmri/002_mask.nii.gz' # A brain mask for the fMRI BOLD image data.

Building upon the previous example, let's say you now wish to analyze the Default network for this same subject's data, but now also using the 264-node atlas parcellation scheme from Power et al. 2011 called 'coords_power_2011', you wish to threshold the graph to achieve a target density of 0.3, and you define your nodes based on spheres with radii at two resolutions (2 and 4 mm), you wish to fit a sparse inverse covariance model in addition to partial correlation, and you wish to plot the results: ::

    pynets -id '002' '/Users/dPys/outputs' \
    -func '/Users/dPys/PyNets/tests/examples/002/fmri/002.nii.gz' \ # The fMRI BOLD image data.
    -a 'coords_dosenbach_2010' 'coords_power_2011' # Multiple spherical atlases.
    -mod 'partcorr' 'sps' \ # The connectivity models.
    -ns 2 4 -spheres \ # Node-making specification.
    -dt -thr 0.3 \ # The thresholding settings.
    -n 'Default' \ # The resting-state network definition to restrict node-making from each of the input atlas.
    -plt # Activate plotting.

Building upon the previous examples, let's say you now wish to analyze the Default and Executive Control Networks for this subject, but this time based on a custom atlas (DesikanKlein2012.nii.gz), this time defining your nodes as parcels (as opposed to spheres), you wish to fit a partial correlation model, you wish to iterate the pipeline over a range of densities (i.e. 0.05-0.10 with 1% step), and you wish to prune disconnected nodes: ::

    pynets -id '002' '/Users/dPys/outputs' \
    -func '/Users/dPys/PyNets/tests/examples/002/fmri/002.nii.gz' \ # The fMRI BOLD image data.
    -ua '/Users/dPys/PyNets/pynets/atlases/DesikanKlein2012.nii.gz' \ # A user-supplied atlas parcellation.
    -mod 'partcorr' \ # The connectivity model.
    -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 \ # The thresholding settings.
    -n 'Default' 'Cont' # The resting-state network definitions to restrict node-making from each of the input atlas.

.. note::
    In general, parcels are preferable to spheres as nodes because parcels more closely respect atlas or cluster topology.

Building upon the previous examples, let's say you now wish to create a subject-specific atlas based on the subject's unique spatial-temporal profile. In this case, you can specify the path to a binarized mask within which to performed spatially-constrained spectral clustering, and you want to try this at multiple resolutions of k clusters/nodes (i.e. k=50,100,150). You again also wish to define your nodes spherically with radii at both 2 and 4 mm, fitting a partial correlation and sparse inverse covariance model, you wish to iterate the pipeline over a range of densities (i.e. 0.05-0.10 with 1% step), you wish to prune disconnected nodes, and you wish to plot your results: ::

    pynets -id '002' '/Users/dPys/outputs' \
    -func '/Users/dPys/PyNets/tests/examples/002/fmri/002.nii.gz' \ # The fMRI BOLD image data.
    -mod 'partcorr' 'sps' \ # The connectivity models.
    -cm '/Users/dPys/PyNets/tests/examples/pDMN_3_bin.nii.gz' -k 50 100 150 -ct 'ward' \ # Node-making specification with spatially-constrained clustering.
    -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 \ # The thresholding settings.
    -plt -names # Activate plotting with automated node labeling by coordinate reference.

You wish to generate a structural connectome, using probabilistic ensemble tractography with 1,000,000 streamlines, based on both constrained-spherical deconvolution (csd) and tensor models, bootstrapped tracking, and direct normalization of streamlines. You wish to use atlas parcels as defined by both DesikanKlein2012, and AALTzourioMazoyer2002, exploring only those nodes belonging to the Default Mode Network, and iterate over a range of densities (i.e. 0.05-0.10 with 1% step), and prune disconnected nodes: ::

    pynets -id '0021001' '/Users/dPys/outputs' \
    -dwi '/Users/dPys/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz' \ # The dMRI diffusion-weighted image data.
    -bval '/Users/dPys/PyNets/tests/examples/002/dmri/bval.bval' \ # The b-values.
    -bvec '/Users/dPys/PyNets/tests/examples/002/dmri/bvec.bvec' \ # The b-vectors.
    -ua '~/.atlases/DesikanKlein2012.nii.gz' '~/.atlases/AALTzourioMazoyer2002.nii.gz' \ # The atlases.
    -mod 'csd' \ # The connectivity model.
    -dg 'prob' 'det' 'tensor' -s 1000000  \ # The tractography settings.
    -anat '/Users/dPys/PyNets/tests/examples/002/anat/s002_anat_brain.nii.gz' \ # The T1w anatomical image.
    -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 \ # The thresholding settings.
    -n 'Default' # The resting-state network definition to restrict node-making from each of the input atlases.

.. note::
    Spherical nodes can be used by triggering the `-spheres` flag, but this approach is **not** recommended for dMRI connectometry.

.. note::
    Iterable sampling parameters specified at runtime should always be space-delimited and, to be safe, contained within single quotes.

There are many other runtime options than these examples demonstrate. To explore all of the possible hyper-parameter combinations that pynets has to offer, see `pynets -h`. A full set of tutorials and python notebooks are coming soon.


********************
Interpreting outputs
********************
(IN CONSTRUCTION)
