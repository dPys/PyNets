.. include:: links.rst

Usage
-----

Execution
=========

The ``PyNets`` workflow takes as principal input the path to an fMRI image
(following the `-func` flag), dMRI image (following the `-dwi` flag),
or pre-generated graphs (following the `-g` flag) which may be stored in any of
a variety of formats (e.g. .npy, .ssv, .txt, .csv, .mat).

In the case of a single subject, these paths may be singular, whereas in the
case of multiple subjects, they should be comma-separated and in consistent
order corresponding to subject id as defined following the `-id` flag.
fMRI files should be preprocessed (minimally normalization to MNI-space,
(e.g. see https://github.com/poldracklab/fmriprep). dMRI files should also be
preprocessed (minimally some form of motion/eddy correction,
e.g. see https://github.com/PennBBL/qsiprep). If a dMRI image is specified,
accompanying .bval and .bvec files are required as inputs as well following
`-bval` and `-bvec` flags, respectively. A T1/T2-weighted anatomical image
(indicated following `-anat`) is also required as input in the case that a dMRI
image is used, but not in the case that an fMRI image is used (though its
inclusion is highly recommended, along with a brain mask indicated following
`-m`).

BIDS Derivatives
================
(in construction)
PyNets includes an API for running single-subject and group workflows on BIDS
derivatives (e.g. produced using popular BIDS apps like fmriprep and qsiprep).
In this case, the input dataset is required to be in valid :abbr:`BIDS (Brain
Imaging Data Structure)` derivative format, and it must include at least one
fMRI image or dMRI image.

The exact command to run ``PyNets`` depends on the Installation_ method.
The common parts of the command follow the `BIDS-Apps
<https://github.com/BIDS-Apps>`_ definition.
Example: ::
    pynets_bids.py data/bids_derivative_root/ participant -w work/


Command-Line Arguments
======================

.. argparse::
   :ref: pynets_run.get_parser
   :prog: pynets
   :nodefault:
   :nodefaultconst:


Debugging
=========

Logs and crashfiles are outputted into the
``<output dir>/sub-<id>/Wf_single_subject_<id>`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

Support and communication
=========================

The documentation of this project is found here: http://fmriprep.readthedocs.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/dPys/PyNets/issues.

If you have a problem or would like to ask a question about how to use ``pynets``,
please submit a question to `NeuroStars.org <http://neurostars.org/tags/pynets>`_ with an ``pynets`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

All previous ``pynets`` questions are available here:
http://neurostars.org/tags/pynets/

To participate in the ``pynets`` development-related discussions please use the
following mailing list: http://mail.python.org/mailman/listinfo/neuroimaging
Please add *[pynets]* to the subject line when posting on the mailing list.


Not running on a local machine? - Data transfer
===============================================

If you intend to run ``pynets`` on a remote system, you will need to
make your data available within that system first.

Alternatively, more comprehensive solutions such as `Datalad
<http://www.datalad.org/>`_ will handle data transfers with the appropriate
settings and commands.
Datalad also performs version control over your data.

Quickstart
===============================================
Example A) You have a preprocessed (minimally -- normalized and skull stripped) functional fMRI dataset called "002.nii.gz" where you assign an arbitrary subject id of 997, you wish to analyze a whole-brain network, using the nilearn atlas 'coords_dosenbach_2010', thresholding the connectivity graph proportionally to retain 0.20% of the strongest connections, and you wish to use partial correlation model estimation: ::
pynets_run.py -func '/Users/dPys/PyNets/tests/examples/002/fmri/002.nii.gz' -id '002' -a 'coords_dosenbach_2010' -mod 'partcorr' -thr 0.20

Example B) Building upon the previous example, let's say you now wish to analyze the Default network for this same dataset, but now also using the 264-node atlas parcellation scheme from Power et al. 2011 called 'coords_power_2011', you wish to threshold the graph iteratively to achieve a target density of 0.3, and you define your node radius at two resolutions (2 and 4 mm), you wish to fit a  sparse inverse covariance model in addition to partial correlation, and you wish to plot the results: ::
pynets_run.py -func '/Users/dPys/PyNets/tests/examples/002/fmri/002.nii.gz' -id '002' -a 'coords_dosenbach_2010,coords_power_2011' -n 'Default' -dt -thr 0.3 -ns 2 4 -mod 'partcorr' 'sps' -plt

Example C) Building upon the previous examples, let's say you now wish to analyze the Default and Executive Control Networks for this subject, but this time based on a custom atlas (DesikanKlein2012.nii.gz), this time defining your nodes as parcels (as opposed to spheres), you wish to fit a partial correlation model, you wish to iterate the pipeline over a range of densities (i.e. 0.05-0.10 with 1% step), and you wish to prune disconnected nodes: ::
pynets_run.py -func '/Users/dPys/PyNets/tests/examples/002/fmri/002.nii.gz' -id '002' -ua '/Users/dPys/PyNets/pynets/atlases/DesikanKlein2012.nii.gz' -n 'Default' 'Cont' -mod 'partcorr' -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -parc -p 1

Example D) Building upon the previous examples, let's say you now wish to create a subject-specific atlas based on the subject's unique spatial-temporal profile. In this case, you can specify the path to a binarized mask within which to performed spatially-constrained spectral clustering, and you want to try this at multiple resolutions of k clusters/nodes (i.e. k=50,100,150). You again also wish to define your node radius at both 2 and 4 mm, fitting a partial correlation and sparse inverse covariance model, you wish to iterate the pipeline over a range of densities (i.e. 0.05-0.10 with 1% step), you wish to prune disconnected nodes, and you wish to plot your results: ::
pynets_run.py -func '/Users/dPys/PyNets/tests/examples/002/fmri/002.nii.gz' -id '002' -cm '/Users/dPys/PyNets/tests/examples/pDMN_3_bin.nii.gz' -ns 2 4 -mod 'partcorr' 'sps' -k_min 50 -k_max 150 -k_step 50 -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 -plt

Example E) You wish to generate a structural connectome, using probabilistic ensemble tractography with 1,000,000 streamlines, based on both constrained-spherical deconvolution (csd) and tensor models, filtering with linear fascicle evaluation (LiFE), and direct normalization of streamlines. You wish to use atlas parcels as defined by both DesikanKlein2012, and AALTzourioMazoyer2002, exploring only those nodes belonging to the Default Mode Network, and iterate over a range of densities (i.e. 0.05-0.10 with 1% step), and prune disconnected nodes: ::
pynets_run.py -dwi '/Users/dPys/PyNets/tests/examples/002/dmri/iso_eddy_corrected_data_denoised.nii.gz' -bval '/Users/dPys/PyNets/tests/examples/002/dmri/bval.bval' -bvec '/Users/dPys/PyNets/tests/examples/002/dmri/bvec.bvec' -id 0021001 -ua '/Users/dPys/PyNets/pynets/atlases/DesikanKlein2012.nii.gz,/Users/dPys/PyNets/pynets/atlases/AALTzourioMazoyer2002' -parc -tt 'prob' -mod 'csd' 'tensor' -anat '/Users/dPys/PyNets/tests/examples/002/anat/s002_anat_brain.nii.gz' -s 1000000 -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 -n 'Default'

Interpreting outputs
===============================================
PyNets outputs various .csv files and pickled pandas dataframes within each subject's derivative directory which contained the initial image(s) fed into the workflow. You can think of these files as 'meta-derivatives'. Files with the suffix '_neat.csv' within each atlas subdirectory contain the graph measure extracted for that subject from that atlas, using all of the graph hyperparameters listed in the titles of those files. Files with the suffix '_mean.csv' within the base subject directory contain averages/weighted averages of each graph measure across all graph-generating parameters specified at runtime.
