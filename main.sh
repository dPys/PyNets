#!/bin/bash
#PBS -l nodes=1:ppn=8,walltime=2:00:00
#PBS -l vmem=50gb
#PBS -N pynet

# This file is used to execute PyNets on brainlife.
# brainlife stages this git repo, writes `config.json` and execute this script.
# this script reads the `config.json` and execute pynets container through singularity

# you can run this script(main) without any parameter to test how this App will run outside brainlife
# you will need to copy config.json.brainlife-sample to config.json before running `main` as `main`
# will read all parameters from config.json

set -x
set -e

mkdir -p output tmp

# usage: pynets [-h] -id A subject id or other unique identifier
#               [A subject id or other unique identifier ...]
#               [-func Path to input functional file required for functional connectomes) [Path to input functional file (required for functional connectomes) ...]]
#               [-dwi Path to diffusion-weighted imaging data file (required for dmri connectomes) [Path to diffusion-weighted imaging data file (required for dmri connectomes) ...]]
#               [-bval Path to b-values file (required for dmri connectomes) [Path to b-values file (required for dmri connectomes) ...]]
#               [-bvec Path to b-vectors file (required for dmri connectomes) [Path to b-vectors file (required for dmri connectomes) ...]]
#               [-anat Path to a skull-stripped anatomical Nifti1Image [Path to a skull-stripped anatomical Nifti1Image ...]]
#               [-m Path to a T1w brain mask image (if available) in native anatomical space [Path to a T1w brain mask image (if available) in native anatomical space ...]]
#               [-conf Confound regressor file (.tsv/.csv format) [Confound regressor file (.tsv/.csv format) ...]]
#               [-g Path to graph file input. [Path to graph file input. ...]]
#               [-roi Path to binarized Region-of-Interest (ROI) Nifti1Image in template MNI space. [Path to binarized Region-of-Interest (ROI) Nifti1Image in template MNI space. ...]]
#               [-ref Atlas reference file path]
#               [-way Path to binarized Nifti1Image to constrain tractography [Path to binarized Nifti1Image to constrain tractography ...]]
#               [-ua Path to custom parcellation file [Path to custom parcellation file ...]]
#               [-mod Connectivity estimation/reconstruction method [Connectivity estimation/reconstruction method ...]]
#               [-a Atlas [Atlas ...]]
#               [-ns Spherical centroid node size [Spherical centroid node size ...]]
#               [-thr Graph threshold]
#               [-min_thr Multi-thresholding minimum threshold]
#               [-max_thr Multi-thresholding maximum threshold]
#               [-step_thr Multi-thresholding step size]
#               [-sm Smoothing value (mm fwhm) [Smoothing value (mm fwhm) ...]]
#               [-hp High-pass filter (Hz) [High-pass filter (Hz) ...]]
#               [-es Node extraction strategy [Node extraction strategy ...]]
#               [-k Number of k clusters [Number of k clusters ...]]
#               [-ct Clustering type [Clustering type ...]]
#               [-cm Cluster mask [Cluster mask ...]]
#               [-ml Minimum fiber length for tracking [Minimum fiber length for tracking ...]]
#               [-dg Direction getter [Direction getter ...]]
#               [-norm Normalization strategy for resulting graph(s)] [-bin]
#               [-dt] [-mst] [-p Pruning Strategy] [-df]
#               [-mplx Perform various levels of multiplex graph analysis (only if both structural and diffusion connectometry is run simultaneously.]
#               [-embed] [-spheres]
#               [-n Resting-state network [Resting-state network ...]]
#               [-vox {1mm,2mm}] [-plt] [-pm Cores,memory]
#               [-plug Scheduler type] [-v] [-clean] [-work Working directory]
#               [--version]
#               output_dir


#construct arguments for optional inputs
optional=""

dwi=$(jq -r .dwi config.json)
if [ $dwi != "null" ]; then
    bval=`pwd`/$(jq -r .bvals config.json)
    bvec=`pwd`/$(jq -r .bvecs config.json)
    optional="$optional -dwi $(pwd)/$dwi -bval $bval -bvec $bvec"
fi

bold=$(jq -r .bold config.json)
if [ $bold != "null" ];then 
    conf=`pwd`/$(jq -r .regressors config.json)
    optional="$optional -func $(pwd)/$bold -conf $conf"
fi

mask=$(jq -r .mask config.json)
if [ $mask != "null" ];then 
    optional="$optional -m $(pwd)/$mask"
fi

useratlas=$(jq -r .useratlas config.json)
if [ $useratlas != "null" ];then
    optional="$optional -m $(pwd)/$useratlas/parc.nii.gz"
else
    optional="$optional -a $(jq -r .atlas config.json)"
fi

singularity run -e docker://dpys/pynets:latest pynets \
    `pwd`/output \
    -id brainlife \
    -anat `pwd`/$(jq -r .t1 config.json) \
    -work `pwd`/tmp \
    $optional \
    -p $(jq -r .p config.json) \
    -min_thr $(jq -r .min_thr config.json) \
    -max_thr $(jq -r .max_thr config.json) \
    -step_thr $(jq -r .step_thr config.json) \
    -mod partcorr csd \
    -plt \
    -n Vis \
    -mplx 2 \
    -pm 6,20

# graph
# diffusion 
# diffusion / functional
# functional
