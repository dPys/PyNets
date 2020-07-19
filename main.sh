#!/bin/bash
#PBS -l nodes=1:ppn=8,walltime=4:00:00
#PBS -l vmem=30gb
#PBS -N pynet

# This file is used to execute PyNets on brainlife.
# brainlife stages this git repo, writes `config.json` and execute this script.
# this script reads the `config.json` and execute pynets container through singularity

# you can run this script(main) without any parameter to test how this App will run outside brainlife
# you will need to copy config.json.brainlife-sample to config.json before running `main` as `main`
# will read all parameters from config.json

set -x
set -e

echo "Running PyNets"
mkdir -p output tmp
singularity run -e docker://dpys/pynets:latest pynets output \
    -p $(jq .p config.json) \
    -mod 'partcorr' 'corr' \
    -min_thr $(jq .min_thr config.json) \
    -max_thr $(jq .max_thr config.json) \
    -step_thr $(jq .step_thr config.json) \
    -sm 0 2 4 -hp 0 0.028 0.080 \
    -a 'BrainnetomeAtlasFan2016'
    -norm 6 \
    -anat $(jq .t1 config.json) \
    -func $(jq .bold config.json) \
    -conf $(jq .regressors config.json) \
    -m $(jq .mask config.json) \
    -id brainlife \
    -work '/tmp' -pm '8,30'
