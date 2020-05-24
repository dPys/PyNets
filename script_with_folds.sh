#!/bin/bash

travis_fold() {
  local action=$1
  local name=$2
  echo -en "travis_fold:${action}:${name}\r"
}

travis_fold start estimation 

echo "ESTIMATION"
pytest tests/test_estimation.py -s

travis_fold end estimation

travis_fold start nodemaker

echo "NODEMAKER"
pytest tests/test_nodemaker.py -s

travis_fold end nodemaker

travis_fold start plotting

echo "PLOTTING"
pytest tests/test_plotting.py -s

travis_fold end plotting

#travis_fold start clustering

#echo "CLUSTERING"
#pytest tests/test_clustering.py -s

#travis_fold end clustering

#travis_fold start workflows

#echo "WORKFLOWS"
#pytest tests/test_workflows.py -s

#travis_fold end workflows

#travis_fold start track

#echo "TRACK"
#pytest tests/test_track.py -s

#travis_fold end track

travis_fold start reg_utils

echo "REG UTILS"
pytest tests/test_reg_utils.py -s

travis_fold end reg_utils

travis_fold start dmri_utils 

echo "dMRI UTILS"
pytest tests/test_dmri_utils.py -s

travis_fold end dmri_utils

travis_fold start refs

echo "refs"
pytest tests/test_refs.py -s

travis_fold end refs

travis_fold start netstats

echo "netstats"
pytest tests/test_netstats.py -s

travis_fold end netstats

travis_fold start thresholding

echo "thresholding"
pytest tests/test_thresholding.py -s

travis_fold end thresholding

travis_fold start utils
   
echo "utils"
pytest tests/test_utils.py -s

travis_fold end utils
