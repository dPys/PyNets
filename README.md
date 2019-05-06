PyNets™
======
[![Build Status](https://travis-ci.org/dPys/PyNets.svg?branch=master)](https://travis-ci.org/dPys/PyNets)


About
-----
A Fully-Automated Workflow for Reproducible Ensemble Graph Analysis of Functional and Structural Connectomes

PyNets harnesses the power of Nipype, Nilearn, Dipy, and Networkx packages to automatically generate graphical ensembles on a subject-by-subject basis, using virtually any combination of graph hyperparameters. PyNets utilities can be integrated with any existing preprocessing workflow, and a docker container is provided to uniquely facilitate complete reproducibility of executions.
-----

1. Installation:

PyNets is available for both python2 and python3. We recommend using python3.
```python
##Clone the PyNets repo and install dependencies
git clone https://github.com/dPys/PyNets.git
cd /path/to/PyNets
python setup.py install

#Or within the PyNets parent directory:
pip install -e .

#Or install from PyPi (recommended)
pip install pynets
```

To install using the included dockerfile, ensure you have installed Docker (https://www.docker.com/) and then run:
```
BUILDIR=$(pwd)
mkdir -p ${BUILDIR}/pynets_images
docker build -t pynets_docker .

docker run -ti --rm --privileged \
    -v /tmp:/tmp \
    -v /var/tmp:/var/tmp \
    pynets_docker
```

and to further convert this into a singularity container (e.g. for use on HPC):

```
docker run -ti --rm \
    --privileged \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ${BUILDIR}/pynets_images:/output \
    filo/docker2singularity "pynets_docker:latest"
```

2. Usage:

See pynets_run.py -h for a complete list of help options.

Example A) You have a preprocessed (minimally -- normalized and skull stripped) functional fMRI dataset called "filtered_func_data_clean_standard.nii.gz" where you assign an arbitrary subject id of 997, you wish to analyze a whole-brain network, using the nilearn atlas 'coords_dosenbach_2010', thresholding the connectivity graph proportionally to retain 0.20% of the strongest connections, and you wish to use partial correlation model estimation:
```python
pynets_run.py -func '/Users/dPys/PyNets_examples/997/filtered_func_data_clean_standard.nii.gz' -id '997' -a 'coords_dosenbach_2010' -mod 'partcorr' -thr '0.20'
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Example B) Building upon the previous example, let's say you now wish to analyze the Default network for this same dataset, but now also using the 264-node atlas parcellation scheme from Power et al. 2011 called 'coords_power_2011', you wish to threshold the graph iteratively to achieve a target density of 0.3, and you define your node radius at two resolutions (2 and 4 mm), you wish to fit a  sparse inverse covariance model in addition to partial correlation, and you wish to plot the results:
```python
pynets_run.py -func '/Users/dPys/PyNets_examples/997/filtered_func_data_clean_standard.nii.gz' -id '997' -a 'coords_dosenbach_2010,coords_power_2011' -n 'Default' -dt -thr '0.3' -ns '2,4' -mod 'partcorr,sps' -plt
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Example C) Building upon the previous examples, let's say you now wish to analyze the Default and Executive Control Networks for this subject, but this time based on a custom atlas (DesikanKlein2012.nii.gz), this time defining your nodes as parcels (as opposed to spheres), you wish to fit a partial correlation model, you wish to iterate the pipeline over a range of densities (i.e. 0.05-0.10 with 1% step), and you wish to prune disconnected nodes:
```python
pynets_run.py -func '/Users/dPys/PyNets_examples/997/filtered_func_data_clean_standard.nii.gz' -id '997' -ua '/Users/dPys/PyNets_example_atlases/DesikanKlein2012.nii.gz' -n 'Default,Cont' -mod 'partcorr' -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -parc -p 1
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Example D) Building upon the previous examples, let's say you now wish to create a subject-specific atlas based on the subject's unique spatial-temporal profile. In this case, you can specify the path to a binarized mask within which to performed spatially-constrained spectral clustering, and you want to try this at multiple resolutions of k clusters/nodes (i.e. k=50,100,150). You again also wish to define your node radius at both 2 and 4 mm, fitting a partial correlation and sparse inverse covariance model, you wish to iterate the pipeline over a range of densities (i.e. 0.05-0.10 with 1% step), you wish to prune disconnected nodes, and you wish to plot your results:
```python
pynets_run.py -func '/Users/dPys/PyNets_examples/997/filtered_func_data_clean_standard.nii.gz' -id '997' -cm '/Users/dPys/PyNets_example/997_grey_matter_mask_bin.nii.gz' -ns '2,4' -mod 'partcorr,sps' -k_min 50 -k_max 150 -k_step 50 -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 -plt
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Example E) You wish to generate a structural connectome, using probabilistic ensemble tractography with 1,000,000 streamlines, based on both constrained-spherical deconvolution (csd) and tensor models, filtering with linear fascicle evaluation (LiFE), and direct normalization of streamlines. You wish to use atlas parcels as defined by both DesikanKlein2012, and AALTzourioMazoyer2002, exploring only those nodes belonging to the Default Mode Network, and iterate over a range of densities (i.e. 0.05-0.10 with 1% step), and prune disconnected nodes:
```python
pynets_run.py -dwi '/Users/derekpisner/Downloads/test_subs/NKI_TRT/sub-0021001/ses-1/dwi/preproc/eddy_corrected_data.nii.gz' -bval '/Users/derekpisner/Downloads/test_subs/NKI_TRT/sub-0021001/ses-1/dwi/preproc/bval.bval' -bvec '/Users/derekpisner/Downloads/test_subs/NKI_TRT/sub-0021001/ses-1/dwi/preproc/bvec.bvec' -id 0021001 -ua '/Users/PSYC-dap3463/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz,/Users/PSYC-dap3463/Applications/PyNets/pynets/atlases/AALTzourioMazoyer2002' -parc -tt 'prob' -mod 'csd,tensor' -anat '/Users/derekpisner/Downloads/test_subs/NKI_TRT/sub-0021001/ses-1/anat/preproc/t1w_brain.nii.gz' -s 1000000 -dt -min_thr 0.05 -max_thr 0.10 -step_thr 0.01 -p 1 -n 'Default'
```

3. Interpreting outputs:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; PyNets outputs various csv files and pickled pandas dataframes within the same subject folder
in which the initial image was fed into the workflow. Files with the suffix '_neat.csv' within each atlas subdirectory contain the graph measure extracted for that subject from that atlas, using all of the graph hyperparameters listed in the titles of those files. Files with the suffix '_mean.csv' within the base subject directory contain averages/weighted averages of each graph measure across all hyperparameters specified at runtime.


Generate a glass brain plot for a functional or structural connectome
![](tests/examples/997/997_whole_brain_cluster_labels_PCA200_sps_connectome_viz.png)
Visualize adjacency matrices for structural or functional connectomes
![](docs/structural_adj_mat.png)
Input a path to a diffusion weighted dataset or bedpostx directory to estimate a structural connectome
![](docs/pynets_diffusion.png)
Visualize communities of restricted networks
![](docs/glass_brain_communities.png)
Use connectograms to visualize community structure (including link communities)
![](docs/link_communities.png)

Happy Netting!

Please cite all uses with reference to the github repository: https://github.com/dPys/PyNets
