# PyNets
A Python-Powered Workflow for Network Analysis of Resting-State fMRI (rsfMRI) and Diffusion MRI (dMRI)

About 
-----

PyNets automates functional and diffusion-weighted MRI network analysis in python using the Networkx package.

Problem: Network analysis packages for neuroimaging are not implemented in python, preventing them from using the power of nipype.

Solution: In PyNets, we harness the power of nipype, nilearn, and networkx to automatically generate a range of graph theory metrics on a subject-by-subject basis.

Learn more about Nipype: http://nipype.readthedocs.io/en/latest/index.html

Learn more about Nilearn: http://nilearn.github.io/

Learn more about Networkx: https://networkx.github.io/

More specifically: PyNets utilizes nilearn and networkx tools in a nipype workflow to automatically generate rsfMRI networks (whole-brain, or RSN's like the DMN) based on a variety of atlas-defined parcellation schemes, and then automatically plot associated adjacency matrices, connectome visualizations, and extract the following graph theoretical measures from those networks (both binary and weighted undirected versions, with a user-defined thresholding):\
global efficiency, local efficiency, transitivity, degree assortativity coefficient, average clustering coefficient, average shortest path length, betweenness centrality, degree pearson correlation coefficient, number of cliques \

For more information on interpeting these measures, see: https://sites.google.com/site/bctnet/measures/list 

-----

Walkthrough:

1- Step 1:

Required User Inputs: 
	
	-Subject's data- Any 4D preprocessed fMRI file in MNI space, a text file containing a 4D time-series matrix, a pre-made grpah/adjacency matrix, or diffusion weighted image file with completed bedpostx outputs(in construction)
	-A subject ID (user-specified name of the directory containing the input data)

2- Step 2: Create correlation graph object for NetworkX (based on covariance or sparse inverse covariance model fitting)

3- Step 3: Extract network statistics

4- Step 4: Create connectome plots and adjacency matrices

5- Step 5: Aggregate individual network data into group database

6- Step 6: Reduce dimensionality of node metrics at the gorup level using cluster analysis and/or PCA

-----

1. Installation
```python
##Clone the PyNets repo and install dependencies
git clone https://github.com/dpisner453/PyNets.git
cd /path/to/PyNets
pip install -r requirements.txt 

##If you run into any problems at run time, try installing the optional features of nipype with:
pip install nipype[all]
```

2. Usage:\
Situation A) You have a normalized (MNI-space), preprocessed functional rsfMRI image called "filtered_func_data_clean_standard.nii.gz" where the subject id=002, you wish to extract network metrics for a whole-brain network, using the 264-node atlas parcellation scheme from Power et al. 2011 called 'coords_power_2011':
```python
python /path/to/PyNets/pynets.py -i '/Users/dpisner453/PyNets_examples/002/filtered_func_data_clean_standard.nii.gz' -ID '002' -a 'coords_power_2011'
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Situation B) You have a normalized (MNI-space), preprocessed functional rsfMRI image called "filtered_func_data_clean_standard.nii.gz" where the subject id=002, you wish to extract network metrics for the DMN network, using the 264-node atlas parcellation scheme from Power et al. 2011 called 'coords_power_2011' (currently the only atlas supported for extracting RSN networks in PyNets!), you wish to threshold the connectivity graph by preserving 95% of the strongest weights (also the default), you define your node radius as 3 voxels in size (also the default), and you wish to fit model with lasso sparse inverse covariance:
```python
python /path/to/PyNets/pynets.py -i '/Users/dpisner453/PyNets_examples/002/filtered_func_data_clean_standard.nii.gz' -ID '002' -a 'coords_power_2011' -n 'DMN' -thr '0.95' -ns '3' -sps
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Situation C) You only have your time-series in a text or csv-like file where the matrix is saved in the format of # of functional volumes x # of ROI's:

```python
python /path/to/PyNets/pynets.py -i '/Users/dpisner453/PyNets_examples/200/roi_CC200.1D' -ID '200'
```

3. Viewing outputs:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; PyNets outputs network metrics into text files and pickled pandas dataframes within the same subject folder 
in which the initial image or time-series was fed into the workflow. To open the pickled pandas dataframes
from within the interpreter, you can:
```python
import pandas as pd
##Assign pickle path for the covariance (as opposed to the sparse inverse covariance net)
pickle_path = '/Users/dpisner453/PyNets_examples/200/200_net_global_scalars_cov_200'
df = pd.read_pickle(pickle_path)
df
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; These dataframes can then be iteratively loaded and aggregated by row into a single dataframe, where there is 1 row per subject. Here is an example of what that code could look like (where 'frame' here is the aggregated df):
```python
import glob
import os
import pandas as pd
###
working_path = r'/Users/dpisner453/PyNets_examples/network_analysis/' # use your path
name_of_network_pickle = 'DMN_net_global_scalars_inv_sps_cov'
###
allFiles = []
for fn in os.listdir(working_path):
    path_name = path + fn + '/' + fn + '_' + name_of_network_pickle + '_' + fn
    if os.path.isfile(path_name):
        print(path_name)
        allFiles.append(path_name)

frame = pd.DataFrame()
list_ = []

for file_ in allFiles:
    df = pd.read_pickle(file_)
    list_.append(df)

frame = pd.concat(list_)
```

![RSN Nets](PyNets_RSNs.png)

Happy Netting!
