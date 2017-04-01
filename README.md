# PyNets
A Nipype-Powered Workflow for Network Analysis of rsfMRI

1. Usage:\
Situation A) You have a standard space-normalized, preprocessed functional rsfMRI image called 
"filtered_func_data_clean_standard.nii.gz" where the subject id=002, the atlas that you wish to
use is the 264-node parcellation scheme from Power et al. 2011 called 'coords_power_2011', and
your rsfMRI image was collected with a TR=2 seconds:

```python
./PyNets.py '/Users/dpisner453/PyNets_examples/002/filtered_func_data_clean_standard.nii.gz' '002' \
'coords_power_2011' '2'
```

  Situation B) You only have your time-series in a text or csv-like file where the matrix is saved
  in the format of # of functional volumes x # of ROI's:

```python
./PyNets.py '/Users/dpisner453/PyNets_examples/200/roi_CC200.1D' '200'
```

2. Viewing outputs:\
PyNets outputs network metrics into text files and pickled pandas dataframes within the same subject folder 
in which the initial image or time-series was fed into the workflow. To open the pickled pandas dataframes, 
you can:

```python
import pandas as pd
##Assign pickle path for the covariance (as opposed to the sparse inverse covariance net)
pickle_path = '/Users/dpisner453/PyNets_examples/200/200_net_global_scalars_cov_200'
df = pd.read_pickle(pickle_path)
df
```

These dataframes can then be iteratively loaded and aggregated by row into a single dataframe, where there is 1 
row per subject.

3. Coming soon (or for any interested developers):\
a) Optionally incorporate a confound regressor into your covariance/ sparse inverse covariance matrix estimation (for now, PyNets assumes this has already been done)\
b) Iterate network metric extraction over atlas-defined and/or group-ICA masked RSN's (e.g. DMN, CCN, etc.)

Happy Netting!
