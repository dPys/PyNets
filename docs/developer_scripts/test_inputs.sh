#!/bin/bash

export PYNETS_dir='/Users/PSYC-dap3463/Applications/PyNets'

##User-atlas, with restricted network
ID='997'
atlas_select='whole_brain_cluster_labels_PCA200'
atlas_name='whole_brain_cluster_labels_PCA200'
network = 'SomMotA'
func_file = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
dir_path = 'tests/examples/997/whole_brain_cluster_labels_PCA200'
coords=pickle.load(open('tests/examples/997/whole_brain_cluster_labels_PCA200/coords_SomMotA_0.95.pkl', 'rb'))
networks_list = None
label_names = pickle.load(open('tests/examples/997/whole_brain_cluster_labels_PCA200/labelnames_SomMotA_0.95.pkl', 'rb'))
conn_matrix = np.genfromtxt('tests/examples/997/whole_brain_cluster_labels_PCA200/sub002_SomMotA_est_sps_inv_0.95.txt')
conn_model = 'sps'
parlistfile = 'tests/examples/whole_brain_cluster_labels_PCA200.nii.gz'
mask = None
thr = 0.95
bedpostx_dir = None
node_size = 3
#conf = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
conf = None
adapt_thresh = False
dens_thresh = None

python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -ua $PYNETS_dir/tests/examples/'whole_brain_cluster_labels_PCA200.nii.gz' -model 'sps' -n 'SomMotA' -conf $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
#python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -ua $PYNETS_dir/tests/examples/'whole_brain_cluster_labels_PCA200.nii.gz' -model 'corr' -n 'SomMotA'

##User-atlas, without restricted network
ID='997'
atlas_select='whole_brain_cluster_labels_PCA200'
atlas_name='whole_brain_cluster_labels_PCA200'
network = None
func_file = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
dir_path = 'tests/examples/997/whole_brain_cluster_labels_PCA200'
coords=pickle.load(open('tests/examples/997/whole_brain_cluster_labels_PCA200/coords_0.95.pkl', 'rb'))
networks_list = None
label_names = pickle.load(open('tests/examples/997/whole_brain_cluster_labels_PCA200/labelnames_0.95.pkl', 'rb'))
conn_matrix = np.genfromtxt('tests/examples/997/whole_brain_cluster_labels_PCA200/sub002_est_sps_inv_0.95.txt')
conn_model = 'sps'
parlistfile = 'tests/examples/whole_brain_cluster_labels_PCA200.nii.gz'
mask = None
thr = 0.95
bedpostx_dir = None
node_size = 3
#conf = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
conf = None
adapt_thresh = False
dens_thresh = None

python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -ua $PYNETS_dir/tests/examples/'whole_brain_cluster_labels_PCA200.nii.gz' -model 'sps' -plt -conf $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
#python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -ua $PYNETS_dir/tests/examples/'whole_brain_cluster_labels_PCA200.nii.gz' -model 'sps' -plt

##Power Atlas, with restricted network
ID='997'
atlas_select='coords_power_2011'
atlas_name='Power Atlas 2011'
network = 'SalVentAttnA'
func_file = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
dir_path = 'tests/examples/997/coords_power_2011'
coords=pickle.load(open('tests/examples/997/coords_power_2011/coords_SalVentAttnA_0.95.pkl', 'rb'))
networks_list = None
label_names = pickle.load(open('tests/examples/997/coords_power_2011/labelnames_SalVentAttnA_0.95.pkl', 'rb'))
conn_matrix = np.genfromtxt('tests/examples/997/coords_power_2011/sub002_SalVentAttnA_est_sps_inv_0.95.txt')
conn_model = 'sps'
parlistfile = None
mask = None
thr = 0.95
bedpostx_dir = None
node_size = 3
#conf = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
conf = None
adapt_thresh = False
dens_thresh = None

python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -a 'coords_power_2011' -model 'sps' -n 'SomMotA' -plt -conf $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
#python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -a 'coords_power_2011' -model 'sps' -n 'SomMotA' -plt

##Power Atlas, without restricted network
ID='997'
atlas_select='coords_power_2011'
atlas_name='Power Atlas 2011'
network = None
func_file = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
dir_path = 'tests/examples/997/coords_power_2011'
coords=pickle.load(open('tests/examples/997/coords_power_2011/coords_0.95.pkl', 'rb'))
networks_list = None
label_names = pickle.load(open('tests/examples/997/coords_power_2011/labelnames_0.95.pkl', 'rb'))
conn_matrix = np.genfromtxt('tests/examples/997/coords_power_2011/sub002_est_sps_inv_0.95.txt')
conn_model = 'sps'
parlistfile = None
mask = None
thr = 0.95
bedpostx_dir = None
node_size = 3
#conf = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
conf = None
adapt_thresh = False
dens_thresh = None

python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -a 'coords_power_2011' -model 'sps' -conf $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
#python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -a 'coords_power_2011' -model 'sps'

##Dosenbach Atlas, with restricted network
ID='997'
atlas_select='coords_dosenbach_2010'
atlas_name='Dosenbach Atlas 2010'
network = 'cingulo-opercular'
func_file = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
dir_path = 'tests/examples/997/coords_dosenbach_2010'
coords=pickle.load(open('tests/examples/997/coords_dosenbach_2010/coords_SalVentAttnA_0.95.pkl', 'rb'))
networks_list = None
label_names = pickle.load(open('tests/examples/997/coords_dosenbach_2010/labelnames_SalVentAttnA_0.95.pkl', 'rb'))
conn_matrix = np.genfromtxt('tests/examples/997/coords_dosenbach_2010/sub002_SalVentAttnA_est_sps_inv_0.95.txt')
conn_model = 'sps'
parlistfile = None
mask = None
thr = 0.95
bedpostx_dir = None
node_size = 3
conf = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
conf = None
adapt_thresh = False
dens_thresh = None

python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -a 'coords_dosenbach_2010' -model 'sps' -n 'cingulo-opercular' -conf $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
#python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -a 'coords_dosenbach_2010' -model 'sps' -n 'cingulo-opercular'

#Dosenbach Atlas, without restricted network
ID='997'
atlas_select='coords_dosenbach_2010'
atlas_name='Dosenbach Atlas 2010'
network = None
func_file = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
dir_path = 'tests/examples/997/coords_dosenbach_2010'
coords=pickle.load(open('tests/examples/997/coords_dosenbach_2010/coords_0.95.pkl', 'rb'))
networks_list = None
label_names = pickle.load(open('tests/examples/997/coords_dosenbach_2010/labelnames_0.95.pkl', 'rb'))
conn_matrix = np.genfromtxt('tests/examples/997/coords_dosenbach_2010/sub002_est_sps_inv_0.95.txt')
conn_model = 'sps'
parlistfile = None
mask = None
thr = 0.95
bedpostx_dir = None
node_size = 3
#conf = 'tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
conf = None
adapt_thresh = False
dens_thresh = None

python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -a 'coords_dosenbach_2010' -model 'sps' -plt -conf $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_confounds.tsv'
#python3 pynets/pynets_run.py -i $PYNETS_dir/tests/examples/997/'sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz' -ID '997' -a 'coords_dosenbach_2010' -model 'sps' -plt
