from pathlib import Path
from pynets import nodemaker
import numpy as np

def test_nodemaker_tools_parlistfile_RSN():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    #base_dir = '/Users/PSYC-dap3463/Applications/PyNets/tests/examples'
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    network='Default'
    parc=True
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    
    [coords, atlas_select, par_max, parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
       
    label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    
    [net_coords, net_parcel_list, net_label_names] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)
     
    [net_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(net_parcel_list)
        
    out_path = nodemaker.gen_network_parcels(parlistfile, network, net_label_names, dir_path)

    assert coords is not None
    assert net_coords is not None
    assert net_label_names is not None
    assert net_parcel_list is not None
    assert out_path is not None
    assert net_parcels_map_nifti is not None
    assert parcel_list_exp is not None

def test_nodemaker_tools_nilearn_coords_RSN():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    network='Default'
    atlas_select = 'coords_dosenbach_2010'
    parc = False
    parcel_list = None
    
    [coords, atlas_select, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
    
    [net_coords, net_parcel_list, net_label_names] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)
    
    assert coords is not None
    assert net_coords is not None
    assert net_label_names is not None

def test_nodemaker_tools_masking_parlistfile_RSN():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    mask = dir_path + '/pDMN_3_bin.nii.gz'
    network='Default'
    ID='997'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    parc = True
    
    [coords, atlas_select, par_max, parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
       
    label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    
    [net_coords, net_parcel_list, net_label_names] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)
    
    [net_coords_masked, net_label_names_masked, net_parcel_list_masked] = nodemaker.parcel_masker(mask, net_coords, net_parcel_list, net_label_names, dir_path, ID)
 
    [net_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(net_parcel_list_masked)
        
    out_path = nodemaker.gen_network_parcels(parlistfile, network, net_label_names_masked, dir_path)

    assert coords is not None
    assert net_coords is not None
    assert net_label_names is not None
    assert net_parcel_list is not None
    assert net_coords_masked is not None
    assert net_label_names_masked is not None
    assert net_parcel_list_masked is not None
    assert out_path is not None
    assert net_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    
def test_nodemaker_tools_masking_coords_RSN():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    func_file = dir_path + '/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz'
    mask = dir_path + '/pDMN_3_bin.nii.gz'    
    atlas_select = 'coords_dosenbach_2010'
    network='Default'
    parc = False
    parcel_list = None
    
    [coords, atlas_select, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
    
    [net_coords, net_parcel_list, net_label_names] = nodemaker.get_node_membership(network, func_file, coords, label_names, parc, parcel_list)
    
    [net_coords_masked, net_label_names_masked] = nodemaker.coord_masker(mask, net_coords, net_label_names)

    assert coords is not None
    assert net_coords is not None
    assert net_coords_masked is not None
    assert net_label_names is not None
    assert net_label_names_masked is not None

def test_nodemaker_tools_parlistfile_WB():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    
    [WB_coords, atlas_select, par_max, WB_parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
       
    WB_label_names = np.arange(len(WB_coords) + 1)[np.arange(len(WB_coords) + 1) != 0].tolist()
         
    [WB_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(WB_parcel_list)

    assert WB_coords is not None
    assert WB_label_names is not None
    assert WB_parcel_list is not None
    assert WB_parcels_map_nifti is not None
    assert parcel_list_exp is not None

def test_nodemaker_tools_nilearn_coords_WB():
    ##Set example inputs##
    atlas_select = 'coords_dosenbach_2010'
    
    [WB_coords, atlas_select, networks_list, WB_label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        
    assert WB_coords is not None
    assert WB_label_names is not None

def test_nodemaker_tools_masking_parlistfile_WB():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    mask = dir_path + '/pDMN_3_bin.nii.gz'
    ID='997'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    parc=True

    
    [WB_coords, atlas_select, par_max, WB_parcel_list] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
       
    WB_label_names = np.arange(len(WB_coords) + 1)[np.arange(len(WB_coords) + 1) != 0].tolist()
        
    [WB_coords_masked, WB_label_names_masked, WB_parcel_list_masked] = nodemaker.parcel_masker(mask, WB_coords, WB_parcel_list, WB_label_names, dir_path, ID)
 
    [WB_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(WB_parcel_list_masked)

    [WB_net_parcels_map_nifti_unmasked, WB_coords_unmasked, WB_label_names_unmasked] = nodemaker.node_gen(WB_coords, WB_parcel_list, WB_label_names, dir_path, ID, parc)
    
    [WB_net_parcels_map_nifti_masked, WB_coords_masked, WB_label_names_masked] = nodemaker.node_gen_masking(mask, WB_coords, WB_parcel_list, WB_label_names, dir_path, ID, parc)
    
    assert WB_coords is not None
    assert WB_label_names is not None
    assert WB_parcel_list is not None
    assert WB_coords_masked is not None
    assert WB_label_names_masked is not None
    assert WB_parcel_list_masked is not None
    assert WB_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert WB_net_parcels_map_nifti_unmasked is not None
    assert WB_coords_unmasked is not None
    assert WB_net_parcels_map_nifti_masked is not None
    assert WB_coords_masked is not None
    
def test_nodemaker_tools_masking_coords_WB():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/997'
    mask = dir_path + '/pDMN_3_bin.nii.gz'
    atlas_select = 'coords_dosenbach_2010'
    
    [WB_coords, atlas_select, networks_list, WB_label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        
    [WB_coords_masked, WB_label_names_masked] = nodemaker.coord_masker(mask, WB_coords, WB_label_names)

    assert WB_coords is not None
    assert WB_coords is not None
    assert WB_coords_masked is not None
    assert WB_label_names is not None
    assert WB_label_names_masked is not None

def test_WB_fetch_nodes_and_labels1():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    ref_txt = None
    parc=True
    
    [label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile] = nodemaker.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc)

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert coords is not None

def test_WB_fetch_nodes_and_labels2():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    ref_txt = None
    parc=False
    
    [label_names, coords, atlas_name, networks_list, parcel_list, par_max, parlistfile] = nodemaker.WB_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc)

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert coords is not None
    
def test_RSN_fetch_nodes_and_labels1():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    ref_txt = None
    parc=True
    
    [RSN_label_names, RSN_coords, atlas_name, networks_list, parcel_list, par_max, parlistfile] = nodemaker.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc)

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert RSN_coords is not None
    assert RSN_label_names is not None

def test_RSN_fetch_nodes_and_labels2():
    ##Set example inputs##
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas_select = 'whole_brain_cluster_labels_PCA200'
    ref_txt = None
    parc=False
    
    [RSN_label_names, RSN_coords, atlas_name, networks_list, parcel_list, par_max, parlistfile] = nodemaker.RSN_fetch_nodes_and_labels(atlas_select, parlistfile, ref_txt, parc)

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert RSN_coords is not None
    assert RSN_label_names is not None
