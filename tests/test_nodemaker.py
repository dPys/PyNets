import pkgutil
import io
from pathlib import Path
import pandas as pd
import pynets
import pkgutil
from pynets import nodemaker
from nilearn import datasets

def test_nodemaker_tools():
    ##Set example inputs##
    network='DefaultA'
    mask = Path(__file__).parent/"examples"/"997"/"pDMN_3_bin.nii.gz"
    dir_path = Path(__file__).parent/"examples"/"997"
    atlas_select = 'coords_power_2011'
    func_file = Path(__file__).parent/"examples"/"997"/"sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz"

    [coords, atlas_name, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)

    net_coords = nodemaker.get_membership_from_coords(network, func_file, coords, networks_list)

    ##Test 1
    [net_coords, label_names_out] = nodemaker.coord_masker(str(mask), net_coords, label_names)
    ##Test 2
    [coords, atlas_name, par_max] = nodemaker.get_names_and_coords_of_parcels(str(parlistfile))
    ##Test 3
    #out_path = nodemaker.gen_network_parcels(str(parlistfile), NETWORK, labels_names)

    assert net_coords is not None
    assert label_names_out is not None
    assert mask is not None
    assert coords is not None
    assert atlas_name is not None
    assert par_max is not None
    #assert out_path is not None


def get_sphere(coords, r, vox_dims, dims):
    """##Adapted from Neurosynth
    Return all points within r mm of coordinates. Generates a cube
    and then discards all points outside sphere. Only returns values that
    fall within the dimensions of the image."""
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[
                        i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(
        vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    return sphere[(np.min(sphere, 1) >= 0) &
                  (np.max(np.subtract(sphere, dims), 1) <= -1), :].astype(int)

def fetch_nilearn_atlas_coords(atlas_select):
    atlas = getattr(datasets, 'fetch_%s' % atlas_select)()
    atlas_name = atlas['description'].splitlines()[0]
    print('\n' + str(atlas_name) + ' comes with {0}'.format(atlas.keys()) + '\n')
    coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
    print('Stacked atlas coordinates in array of shape {0}.'.format(coords.shape) + '\n')
    try:
        networks_list = atlas.networks.astype('U')
    except:
        networks_list = None
    try:
        label_names=atlas.labels.astype('U')
        label_names=np.array([s.strip('b\'') for s in label_names]).astype('U')
    except:
        label_names=None
    return(coords, atlas_name, networks_list, label_names)

def get_membership_from_coords(network, func_file, coords, networks_list):
    ##Load subject func data
    bna_img = nib.load(func_file)

    if networks_list is None:
        x_vox = np.diagonal(bna_img.affine[:3,0:3])[0]
        y_vox = np.diagonal(bna_img.affine[:3,0:3])[1]
        z_vox = np.diagonal(bna_img.affine[:3,0:3])[2]

        if x_vox <= 1 and y_vox <= 1 and z_vox <=1:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/BIGREF1mm.nii.gz")
        else:
            par_file = pkg_resources.resource_filename("pynets", "rsnrefs/BIGREF2mm.nii.gz")

        ##Grab RSN reference file
        nets_ref_txt = '/Users/PSYC-dap3463/Applications/PyNets/pynets/rsnrefs/Schaefer2018_1000_17nets_ref.txt'

        ##Create membership dictionary
        dict_df = pd.read_csv(nets_ref_txt, sep="\t", header=None, names=["Index", "Region", "X", "Y", "Z"])
        dict_df.Region.unique().tolist()
        ref_dict = {v: k for v, k in enumerate(dict_df.Region.unique().tolist())}

        par_img = nib.load(par_file)
        par_data = par_img.get_data()

        RSN_ix=list(ref_dict.keys())[list(ref_dict.values()).index(network)]
        RSNmask = par_data[:,:,:,RSN_ix]

        def mmToVox(mmcoords):
            voxcoords = ['','','']
            voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
            voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
            voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)
            return voxcoords

        def VoxTomm(voxcoords):
            mmcoords = ['','','']
            mmcoords[0] = int((round(int(voxcoords[0])-45)*x_vox))
            mmcoords[1] = int((round(int(voxcoords[1])-63)*y_vox))
            mmcoords[2] = int((round(int(voxcoords[2])-36)*z_vox))
            return mmcoords

        coords_vox = []
        for i in coords:
            coords_vox.append(mmToVox(i))
        coords_vox = list(tuple(x) for x in coords_vox)

        error=4
        RSN_coords_vox = []
        for coord in coords_vox:
            sphere_vol = np.zeros(RSNmask.shape, dtype=bool)
            sphere_vol[tuple(coord)] = 1
            if (RSNmask.astype('bool') & sphere_vol).any():
                print(str(coord) + ' falls within mask...')
                RSN_coords_vox.append(coord)
            inds = get_sphere(coord, error, (np.abs(x_vox), y_vox, z_vox), RSNmask.shape)
            sphere_vol[tuple(inds.T)] = 1
            if (RSNmask.astype('bool') & sphere_vol).any():
                print(str(coord) + ' is within a + or - ' + str(error) + ' mm neighborhood...')
                RSN_coords_vox.append(coord)

        coords_mm = []
        for i in RSN_coords_vox:
            coords_mm.append(VoxTomm(i))
        coords_mm = list(tuple(x) for x in coords_mm)

    else:
        '''Fix this later'''
        membership = pd.Series(list(tuple(x) for x in coords), networks_list)
        ##Convert to membership dataframe
        mem_df = membership.to_frame().reset_index()

        nets_avail=list(set(list(mem_df['index'])))
        ##Get network name equivalents
        if network == 'DMN':
            network = 'default'
        elif network == 'FPTC':
            network = 'fronto-parietal'
        elif network == 'CON':
            network = 'cingulo-opercular'
        elif network not in nets_avail:
            print('Error: ' + network + ' not available with this atlas!')
            sys.exit()

        ##Get coords for network-of-interest
        mem_df.loc[mem_df['index'] == network]
        net_coords = mem_df.loc[mem_df['index'] == network][[0]].values[:,0]
        coord_mm = list(tuple(x) for x in net_coords)
        '''Fix this later'''
    return coords_mm

def coord_masker(mask, coords, label_names):
    x_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[0]
    y_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[1]
    z_vox = np.diagonal(masking._load_mask_img(mask)[1][:3,0:3])[2]
    def mmToVox(mmcoords):
        voxcoords = ['','','']
        voxcoords[0] = int((round(int(mmcoords[0])/x_vox))+45)
        voxcoords[1] = int((round(int(mmcoords[1])/y_vox))+63)
        voxcoords[2] = int((round(int(mmcoords[2])/z_vox))+36)
        return voxcoords

    mask_data, _ = masking._load_mask_img(mask)
    mask_coords = list(zip(*np.where(mask_data == True)))
    coords_vox = []
    for i in coords:
        coords_vox.append(mmToVox(i))
    coords_vox = list(tuple(x) for x in coords_vox)

    bad_coords = []
    error=8
    for coord in coords_vox:
        sphere_vol = np.zeros(mask_data.shape, dtype=bool)
        sphere_vol[tuple(coord)] = 1
        if (mask_data & sphere_vol).any():
            print(str(coord) + ' falls within mask...')
            continue
        inds = get_sphere(coord, error, (np.abs(x_vox), y_vox, z_vox), mask_data.shape)
        sphere_vol[tuple(inds.T)] = 1
        if (mask_data & sphere_vol).any():
            print(str(coord) + ' is within a + or - ' + str(error) + ' mm neighborhood...')
            continue
        bad_coords.append(coord)

    bad_coords = [x for x in bad_coords if x is not None]
    indices=[]
    for bad_coord in bad_coords:
        indices.append(coords_vox.index(bad_coord))

    for ix in sorted(indices, reverse=True):
        print('Removing: ' + str(label_names[ix]) + ' at ' + str(coords[ix]))
        del label_names[ix]
        del coords[ix]
    return(coords, label_names)

def get_names_and_coords_of_parcels(parlistfile):
    atlas_name = parlistfile.split('/')[-1].split('.')[0]
    bna_img = nib.load(parlistfile)
    bna_data = bna_img.get_data()
    if bna_img.get_data_dtype() != np.dtype(np.int):
        ##Get an array of unique parcels
        bna_data_for_coords_uniq = np.round(np.unique(bna_data))
        ##Number of parcels:
        par_max = len(bna_data_for_coords_uniq) - 1
        bna_data = bna_data.astype('int16')
    img_stack = []
    for idx in range(1, par_max+1):
        roi_img = bna_data == bna_data_for_coords_uniq[idx]
        img_stack.append(roi_img)
    img_stack = np.array(img_stack)
    img_list = []
    for idx in range(par_max):
        roi_img = nilearn.image.new_img_like(bna_img, img_stack[idx])
        img_list.append(roi_img)
    bna_4D = nilearn.image.concat_imgs(img_list).get_data()
    coords = []
    for roi_img in img_list:
        coords.append(nilearn.plotting.find_xyz_cut_coords(roi_img))
    coords = np.array(coords)
    return(coords, atlas_name, par_max)

def gen_network_parcels(parlistfile, network, labels, dir_path):
    bna_img = nib.load(parlistfile)
    if bna_img.get_data_dtype() != np.dtype(np.int):
        bna_data_for_coords = bna_img.get_data()
        # Number of parcels:
        par_max = np.ceil(np.max(bna_data_for_coords)).astype('int')
        bna_data = bna_data.astype('int16')
    else:
        bna_data = bna_img.get_data()
        par_max = np.max(bna_data)
    img_stack = []
    ##Set indices
    for idx in range(1, par_max+1):
        roi_img = bna_data == idx
        img_stack.append(roi_img)
    img_stack = np.array(img_stack)
    img_list = []
    for idx in range(par_max):
        roi_img = nilearn.image.new_img_like(bna_img, img_stack[idx])
        img_list.append(roi_img)
    print('Extracting parcels associated with ' + network + ' locations...')
    net_parcels = [i for j, i in enumerate(img_list) if j in labels]
    bna_4D = nilearn.image.concat_imgs(net_parcels).get_data()
    index_vec = np.array(range(len(net_parcels))) + 1
    net_parcels_sum = np.sum(index_vec * bna_4D, axis=3)
    net_parcels_map_nifti = nib.Nifti1Image(net_parcels_sum, affine=np.eye(4))
    out_path = dir_path + '/' + network + '_parcels.nii.gz'
    nib.save(net_parcels_map_nifti, out_path)
    return(out_path)
