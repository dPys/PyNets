Node: meta_wf_002 (fmri_connectometry_002 (fetch_nodes_and_labels_node (utility)
================================================================================


 Hierarchy : wf_single_sub_002_fmri_419.meta_wf_002.fmri_connectometry_002.fetch_nodes_and_labels_node
 Exec ID : fetch_nodes_and_labels_node


Original Inputs
---------------


* atlas : coords_dosenbach_2010
* clustering : <undefined>
* function_str : def fetch_nodes_and_labels(atlas, uatlas, ref_txt, parc, in_file, use_AAL_naming, clustering=False):
    """
    General API for fetching, identifying, and defining atlas nodes based on coordinates and/or labels.

    Parameters
    ----------
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    ref_txt : str
        Path to an atlas reference .txt file that maps labels to intensities corresponding to uatlas.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    in_file : str
        File path to Nifti1Image object whose affine will provide sampling reference for fetching.
    use_AAL_naming : bool
        Indicates whether to perform Automated-Anatomical Labeling of each coordinate from a list of a voxel
        coordinates.
    clustering : bool
        Indicates whether clustering was performed. Default is False.

    Returns
    -------
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    atlas_name : str
        Name of atlas parcellation (can differ slightly from fetch API string).
    networks_list : list
        List of RSN's and their associated cooordinates, if predefined uniquely for a given atlas.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    par_max : int
        The maximum label intensity in the parcellation image.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from pynets.core import utils, nodemaker
    import pandas as pd
    import time
    from pathlib import Path
    import os.path as op

    base_path = utils.get_file()
    # Test if atlas is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009', 'atlas_talairach_gyrus',
                            'atlas_talairach_ba', 'atlas_talairach_lobe']
    nilearn_coords_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
    nilearn_prob_atlases = ['atlas_msdl', 'atlas_pauli_2017']
    if uatlas is None and atlas in nilearn_parc_atlases:
        [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(atlas, parc)
        if uatlas:
            if not isinstance(uatlas, str):
                nib.save(uatlas, "%s%s%s" % ('/tmp/', atlas, '.nii.gz'))
                uatlas = "%s%s%s" % ('/tmp/', atlas, '.nii.gz')
            [coords, _, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas, ' not found!'))
    elif uatlas is None and parc is False and atlas in nilearn_coords_atlases:
        print('Fetching coords and labels from nilearn coordinate-based atlas library...')
        # Fetch nilearn atlas coords
        [coords, _, networks_list, labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
        parcel_list = None
        par_max = None
    elif uatlas is None and parc is False and atlas in nilearn_prob_atlases:
        from nilearn.plotting import find_probabilistic_atlas_cut_coords
        print('Fetching coords and labels from nilearn probabilistic atlas library...')
        # Fetch nilearn atlas coords
        [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(atlas, parc)
        coords = find_probabilistic_atlas_cut_coords(maps_img=uatlas)
        if uatlas:
            if not isinstance(uatlas, str):
                nib.save(uatlas, "%s%s%s" % ('/tmp/', atlas, '.nii.gz'))
                uatlas = "%s%s%s" % ('/tmp/', atlas, '.nii.gz')
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas, ' not found!'))
        par_max = None
    elif uatlas:
        if clustering is True:
            while True:
                if op.isfile(uatlas):
                    break
                else:
                    print('Waiting for atlas file...')
                    time.sleep(15)
        atlas = uatlas.split('/')[-1].split('.')[0]
        try:
            # Fetch user-specified atlas coords
            [coords, atlas, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
            # Describe user atlas coords
            print("%s%s%s%s" % ('\n', atlas, ' comes with {0} '.format(par_max), 'parcels\n'))
        except ValueError:
            print('\n\nError: Either you have specified the name of a nilearn atlas that does not exist or '
                  'you have not supplied a 3d atlas parcellation image!\n\n')
            parcel_list = None
            par_max = None
            coords = None
        labels = None
        networks_list = None
    else:
        networks_list = None
        labels = None
        parcel_list = None
        par_max = None
        coords = None

    # Labels prep
    if atlas:
        if labels:
            pass
        else:
            if (ref_txt is not None) and (op.exists(ref_txt)) and (use_AAL_naming is False):
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                labels = dict_df['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas, '.txt')
                    if op.exists(ref_txt) and (use_AAL_naming is False):
                        try:
                            dict_df = pd.read_csv(ref_txt, sep="\t", header=None, names=["Index", "Region"])
                            labels = dict_df['Region'].tolist()
                        except:
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        if use_AAL_naming is True:
                            try:
                                labels = nodemaker.AAL_naming(coords)
                            except:
                                print('AAL reference labeling failed!')
                                labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                        else:
                            print('Using generic numbering labels...')
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                except:
                    print("Label reference file not found. Attempting AAL naming...")
                    if use_AAL_naming is True:
                        try:
                            labels = nodemaker.AAL_naming(coords)
                        except:
                            print('AAL reference labeling failed!')
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        print('Using generic numbering labels...')
                        labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    else:
        print('WARNING: No labels available since atlas name is not specified!')

    print("%s%s" % ('Labels:\n', labels))
    atlas_name = atlas
    dir_path = utils.do_dir_path(atlas, in_file)

    if len(coords) != len(labels):
        labels = len(coords) * [np.nan]
        if len(coords) != len(labels):
            raise ValueError('ERROR: length of coordinates is not equal to length of label names')

    return labels, coords, atlas_name, networks_list, parcel_list, par_max, uatlas, dir_path

* in_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002_reor_RAS_nores2mm.nii.gz
* parc : False
* ref_txt : None
* uatlas : None
* use_AAL_naming : False

Execution Inputs
----------------


* atlas : coords_dosenbach_2010
* clustering : <undefined>
* function_str : def fetch_nodes_and_labels(atlas, uatlas, ref_txt, parc, in_file, use_AAL_naming, clustering=False):
    """
    General API for fetching, identifying, and defining atlas nodes based on coordinates and/or labels.

    Parameters
    ----------
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    ref_txt : str
        Path to an atlas reference .txt file that maps labels to intensities corresponding to uatlas.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    in_file : str
        File path to Nifti1Image object whose affine will provide sampling reference for fetching.
    use_AAL_naming : bool
        Indicates whether to perform Automated-Anatomical Labeling of each coordinate from a list of a voxel
        coordinates.
    clustering : bool
        Indicates whether clustering was performed. Default is False.

    Returns
    -------
    labels : list
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    atlas_name : str
        Name of atlas parcellation (can differ slightly from fetch API string).
    networks_list : list
        List of RSN's and their associated cooordinates, if predefined uniquely for a given atlas.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    par_max : int
        The maximum label intensity in the parcellation image.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from pynets.core import utils, nodemaker
    import pandas as pd
    import time
    from pathlib import Path
    import os.path as op

    base_path = utils.get_file()
    # Test if atlas is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009', 'atlas_talairach_gyrus',
                            'atlas_talairach_ba', 'atlas_talairach_lobe']
    nilearn_coords_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
    nilearn_prob_atlases = ['atlas_msdl', 'atlas_pauli_2017']
    if uatlas is None and atlas in nilearn_parc_atlases:
        [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(atlas, parc)
        if uatlas:
            if not isinstance(uatlas, str):
                nib.save(uatlas, "%s%s%s" % ('/tmp/', atlas, '.nii.gz'))
                uatlas = "%s%s%s" % ('/tmp/', atlas, '.nii.gz')
            [coords, _, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas, ' not found!'))
    elif uatlas is None and parc is False and atlas in nilearn_coords_atlases:
        print('Fetching coords and labels from nilearn coordinate-based atlas library...')
        # Fetch nilearn atlas coords
        [coords, _, networks_list, labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
        parcel_list = None
        par_max = None
    elif uatlas is None and parc is False and atlas in nilearn_prob_atlases:
        from nilearn.plotting import find_probabilistic_atlas_cut_coords
        print('Fetching coords and labels from nilearn probabilistic atlas library...')
        # Fetch nilearn atlas coords
        [labels, networks_list, uatlas] = nodemaker.nilearn_atlas_helper(atlas, parc)
        coords = find_probabilistic_atlas_cut_coords(maps_img=uatlas)
        if uatlas:
            if not isinstance(uatlas, str):
                nib.save(uatlas, "%s%s%s" % ('/tmp/', atlas, '.nii.gz'))
                uatlas = "%s%s%s" % ('/tmp/', atlas, '.nii.gz')
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas, ' not found!'))
        par_max = None
    elif uatlas:
        if clustering is True:
            while True:
                if op.isfile(uatlas):
                    break
                else:
                    print('Waiting for atlas file...')
                    time.sleep(15)
        atlas = uatlas.split('/')[-1].split('.')[0]
        try:
            # Fetch user-specified atlas coords
            [coords, atlas, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas)
            else:
                parcel_list = None
            # Describe user atlas coords
            print("%s%s%s%s" % ('\n', atlas, ' comes with {0} '.format(par_max), 'parcels\n'))
        except ValueError:
            print('\n\nError: Either you have specified the name of a nilearn atlas that does not exist or '
                  'you have not supplied a 3d atlas parcellation image!\n\n')
            parcel_list = None
            par_max = None
            coords = None
        labels = None
        networks_list = None
    else:
        networks_list = None
        labels = None
        parcel_list = None
        par_max = None
        coords = None

    # Labels prep
    if atlas:
        if labels:
            pass
        else:
            if (ref_txt is not None) and (op.exists(ref_txt)) and (use_AAL_naming is False):
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                labels = dict_df['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas, '.txt')
                    if op.exists(ref_txt) and (use_AAL_naming is False):
                        try:
                            dict_df = pd.read_csv(ref_txt, sep="\t", header=None, names=["Index", "Region"])
                            labels = dict_df['Region'].tolist()
                        except:
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        if use_AAL_naming is True:
                            try:
                                labels = nodemaker.AAL_naming(coords)
                            except:
                                print('AAL reference labeling failed!')
                                labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                        else:
                            print('Using generic numbering labels...')
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                except:
                    print("Label reference file not found. Attempting AAL naming...")
                    if use_AAL_naming is True:
                        try:
                            labels = nodemaker.AAL_naming(coords)
                        except:
                            print('AAL reference labeling failed!')
                            labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        print('Using generic numbering labels...')
                        labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    else:
        print('WARNING: No labels available since atlas name is not specified!')

    print("%s%s" % ('Labels:\n', labels))
    atlas_name = atlas
    dir_path = utils.do_dir_path(atlas, in_file)

    if len(coords) != len(labels):
        labels = len(coords) * [np.nan]
        if len(coords) != len(labels):
            raise ValueError('ERROR: length of coordinates is not equal to length of label names')

    return labels, coords, atlas_name, networks_list, parcel_list, par_max, uatlas, dir_path

* in_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002_reor_RAS_nores2mm.nii.gz
* parc : False
* ref_txt : None
* uatlas : None
* use_AAL_naming : False


Execution Outputs
-----------------


* atlas : coords_dosenbach_2010
* coords : [[ 18 -81 -33]
 [-21 -79 -33]
 [ -6 -79 -33]
 [ 33 -73 -30]
 [-34 -67 -29]
 [ 32 -61 -31]
 [-25 -60 -34]
 [-37 -54 -37]
 [ 21 -64 -22]
 [-34 -57 -24]
 [-24 -54 -21]
 [-28 -44 -25]
 [  5 -75 -11]
 [ 14 -75 -21]
 [-11 -72 -14]
 [  1 -66 -24]
 [-16 -64 -21]
 [ -6 -60 -15]
 [ -2  30  27]
 [-52 -63  15]
 [ 27  49  26]
 [-41 -47  29]
 [-36  18   2]
 [ 38  21  -1]
 [ 11 -24   2]
 [-20   6   7]
 [ 14   6   7]
 [ -6  17  34]
 [  9  20  34]
 [ 54 -31 -18]
 [  0  15  45]
 [-30 -14   1]
 [ 32 -12   2]
 [ 37  -2  -3]
 [-55 -44  30]
 [ 58 -41  20]
 [ -4 -31  -4]
 [-30 -28   9]
 [  8 -40  50]
 [ 42 -46  21]
 [-59 -47  11]
 [ 43 -43   8]
 [ 51 -30   5]
 [-12 -12   6]
 [ 11 -12   6]
 [-12  -3  13]
 [-48   6   1]
 [-46  10  14]
 [ 51  23   8]
 [ 34  32   7]
 [  9  39  20]
 [-36 -69  40]
 [-25  51  27]
 [-48 -63  35]
 [ 51 -59  34]
 [ 28 -37 -15]
 [-61 -41  -2]
 [-59 -25 -15]
 [ 52 -15 -13]
 [  0  51  32]
 [-42 -76  26]
 [ -2 -75  32]
 [ -9 -72  41]
 [ 45 -72  29]
 [-28 -42 -11]
 [-11 -58  17]
 [ 10 -55  17]
 [ -5 -52  17]
 [ -5 -43  25]
 [ -8 -41   3]
 [  1 -26  31]
 [ 11 -68  42]
 [ -6 -56  29]
 [  5 -50  33]
 [  9 -43  25]
 [ -3 -38  45]
 [-16  29  54]
 [ 23  33  47]
 [ 46  39 -15]
 [  8  42  -5]
 [-11  45  17]
 [ -6  50  -1]
 [  9  51  16]
 [  6  64   3]
 [ -1  28  40]
 [ 44 -52  47]
 [-53 -50  39]
 [-48 -47  49]
 [ 54 -44  43]
 [-41 -40  42]
 [ 32 -59  41]
 [-32 -58  46]
 [ 29  57  18]
 [-29  57  10]
 [-42   7  36]
 [ 44   8  34]
 [ 40  17  40]
 [-44  27  33]
 [ 46  28  31]
 [ 40  36  29]
 [-35 -46  48]
 [-52  28  17]
 [-43  47   2]
 [ 42  48  -3]
 [ 39  42  16]
 [ 20 -78  -2]
 [ 15 -77  32]
 [-16 -76  33]
 [  9 -76  14]
 [-29 -75  28]
 [ 29 -73  29]
 [ 39 -71  13]
 [ 17 -68  20]
 [ 19 -66  -1]
 [-44 -63  -7]
 [-34 -60  -5]
 [ 36 -60  -8]
 [-18 -50   1]
 [ -4 -94  12]
 [ 13 -91   2]
 [ 27 -91   2]
 [-29 -88   8]
 [-37 -83  -2]
 [ 29 -81  14]
 [ 33 -81  -2]
 [ -5 -80   9]
 [ 46 -62   5]
 [  0  -1  52]
 [ 60   8  34]
 [ 53  -3  32]
 [ 58  11  14]
 [ 33 -12  16]
 [-36 -12  15]
 [-42  -3  11]
 [-24 -30  64]
 [ 18 -27  62]
 [-38 -27  60]
 [ 41 -23  55]
 [-55 -22  38]
 [ 46 -20  45]
 [-47 -18  50]
 [-38 -15  59]
 [-47 -12  36]
 [-26  -8  54]
 [ 42 -24  17]
 [-41 -31  48]
 [ 10   5  51]
 [-54 -22  22]
 [ 44 -11  38]
 [-54  -9  23]
 [ 46  -8  24]
 [-44  -6  49]
 [ 58  -3  17]
 [ 34 -39  65]
 [-41 -37  16]
 [-53 -37  13]
 [-54 -22   9]
 [ 59 -13   8]
 [ 43   1  12]
 [-55   7  23]]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/coords_dosenbach_2010
* labels : ["inf cerebellum' 155", "inf cerebellum' 150", "inf cerebellum' 151", "inf cerebellum' 140", "inf cerebellum' 131", "inf cerebellum' 122", "inf cerebellum' 121", "inf cerebellum' 110", "lat cerebellum' 128", "lat cerebellum' 113", "lat cerebellum' 109", "lat cerebellum' 98", "med cerebellum' 143", "med cerebellum' 144", "med cerebellum' 138", "med cerebellum' 130", "med cerebellum' 127", "med cerebellum' 120", "ACC' 19", "TPJ' 125", "aPFC' 8", "angular gyrus' 102", "ant insula' 28", "ant insula' 26", "asal ganglia' 71", "asal ganglia' 38", "asal ganglia' 39", "asal ganglia' 30", "dACC' 27", "fusiform' 81", "mFC' 31", "mid insula' 61", "mid insula' 59", "mid insula' 44", "parietal' 97", "parietal' 89", "post cingulate' 80", "post insula' 76", "precuneus' 87", "sup temporal' 100", "temporal' 103", "temporal' 95", "temporal' 78", "thalamus' 57", "thalamus' 58", "thalamus' 47", "vFC' 40", "vFC' 33", "vFC' 25", "vPFC' 18", "ACC' 14", "IPS' 134", "aPFC' 5", "angular gyrus' 124", "angular gyrus' 117", "fusiform' 84", "inf temporal' 91", "inf temporal' 72", "inf temporal' 63", "mPFC' 4", "occipital' 146", "occipital' 141", "occipital' 136", "occipital' 137", "occipital' 92", "post cingulate' 115", "post cingulate' 111", "post cingulate' 108", "post cingulate' 93", "post cingulate' 90", "post cingulate' 73", "precuneus' 132", "precuneus' 112", "precuneus' 105", "precuneus' 94", "precuneus' 85", "sup frontal' 20", "sup frontal' 17", "vlPFC' 15", "vmPFC' 13", "vmPFC' 11", "vmPFC' 7", "vmPFC' 6", "vmPFC' 1", "ACC' 21", "IPL' 107", "IPL' 104", "IPL' 101", "IPL' 96", "IPL' 88", "IPS' 116", "IPS' 114", "aPFC' 2", "aPFC' 3", "dFC' 36", "dFC' 34", "dFC' 29", "dlPFC' 24", "dlPFC' 22", "dlPFC' 16", "post parietal' 99", "vPFC' 23", "vent aPFC' 10", "vent aPFC' 9", "vlPFC' 12", "occipital' 149", "occipital' 148", "occipital' 145", "occipital' 147", "occipital' 142", "occipital' 139", "occipital' 135", "occipital' 133", "occipital' 129", "occipital' 126", "occipital' 118", "occipital' 119", "occipital' 106", "post occipital' 160", "post occipital' 158", "post occipital' 159", "post occipital' 157", "post occipital' 156", "post occipital' 153", "post occipital' 154", "post occipital' 152", "temporal' 123", "SMA' 43", "dFC' 35", "frontal' 45", "frontal' 32", "mid insula' 55", "mid insula' 56", "mid insula' 48", "parietal' 77", "parietal' 74", "parietal' 75", "parietal' 69", "parietal' 66", "parietal' 65", "parietal' 64", "parietal' 62", "parietal' 54", "parietal' 50", "post insula' 70", "post parietal' 79", "pre-SMA' 41", "precentral gyrus' 67", "precentral gyrus' 53", "precentral gyrus' 52", "precentral gyrus' 51", "precentral gyrus' 49", "precentral gyrus' 46", "sup parietal' 86", "temporal' 82", "temporal' 83", "temporal' 68", "temporal' 60", "vFC' 42", "vFC' 37"]
* networks_list : ['cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cerebellum', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'cingulo-opercular', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'default', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'fronto-parietal', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'occipital', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor', 'sensorimotor']
* par_max : None
* parcel_list : None
* uatlas : None


Runtime info
------------


* duration : 1.14302
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_002/wf_single_sub_002_fmri_419/meta_wf_002/fmri_connectometry_002/fetch_nodes_and_labels_node


Environment
~~~~~~~~~~~


* ANTSPATH : /Users/derekpisner/bin/ants/bin/
* Apple_PubSub_Socket_Render : /private/tmp/com.apple.launchd.v8x0bpN28D/Render
* CC : /usr/bin/clang
* CFLAGS :  -I/usr/local/opt/libomp/include
* CONDA_DEFAULT_ENV : base
* CONDA_EXE : /usr/local/anaconda3/bin/conda
* CONDA_PREFIX : /usr/local/anaconda3
* CONDA_PROMPT_MODIFIER : (base) 
* CONDA_PYTHON_EXE : /usr/local/anaconda3/bin/python
* CONDA_SHLVL : 1
* CPPFLAGS : -I/usr/local/opt/libxml2/include -Xpreprocessor -fopenmp
* CXX : /usr/bin/clang++
* CXXFLAGS :  -I/usr/local/opt/libomp/include
* DISPLAY : dpys:0.0
* DYLD_LIBRARY_PATH : /usr/local/opt/libomp/lib
* FIX_VERTEX_AREA : 
* FMRI_ANALYSIS_DIR : /Applications/freesurfer/fsfast
* FREESURFER_HOME : /Applications/freesurfer
* FSFAST_HOME : /Applications/freesurfer/fsfast
* FSF_OUTPUT_FORMAT : nii.gz
* FSLDIR : /usr/local/fsl
* FSLGECUDAQ : cuda.q
* FSLLOCKDIR : 
* FSLMACHINELIST : 
* FSLMULTIFILEQUIT : TRUE
* FSLOUTPUTTYPE : NIFTI_GZ
* FSLREMOTECALL : 
* FSLTCLSH : /usr/local/fsl/bin/fsltclsh
* FSLWISH : /usr/local/fsl/bin/fslwish
* FSL_BIN : /usr/local/fsl/bin
* FSL_DIR : /usr/local/fsl
* FS_OVERRIDE : 0
* FUNCTIONALS_DIR : /Applications/freesurfer/sessions
* HOME : /Users/derekpisner
* KMP_DUPLICATE_LIB_OK : True
* LANG : en_US.UTF-8
* LDFLAGS : -L/usr/local/opt/libxml2/lib -L/usr/local/opt/libomp/lib -lomp
* LOCAL_DIR : /Applications/freesurfer/local
* LOGNAME : derekpisner
* MINC_BIN_DIR : /Applications/freesurfer/mni/bin
* MINC_LIB_DIR : /Applications/freesurfer/mni/lib
* MNI_DATAPATH : /Applications/freesurfer/mni/data
* MNI_DIR : /Applications/freesurfer/mni
* MNI_PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* OLDPWD : /Users/derekpisner/Applications/PyNets/tests/examples/002
* OS : Darwin
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin:/Users/derekpisner/Applications/autoDTI/Batch_scripts:/Users/derekpisner/Applications/autoDTI/Main_scripts:/Users/derekpisner/Applications/autoDTI/Stage_scripts:/Users/derekpisner/Applications/autoDTI/Py_function_library:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/DTI_TK:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Conversion_scripts:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Motion_plotting_scripts:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/Py_function_library:/Users/derekpisner/Applications/autoDTI/3rd_party_scripts_library/QAtools
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri
* SHELL : /bin/bash
* SHLVL : 2
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.0OgUYjH7Dp/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.2
* TERM_SESSION_ID : 035E9BD7-B8A4-4DE5-A5A5-FD6EC00B205C
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0
* autoDTI_HOME : /Users/derekpisner/Applications/autoDTI

