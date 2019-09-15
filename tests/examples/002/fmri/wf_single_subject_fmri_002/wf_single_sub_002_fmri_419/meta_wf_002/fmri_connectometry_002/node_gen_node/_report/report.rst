Node: meta_wf_002 (fmri_connectometry_002 (node_gen_node (utility)
==================================================================


 Hierarchy : wf_single_sub_002_fmri_419.meta_wf_002.fmri_connectometry_002.node_gen_node
 Exec ID : node_gen_node


Original Inputs
---------------


* ID : 002
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
* function_str : def node_gen(coords, parcel_list, labels, dir_path, ID, parc, atlas, uatlas):
    """
    In the case that masking was not applied, this function generate nodes based on atlas definitions established by
    fetch_nodes_and_labels.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets.core import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list)
    else:
        net_parcels_map_nifti = None
        print('No additional roi masking...')

    coords = list(tuple(x) for x in coords)
    if pick_dump is True:
        # Save coords to pickle
        coords_path = "%s%s" % (dir_path, '/atlas_coords_wb.pkl')
        with open(coords_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        # Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f, protocol=2)

    return net_parcels_map_nifti, coords, labels, atlas, uatlas

* labels : ["inf cerebellum' 155", "inf cerebellum' 150", "inf cerebellum' 151", "inf cerebellum' 140", "inf cerebellum' 131", "inf cerebellum' 122", "inf cerebellum' 121", "inf cerebellum' 110", "lat cerebellum' 128", "lat cerebellum' 113", "lat cerebellum' 109", "lat cerebellum' 98", "med cerebellum' 143", "med cerebellum' 144", "med cerebellum' 138", "med cerebellum' 130", "med cerebellum' 127", "med cerebellum' 120", "ACC' 19", "TPJ' 125", "aPFC' 8", "angular gyrus' 102", "ant insula' 28", "ant insula' 26", "asal ganglia' 71", "asal ganglia' 38", "asal ganglia' 39", "asal ganglia' 30", "dACC' 27", "fusiform' 81", "mFC' 31", "mid insula' 61", "mid insula' 59", "mid insula' 44", "parietal' 97", "parietal' 89", "post cingulate' 80", "post insula' 76", "precuneus' 87", "sup temporal' 100", "temporal' 103", "temporal' 95", "temporal' 78", "thalamus' 57", "thalamus' 58", "thalamus' 47", "vFC' 40", "vFC' 33", "vFC' 25", "vPFC' 18", "ACC' 14", "IPS' 134", "aPFC' 5", "angular gyrus' 124", "angular gyrus' 117", "fusiform' 84", "inf temporal' 91", "inf temporal' 72", "inf temporal' 63", "mPFC' 4", "occipital' 146", "occipital' 141", "occipital' 136", "occipital' 137", "occipital' 92", "post cingulate' 115", "post cingulate' 111", "post cingulate' 108", "post cingulate' 93", "post cingulate' 90", "post cingulate' 73", "precuneus' 132", "precuneus' 112", "precuneus' 105", "precuneus' 94", "precuneus' 85", "sup frontal' 20", "sup frontal' 17", "vlPFC' 15", "vmPFC' 13", "vmPFC' 11", "vmPFC' 7", "vmPFC' 6", "vmPFC' 1", "ACC' 21", "IPL' 107", "IPL' 104", "IPL' 101", "IPL' 96", "IPL' 88", "IPS' 116", "IPS' 114", "aPFC' 2", "aPFC' 3", "dFC' 36", "dFC' 34", "dFC' 29", "dlPFC' 24", "dlPFC' 22", "dlPFC' 16", "post parietal' 99", "vPFC' 23", "vent aPFC' 10", "vent aPFC' 9", "vlPFC' 12", "occipital' 149", "occipital' 148", "occipital' 145", "occipital' 147", "occipital' 142", "occipital' 139", "occipital' 135", "occipital' 133", "occipital' 129", "occipital' 126", "occipital' 118", "occipital' 119", "occipital' 106", "post occipital' 160", "post occipital' 158", "post occipital' 159", "post occipital' 157", "post occipital' 156", "post occipital' 153", "post occipital' 154", "post occipital' 152", "temporal' 123", "SMA' 43", "dFC' 35", "frontal' 45", "frontal' 32", "mid insula' 55", "mid insula' 56", "mid insula' 48", "parietal' 77", "parietal' 74", "parietal' 75", "parietal' 69", "parietal' 66", "parietal' 65", "parietal' 64", "parietal' 62", "parietal' 54", "parietal' 50", "post insula' 70", "post parietal' 79", "pre-SMA' 41", "precentral gyrus' 67", "precentral gyrus' 53", "precentral gyrus' 52", "precentral gyrus' 51", "precentral gyrus' 49", "precentral gyrus' 46", "sup parietal' 86", "temporal' 82", "temporal' 83", "temporal' 68", "temporal' 60", "vFC' 42", "vFC' 37"]
* par_max : None
* parc : False
* parcel_list : None
* roi : None
* uatlas : None

Execution Inputs
----------------


* ID : 002
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
* function_str : def node_gen(coords, parcel_list, labels, dir_path, ID, parc, atlas, uatlas):
    """
    In the case that masking was not applied, this function generate nodes based on atlas definitions established by
    fetch_nodes_and_labels.

    Parameters
    ----------
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    parcel_list : list
        List of 3D boolean numpy arrays or binarized Nifti1Images corresponding to ROI masks.
    labels : list
        List of string labels corresponding to ROI nodes.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    ID : str
        A subject id or other unique identifier.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.

    Returns
    -------
    net_parcels_map_nifti : Nifti1Image
        A nibabel-based nifti image consisting of a 3D array with integer voxel intensities corresponding to ROI
        membership.
    coords : list
        List of (x, y, z) tuples in mm-space corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    labels : list
        List of string labels corresponding to ROI nodes.
    atlas : str
        Name of a Nilearn-hosted coordinate or parcellation/label-based atlas supported for fetching.
        See Nilearn's datasets.atlas module for more detailed reference.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets.core import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list)
    else:
        net_parcels_map_nifti = None
        print('No additional roi masking...')

    coords = list(tuple(x) for x in coords)
    if pick_dump is True:
        # Save coords to pickle
        coords_path = "%s%s" % (dir_path, '/atlas_coords_wb.pkl')
        with open(coords_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        # Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f, protocol=2)

    return net_parcels_map_nifti, coords, labels, atlas, uatlas

* labels : ["inf cerebellum' 155", "inf cerebellum' 150", "inf cerebellum' 151", "inf cerebellum' 140", "inf cerebellum' 131", "inf cerebellum' 122", "inf cerebellum' 121", "inf cerebellum' 110", "lat cerebellum' 128", "lat cerebellum' 113", "lat cerebellum' 109", "lat cerebellum' 98", "med cerebellum' 143", "med cerebellum' 144", "med cerebellum' 138", "med cerebellum' 130", "med cerebellum' 127", "med cerebellum' 120", "ACC' 19", "TPJ' 125", "aPFC' 8", "angular gyrus' 102", "ant insula' 28", "ant insula' 26", "asal ganglia' 71", "asal ganglia' 38", "asal ganglia' 39", "asal ganglia' 30", "dACC' 27", "fusiform' 81", "mFC' 31", "mid insula' 61", "mid insula' 59", "mid insula' 44", "parietal' 97", "parietal' 89", "post cingulate' 80", "post insula' 76", "precuneus' 87", "sup temporal' 100", "temporal' 103", "temporal' 95", "temporal' 78", "thalamus' 57", "thalamus' 58", "thalamus' 47", "vFC' 40", "vFC' 33", "vFC' 25", "vPFC' 18", "ACC' 14", "IPS' 134", "aPFC' 5", "angular gyrus' 124", "angular gyrus' 117", "fusiform' 84", "inf temporal' 91", "inf temporal' 72", "inf temporal' 63", "mPFC' 4", "occipital' 146", "occipital' 141", "occipital' 136", "occipital' 137", "occipital' 92", "post cingulate' 115", "post cingulate' 111", "post cingulate' 108", "post cingulate' 93", "post cingulate' 90", "post cingulate' 73", "precuneus' 132", "precuneus' 112", "precuneus' 105", "precuneus' 94", "precuneus' 85", "sup frontal' 20", "sup frontal' 17", "vlPFC' 15", "vmPFC' 13", "vmPFC' 11", "vmPFC' 7", "vmPFC' 6", "vmPFC' 1", "ACC' 21", "IPL' 107", "IPL' 104", "IPL' 101", "IPL' 96", "IPL' 88", "IPS' 116", "IPS' 114", "aPFC' 2", "aPFC' 3", "dFC' 36", "dFC' 34", "dFC' 29", "dlPFC' 24", "dlPFC' 22", "dlPFC' 16", "post parietal' 99", "vPFC' 23", "vent aPFC' 10", "vent aPFC' 9", "vlPFC' 12", "occipital' 149", "occipital' 148", "occipital' 145", "occipital' 147", "occipital' 142", "occipital' 139", "occipital' 135", "occipital' 133", "occipital' 129", "occipital' 126", "occipital' 118", "occipital' 119", "occipital' 106", "post occipital' 160", "post occipital' 158", "post occipital' 159", "post occipital' 157", "post occipital' 156", "post occipital' 153", "post occipital' 154", "post occipital' 152", "temporal' 123", "SMA' 43", "dFC' 35", "frontal' 45", "frontal' 32", "mid insula' 55", "mid insula' 56", "mid insula' 48", "parietal' 77", "parietal' 74", "parietal' 75", "parietal' 69", "parietal' 66", "parietal' 65", "parietal' 64", "parietal' 62", "parietal' 54", "parietal' 50", "post insula' 70", "post parietal' 79", "pre-SMA' 41", "precentral gyrus' 67", "precentral gyrus' 53", "precentral gyrus' 52", "precentral gyrus' 51", "precentral gyrus' 49", "precentral gyrus' 46", "sup parietal' 86", "temporal' 82", "temporal' 83", "temporal' 68", "temporal' 60", "vFC' 42", "vFC' 37"]
* par_max : None
* parc : False
* parcel_list : None
* roi : None
* uatlas : None


Execution Outputs
-----------------


* atlas : coords_dosenbach_2010
* coords : [(18, -81, -33), (-21, -79, -33), (-6, -79, -33), (33, -73, -30), (-34, -67, -29), (32, -61, -31), (-25, -60, -34), (-37, -54, -37), (21, -64, -22), (-34, -57, -24), (-24, -54, -21), (-28, -44, -25), (5, -75, -11), (14, -75, -21), (-11, -72, -14), (1, -66, -24), (-16, -64, -21), (-6, -60, -15), (-2, 30, 27), (-52, -63, 15), (27, 49, 26), (-41, -47, 29), (-36, 18, 2), (38, 21, -1), (11, -24, 2), (-20, 6, 7), (14, 6, 7), (-6, 17, 34), (9, 20, 34), (54, -31, -18), (0, 15, 45), (-30, -14, 1), (32, -12, 2), (37, -2, -3), (-55, -44, 30), (58, -41, 20), (-4, -31, -4), (-30, -28, 9), (8, -40, 50), (42, -46, 21), (-59, -47, 11), (43, -43, 8), (51, -30, 5), (-12, -12, 6), (11, -12, 6), (-12, -3, 13), (-48, 6, 1), (-46, 10, 14), (51, 23, 8), (34, 32, 7), (9, 39, 20), (-36, -69, 40), (-25, 51, 27), (-48, -63, 35), (51, -59, 34), (28, -37, -15), (-61, -41, -2), (-59, -25, -15), (52, -15, -13), (0, 51, 32), (-42, -76, 26), (-2, -75, 32), (-9, -72, 41), (45, -72, 29), (-28, -42, -11), (-11, -58, 17), (10, -55, 17), (-5, -52, 17), (-5, -43, 25), (-8, -41, 3), (1, -26, 31), (11, -68, 42), (-6, -56, 29), (5, -50, 33), (9, -43, 25), (-3, -38, 45), (-16, 29, 54), (23, 33, 47), (46, 39, -15), (8, 42, -5), (-11, 45, 17), (-6, 50, -1), (9, 51, 16), (6, 64, 3), (-1, 28, 40), (44, -52, 47), (-53, -50, 39), (-48, -47, 49), (54, -44, 43), (-41, -40, 42), (32, -59, 41), (-32, -58, 46), (29, 57, 18), (-29, 57, 10), (-42, 7, 36), (44, 8, 34), (40, 17, 40), (-44, 27, 33), (46, 28, 31), (40, 36, 29), (-35, -46, 48), (-52, 28, 17), (-43, 47, 2), (42, 48, -3), (39, 42, 16), (20, -78, -2), (15, -77, 32), (-16, -76, 33), (9, -76, 14), (-29, -75, 28), (29, -73, 29), (39, -71, 13), (17, -68, 20), (19, -66, -1), (-44, -63, -7), (-34, -60, -5), (36, -60, -8), (-18, -50, 1), (-4, -94, 12), (13, -91, 2), (27, -91, 2), (-29, -88, 8), (-37, -83, -2), (29, -81, 14), (33, -81, -2), (-5, -80, 9), (46, -62, 5), (0, -1, 52), (60, 8, 34), (53, -3, 32), (58, 11, 14), (33, -12, 16), (-36, -12, 15), (-42, -3, 11), (-24, -30, 64), (18, -27, 62), (-38, -27, 60), (41, -23, 55), (-55, -22, 38), (46, -20, 45), (-47, -18, 50), (-38, -15, 59), (-47, -12, 36), (-26, -8, 54), (42, -24, 17), (-41, -31, 48), (10, 5, 51), (-54, -22, 22), (44, -11, 38), (-54, -9, 23), (46, -8, 24), (-44, -6, 49), (58, -3, 17), (34, -39, 65), (-41, -37, 16), (-53, -37, 13), (-54, -22, 9), (59, -13, 8), (43, 1, 12), (-55, 7, 23)]
* labels : ["inf cerebellum' 155", "inf cerebellum' 150", "inf cerebellum' 151", "inf cerebellum' 140", "inf cerebellum' 131", "inf cerebellum' 122", "inf cerebellum' 121", "inf cerebellum' 110", "lat cerebellum' 128", "lat cerebellum' 113", "lat cerebellum' 109", "lat cerebellum' 98", "med cerebellum' 143", "med cerebellum' 144", "med cerebellum' 138", "med cerebellum' 130", "med cerebellum' 127", "med cerebellum' 120", "ACC' 19", "TPJ' 125", "aPFC' 8", "angular gyrus' 102", "ant insula' 28", "ant insula' 26", "asal ganglia' 71", "asal ganglia' 38", "asal ganglia' 39", "asal ganglia' 30", "dACC' 27", "fusiform' 81", "mFC' 31", "mid insula' 61", "mid insula' 59", "mid insula' 44", "parietal' 97", "parietal' 89", "post cingulate' 80", "post insula' 76", "precuneus' 87", "sup temporal' 100", "temporal' 103", "temporal' 95", "temporal' 78", "thalamus' 57", "thalamus' 58", "thalamus' 47", "vFC' 40", "vFC' 33", "vFC' 25", "vPFC' 18", "ACC' 14", "IPS' 134", "aPFC' 5", "angular gyrus' 124", "angular gyrus' 117", "fusiform' 84", "inf temporal' 91", "inf temporal' 72", "inf temporal' 63", "mPFC' 4", "occipital' 146", "occipital' 141", "occipital' 136", "occipital' 137", "occipital' 92", "post cingulate' 115", "post cingulate' 111", "post cingulate' 108", "post cingulate' 93", "post cingulate' 90", "post cingulate' 73", "precuneus' 132", "precuneus' 112", "precuneus' 105", "precuneus' 94", "precuneus' 85", "sup frontal' 20", "sup frontal' 17", "vlPFC' 15", "vmPFC' 13", "vmPFC' 11", "vmPFC' 7", "vmPFC' 6", "vmPFC' 1", "ACC' 21", "IPL' 107", "IPL' 104", "IPL' 101", "IPL' 96", "IPL' 88", "IPS' 116", "IPS' 114", "aPFC' 2", "aPFC' 3", "dFC' 36", "dFC' 34", "dFC' 29", "dlPFC' 24", "dlPFC' 22", "dlPFC' 16", "post parietal' 99", "vPFC' 23", "vent aPFC' 10", "vent aPFC' 9", "vlPFC' 12", "occipital' 149", "occipital' 148", "occipital' 145", "occipital' 147", "occipital' 142", "occipital' 139", "occipital' 135", "occipital' 133", "occipital' 129", "occipital' 126", "occipital' 118", "occipital' 119", "occipital' 106", "post occipital' 160", "post occipital' 158", "post occipital' 159", "post occipital' 157", "post occipital' 156", "post occipital' 153", "post occipital' 154", "post occipital' 152", "temporal' 123", "SMA' 43", "dFC' 35", "frontal' 45", "frontal' 32", "mid insula' 55", "mid insula' 56", "mid insula' 48", "parietal' 77", "parietal' 74", "parietal' 75", "parietal' 69", "parietal' 66", "parietal' 65", "parietal' 64", "parietal' 62", "parietal' 54", "parietal' 50", "post insula' 70", "post parietal' 79", "pre-SMA' 41", "precentral gyrus' 67", "precentral gyrus' 53", "precentral gyrus' 52", "precentral gyrus' 51", "precentral gyrus' 49", "precentral gyrus' 46", "sup parietal' 86", "temporal' 82", "temporal' 83", "temporal' 68", "temporal' 60", "vFC' 42", "vFC' 37"]
* net_parcels_map_nifti : None
* uatlas : None


Runtime info
------------


* duration : 0.002544
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_fmri_002/wf_single_sub_002_fmri_419/meta_wf_002/fmri_connectometry_002/node_gen_node


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

