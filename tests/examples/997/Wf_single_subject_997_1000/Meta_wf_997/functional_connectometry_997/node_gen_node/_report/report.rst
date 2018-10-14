Node: Meta_wf_997 (functional_connectometry_997 (node_gen_node (utility)
========================================================================


 Hierarchy : Wf_single_subject_997_1000.Meta_wf_997.functional_connectometry_997.node_gen_node
 Exec ID : node_gen_node


Original Inputs
---------------


* ID : 997
* atlas_select : coords_dosenbach_2010
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
* dir_path : /Users/PSYC-dap3463/Applications/PyNets/tests/examples/997/coords_dosenbach_2010
* function_str : def node_gen(coords, parcel_list, label_names, dir_path, ID, parc, atlas_select, uatlas_select):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list)
        vox_list = []
        for i in range(len(parcel_list)):
            vox_list.append(np.count_nonzero(parcel_list[i].get_data()))
        vox_array = np.array(vox_list).astype('float64')
    else:
        net_parcels_map_nifti = None
        vox_array = None
        print('No additional masking...')

    coords = list(tuple(x) for x in coords)
    if pick_dump is True:
        # Save coords to pickle
        coord_path = "%s%s" % (dir_path, '/atlas_coords_wb.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        # Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f, protocol=2)

    return net_parcels_map_nifti, coords, label_names, atlas_select, uatlas_select, vox_array

* label_names : ["inf cerebellum' 155", "inf cerebellum' 150", "inf cerebellum' 151", "inf cerebellum' 140", "inf cerebellum' 131", "inf cerebellum' 122", "inf cerebellum' 121", "inf cerebellum' 110", "lat cerebellum' 128", "lat cerebellum' 113", "lat cerebellum' 109", "lat cerebellum' 98", "med cerebellum' 143", "med cerebellum' 144", "med cerebellum' 138", "med cerebellum' 130", "med cerebellum' 127", "med cerebellum' 120", "ACC' 19", "TPJ' 125", "aPFC' 8", "angular gyrus' 102", "ant insula' 28", "ant insula' 26", "asal ganglia' 71", "asal ganglia' 38", "asal ganglia' 39", "asal ganglia' 30", "dACC' 27", "fusiform' 81", "mFC' 31", "mid insula' 61", "mid insula' 59", "mid insula' 44", "parietal' 97", "parietal' 89", "post cingulate' 80", "post insula' 76", "precuneus' 87", "sup temporal' 100", "temporal' 103", "temporal' 95", "temporal' 78", "thalamus' 57", "thalamus' 58", "thalamus' 47", "vFC' 40", "vFC' 33", "vFC' 25", "vPFC' 18", "ACC' 14", "IPS' 134", "aPFC' 5", "angular gyrus' 124", "angular gyrus' 117", "fusiform' 84", "inf temporal' 91", "inf temporal' 72", "inf temporal' 63", "mPFC' 4", "occipital' 146", "occipital' 141", "occipital' 136", "occipital' 137", "occipital' 92", "post cingulate' 115", "post cingulate' 111", "post cingulate' 108", "post cingulate' 93", "post cingulate' 90", "post cingulate' 73", "precuneus' 132", "precuneus' 112", "precuneus' 105", "precuneus' 94", "precuneus' 85", "sup frontal' 20", "sup frontal' 17", "vlPFC' 15", "vmPFC' 13", "vmPFC' 11", "vmPFC' 7", "vmPFC' 6", "vmPFC' 1", "ACC' 21", "IPL' 107", "IPL' 104", "IPL' 101", "IPL' 96", "IPL' 88", "IPS' 116", "IPS' 114", "aPFC' 2", "aPFC' 3", "dFC' 36", "dFC' 34", "dFC' 29", "dlPFC' 24", "dlPFC' 22", "dlPFC' 16", "post parietal' 99", "vPFC' 23", "vent aPFC' 10", "vent aPFC' 9", "vlPFC' 12", "occipital' 149", "occipital' 148", "occipital' 145", "occipital' 147", "occipital' 142", "occipital' 139", "occipital' 135", "occipital' 133", "occipital' 129", "occipital' 126", "occipital' 118", "occipital' 119", "occipital' 106", "post occipital' 160", "post occipital' 158", "post occipital' 159", "post occipital' 157", "post occipital' 156", "post occipital' 153", "post occipital' 154", "post occipital' 152", "temporal' 123", "SMA' 43", "dFC' 35", "frontal' 45", "frontal' 32", "mid insula' 55", "mid insula' 56", "mid insula' 48", "parietal' 77", "parietal' 74", "parietal' 75", "parietal' 69", "parietal' 66", "parietal' 65", "parietal' 64", "parietal' 62", "parietal' 54", "parietal' 50", "post insula' 70", "post parietal' 79", "pre-SMA' 41", "precentral gyrus' 67", "precentral gyrus' 53", "precentral gyrus' 52", "precentral gyrus' 51", "precentral gyrus' 49", "precentral gyrus' 46", "sup parietal' 86", "temporal' 82", "temporal' 83", "temporal' 68", "temporal' 60", "vFC' 42", "vFC' 37"]
* mask : None
* par_max : None
* parc : False
* parcel_list : None
* uatlas_select : None

Execution Inputs
----------------


* ID : 997
* atlas_select : coords_dosenbach_2010
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
* dir_path : /Users/PSYC-dap3463/Applications/PyNets/tests/examples/997/coords_dosenbach_2010
* function_str : def node_gen(coords, parcel_list, label_names, dir_path, ID, parc, atlas_select, uatlas_select):
    try:
        import cPickle as pickle
    except ImportError:
        import _pickle as pickle
    from pynets import nodemaker
    pick_dump = False

    if parc is True:
        [net_parcels_map_nifti, _] = nodemaker.create_parcel_atlas(parcel_list)
        vox_list = []
        for i in range(len(parcel_list)):
            vox_list.append(np.count_nonzero(parcel_list[i].get_data()))
        vox_array = np.array(vox_list).astype('float64')
    else:
        net_parcels_map_nifti = None
        vox_array = None
        print('No additional masking...')

    coords = list(tuple(x) for x in coords)
    if pick_dump is True:
        # Save coords to pickle
        coord_path = "%s%s" % (dir_path, '/atlas_coords_wb.pkl')
        with open(coord_path, 'wb') as f:
            pickle.dump(coords, f, protocol=2)
        # Save labels to pickle
        labels_path = "%s%s" % (dir_path, '/atlas_labelnames_wb.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(label_names, f, protocol=2)

    return net_parcels_map_nifti, coords, label_names, atlas_select, uatlas_select, vox_array

* label_names : ["inf cerebellum' 155", "inf cerebellum' 150", "inf cerebellum' 151", "inf cerebellum' 140", "inf cerebellum' 131", "inf cerebellum' 122", "inf cerebellum' 121", "inf cerebellum' 110", "lat cerebellum' 128", "lat cerebellum' 113", "lat cerebellum' 109", "lat cerebellum' 98", "med cerebellum' 143", "med cerebellum' 144", "med cerebellum' 138", "med cerebellum' 130", "med cerebellum' 127", "med cerebellum' 120", "ACC' 19", "TPJ' 125", "aPFC' 8", "angular gyrus' 102", "ant insula' 28", "ant insula' 26", "asal ganglia' 71", "asal ganglia' 38", "asal ganglia' 39", "asal ganglia' 30", "dACC' 27", "fusiform' 81", "mFC' 31", "mid insula' 61", "mid insula' 59", "mid insula' 44", "parietal' 97", "parietal' 89", "post cingulate' 80", "post insula' 76", "precuneus' 87", "sup temporal' 100", "temporal' 103", "temporal' 95", "temporal' 78", "thalamus' 57", "thalamus' 58", "thalamus' 47", "vFC' 40", "vFC' 33", "vFC' 25", "vPFC' 18", "ACC' 14", "IPS' 134", "aPFC' 5", "angular gyrus' 124", "angular gyrus' 117", "fusiform' 84", "inf temporal' 91", "inf temporal' 72", "inf temporal' 63", "mPFC' 4", "occipital' 146", "occipital' 141", "occipital' 136", "occipital' 137", "occipital' 92", "post cingulate' 115", "post cingulate' 111", "post cingulate' 108", "post cingulate' 93", "post cingulate' 90", "post cingulate' 73", "precuneus' 132", "precuneus' 112", "precuneus' 105", "precuneus' 94", "precuneus' 85", "sup frontal' 20", "sup frontal' 17", "vlPFC' 15", "vmPFC' 13", "vmPFC' 11", "vmPFC' 7", "vmPFC' 6", "vmPFC' 1", "ACC' 21", "IPL' 107", "IPL' 104", "IPL' 101", "IPL' 96", "IPL' 88", "IPS' 116", "IPS' 114", "aPFC' 2", "aPFC' 3", "dFC' 36", "dFC' 34", "dFC' 29", "dlPFC' 24", "dlPFC' 22", "dlPFC' 16", "post parietal' 99", "vPFC' 23", "vent aPFC' 10", "vent aPFC' 9", "vlPFC' 12", "occipital' 149", "occipital' 148", "occipital' 145", "occipital' 147", "occipital' 142", "occipital' 139", "occipital' 135", "occipital' 133", "occipital' 129", "occipital' 126", "occipital' 118", "occipital' 119", "occipital' 106", "post occipital' 160", "post occipital' 158", "post occipital' 159", "post occipital' 157", "post occipital' 156", "post occipital' 153", "post occipital' 154", "post occipital' 152", "temporal' 123", "SMA' 43", "dFC' 35", "frontal' 45", "frontal' 32", "mid insula' 55", "mid insula' 56", "mid insula' 48", "parietal' 77", "parietal' 74", "parietal' 75", "parietal' 69", "parietal' 66", "parietal' 65", "parietal' 64", "parietal' 62", "parietal' 54", "parietal' 50", "post insula' 70", "post parietal' 79", "pre-SMA' 41", "precentral gyrus' 67", "precentral gyrus' 53", "precentral gyrus' 52", "precentral gyrus' 51", "precentral gyrus' 49", "precentral gyrus' 46", "sup parietal' 86", "temporal' 82", "temporal' 83", "temporal' 68", "temporal' 60", "vFC' 42", "vFC' 37"]
* mask : None
* par_max : None
* parc : False
* parcel_list : None
* uatlas_select : None


Execution Outputs
-----------------


* atlas_select : coords_dosenbach_2010
* coords : [(18, -81, -33), (-21, -79, -33), (-6, -79, -33), (33, -73, -30), (-34, -67, -29), (32, -61, -31), (-25, -60, -34), (-37, -54, -37), (21, -64, -22), (-34, -57, -24), (-24, -54, -21), (-28, -44, -25), (5, -75, -11), (14, -75, -21), (-11, -72, -14), (1, -66, -24), (-16, -64, -21), (-6, -60, -15), (-2, 30, 27), (-52, -63, 15), (27, 49, 26), (-41, -47, 29), (-36, 18, 2), (38, 21, -1), (11, -24, 2), (-20, 6, 7), (14, 6, 7), (-6, 17, 34), (9, 20, 34), (54, -31, -18), (0, 15, 45), (-30, -14, 1), (32, -12, 2), (37, -2, -3), (-55, -44, 30), (58, -41, 20), (-4, -31, -4), (-30, -28, 9), (8, -40, 50), (42, -46, 21), (-59, -47, 11), (43, -43, 8), (51, -30, 5), (-12, -12, 6), (11, -12, 6), (-12, -3, 13), (-48, 6, 1), (-46, 10, 14), (51, 23, 8), (34, 32, 7), (9, 39, 20), (-36, -69, 40), (-25, 51, 27), (-48, -63, 35), (51, -59, 34), (28, -37, -15), (-61, -41, -2), (-59, -25, -15), (52, -15, -13), (0, 51, 32), (-42, -76, 26), (-2, -75, 32), (-9, -72, 41), (45, -72, 29), (-28, -42, -11), (-11, -58, 17), (10, -55, 17), (-5, -52, 17), (-5, -43, 25), (-8, -41, 3), (1, -26, 31), (11, -68, 42), (-6, -56, 29), (5, -50, 33), (9, -43, 25), (-3, -38, 45), (-16, 29, 54), (23, 33, 47), (46, 39, -15), (8, 42, -5), (-11, 45, 17), (-6, 50, -1), (9, 51, 16), (6, 64, 3), (-1, 28, 40), (44, -52, 47), (-53, -50, 39), (-48, -47, 49), (54, -44, 43), (-41, -40, 42), (32, -59, 41), (-32, -58, 46), (29, 57, 18), (-29, 57, 10), (-42, 7, 36), (44, 8, 34), (40, 17, 40), (-44, 27, 33), (46, 28, 31), (40, 36, 29), (-35, -46, 48), (-52, 28, 17), (-43, 47, 2), (42, 48, -3), (39, 42, 16), (20, -78, -2), (15, -77, 32), (-16, -76, 33), (9, -76, 14), (-29, -75, 28), (29, -73, 29), (39, -71, 13), (17, -68, 20), (19, -66, -1), (-44, -63, -7), (-34, -60, -5), (36, -60, -8), (-18, -50, 1), (-4, -94, 12), (13, -91, 2), (27, -91, 2), (-29, -88, 8), (-37, -83, -2), (29, -81, 14), (33, -81, -2), (-5, -80, 9), (46, -62, 5), (0, -1, 52), (60, 8, 34), (53, -3, 32), (58, 11, 14), (33, -12, 16), (-36, -12, 15), (-42, -3, 11), (-24, -30, 64), (18, -27, 62), (-38, -27, 60), (41, -23, 55), (-55, -22, 38), (46, -20, 45), (-47, -18, 50), (-38, -15, 59), (-47, -12, 36), (-26, -8, 54), (42, -24, 17), (-41, -31, 48), (10, 5, 51), (-54, -22, 22), (44, -11, 38), (-54, -9, 23), (46, -8, 24), (-44, -6, 49), (58, -3, 17), (34, -39, 65), (-41, -37, 16), (-53, -37, 13), (-54, -22, 9), (59, -13, 8), (43, 1, 12), (-55, 7, 23)]
* label_names : ["inf cerebellum' 155", "inf cerebellum' 150", "inf cerebellum' 151", "inf cerebellum' 140", "inf cerebellum' 131", "inf cerebellum' 122", "inf cerebellum' 121", "inf cerebellum' 110", "lat cerebellum' 128", "lat cerebellum' 113", "lat cerebellum' 109", "lat cerebellum' 98", "med cerebellum' 143", "med cerebellum' 144", "med cerebellum' 138", "med cerebellum' 130", "med cerebellum' 127", "med cerebellum' 120", "ACC' 19", "TPJ' 125", "aPFC' 8", "angular gyrus' 102", "ant insula' 28", "ant insula' 26", "asal ganglia' 71", "asal ganglia' 38", "asal ganglia' 39", "asal ganglia' 30", "dACC' 27", "fusiform' 81", "mFC' 31", "mid insula' 61", "mid insula' 59", "mid insula' 44", "parietal' 97", "parietal' 89", "post cingulate' 80", "post insula' 76", "precuneus' 87", "sup temporal' 100", "temporal' 103", "temporal' 95", "temporal' 78", "thalamus' 57", "thalamus' 58", "thalamus' 47", "vFC' 40", "vFC' 33", "vFC' 25", "vPFC' 18", "ACC' 14", "IPS' 134", "aPFC' 5", "angular gyrus' 124", "angular gyrus' 117", "fusiform' 84", "inf temporal' 91", "inf temporal' 72", "inf temporal' 63", "mPFC' 4", "occipital' 146", "occipital' 141", "occipital' 136", "occipital' 137", "occipital' 92", "post cingulate' 115", "post cingulate' 111", "post cingulate' 108", "post cingulate' 93", "post cingulate' 90", "post cingulate' 73", "precuneus' 132", "precuneus' 112", "precuneus' 105", "precuneus' 94", "precuneus' 85", "sup frontal' 20", "sup frontal' 17", "vlPFC' 15", "vmPFC' 13", "vmPFC' 11", "vmPFC' 7", "vmPFC' 6", "vmPFC' 1", "ACC' 21", "IPL' 107", "IPL' 104", "IPL' 101", "IPL' 96", "IPL' 88", "IPS' 116", "IPS' 114", "aPFC' 2", "aPFC' 3", "dFC' 36", "dFC' 34", "dFC' 29", "dlPFC' 24", "dlPFC' 22", "dlPFC' 16", "post parietal' 99", "vPFC' 23", "vent aPFC' 10", "vent aPFC' 9", "vlPFC' 12", "occipital' 149", "occipital' 148", "occipital' 145", "occipital' 147", "occipital' 142", "occipital' 139", "occipital' 135", "occipital' 133", "occipital' 129", "occipital' 126", "occipital' 118", "occipital' 119", "occipital' 106", "post occipital' 160", "post occipital' 158", "post occipital' 159", "post occipital' 157", "post occipital' 156", "post occipital' 153", "post occipital' 154", "post occipital' 152", "temporal' 123", "SMA' 43", "dFC' 35", "frontal' 45", "frontal' 32", "mid insula' 55", "mid insula' 56", "mid insula' 48", "parietal' 77", "parietal' 74", "parietal' 75", "parietal' 69", "parietal' 66", "parietal' 65", "parietal' 64", "parietal' 62", "parietal' 54", "parietal' 50", "post insula' 70", "post parietal' 79", "pre-SMA' 41", "precentral gyrus' 67", "precentral gyrus' 53", "precentral gyrus' 52", "precentral gyrus' 51", "precentral gyrus' 49", "precentral gyrus' 46", "sup parietal' 86", "temporal' 82", "temporal' 83", "temporal' 68", "temporal' 60", "vFC' 42", "vFC' 37"]
* net_parcels_map_nifti : None
* uatlas_select : None
* vox_array : None


Runtime info
------------


* duration : 0.013192
* hostname : Dereks-MAC
* prev_wd : /private/tmp
* working_dir : /Users/PSYC-dap3463/Applications/PyNets/tests/examples/997/Wf_single_subject_997_1000/Meta_wf_997/functional_connectometry_997/node_gen_node


Environment
~~~~~~~~~~~


* AWARE : /Users/PSYC-dap3463/Box Sync/MDL Projects/Derek/Mobile_data/AWS_credentials
* Apple_PubSub_Socket_Render : /private/tmp/com.apple.launchd.GmFiGdztnp/Render
* CAMINO : /Applications/camino
* CC : /usr/local/opt/llvm/bin/clang
* CPPFLAGS : -I/usr/local/opt/llvm/include 
* CXX : /usr/local/opt/llvm/bin/clang++
* DISPLAY : /private/tmp/com.apple.launchd.dAoTcOiHX7/org.macosforge.xquartz:0
* DYLD_LIBRARY_PATH : /opt/X11/lib/flat_namespace
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
* GLOBUS_LOCATION : /Library/Globus
* HOME : /Users/PSYC-dap3463
* LANG : en_US.UTF-8
* LDFLAGS : -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib 
* LOCAL_DIR : /Applications/freesurfer/local
* LOGNAME : PSYC-dap3463
* MINC_BIN_DIR : /Applications/freesurfer/mni/bin
* MINC_LIB_DIR : /Applications/freesurfer/mni/lib
* MNI_DATAPATH : /Applications/freesurfer/mni/data
* MNI_DIR : /Applications/freesurfer/mni
* MNI_PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.10.0
* NIPYPE : /usr/local/lib/python2.7/site-packages/nipype
* OLDPWD : /Users/PSYC-dap3463/Applications/pyAWARE
* OS : Darwin
* PATH : /opt/local/bin:/opt/local/sbin:/Users/PSYC-dap3463/perl5/bin:/Applications/camino/bin:/usr/local/opt/qt5/bin:/Library/Java/JavaVirtualMachines/jdk1.8.0_121.jdk/Contents/Home/bin:/Applications/FSLeyes.app/Contents/MacOS/fsleyes:/usr/local/Cellar/python3:/usr/local/bin:/usr/local/sbin:/usr/local/fsl/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/opt/sqlite/bin:/opt/local/bin:/opt/local/sbin:/usr/local/opt/llvm/bin:/Applications/anaconda2/bin:/Users/PSYC-dap3463/Library/Python/3.6/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/Library/Globus/bin:/Library/Globus/sbin:/Library/TeX/texbin:/Users/PSYC-dap3463/abin:/Library/Globus/bin:/Users/PSYC-dap3463/perl5/bin:/Applications/camino/bin:/usr/local/opt/qt5/bin:/Library/Java/JavaVirtualMachines/jdk1.8.0_121.jdk/Contents/Home/bin:/Applications/FSLeyes.app/Contents/MacOS/fsleyes:/usr/local/Cellar/python3:/usr/local/bin:/usr/local/sbin:/usr/local/fsl/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/opt/sqlite/bin:/opt/local/bin:/opt/local/sbin:/usr/local/opt/llvm/bin:/Applications/anaconda2/bin:/Users/PSYC-dap3463/Library/Python/3.6/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/Library/Globus/bin:/Library/Globus/sbin:/Library/TeX/texbin:/Users/PSYC-dap3463/abin
* PERL5LIB : /Users/PSYC-dap3463/perl5/lib/perl5:/Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.10.0
* PERL_LOCAL_LIB_ROOT : /Users/PSYC-dap3463/perl5
* PERL_MB_OPT : --install_base "/Users/PSYC-dap3463/perl5"
* PERL_MM_OPT : INSTALL_BASE=/Users/PSYC-dap3463/perl5
* PWD : /tmp
* SHELL : /bin/bash
* SHLVL : 2
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.QrYdJrCXN6/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 388.1.1
* TERM_SESSION_ID : 181D75C4-E4B4-4F37-A51D-24D23C03C98A
* TMPDIR : /var/folders/dk/9n8ng3n93q95kvx0zppcdtlh0000gp/T/
* USER : PSYC-dap3463
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/bin/pynets_run.py
* __CF_USER_TEXT_ENCODING : 0x1F6:0x0:0x0
* __PYVENV_LAUNCHER__ : /usr/local/Cellar/python/3.6.5/bin/python3.6

