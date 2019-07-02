Node: meta_wf_0021001 (fmri_connectometry_0021001 (fetch_nodes_and_labels_node (utility)
========================================================================================


 Hierarchy : wf_single_sub_0021001_981.meta_wf_0021001.fmri_connectometry_0021001.fetch_nodes_and_labels_node
 Exec ID : fetch_nodes_and_labels_node


Original Inputs
---------------


* atlas : None
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
    from pynets import utils, nodemaker
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
            if ref_txt is not None and op.exists(ref_txt):
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                labels = dict_df['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas, '.txt')
                    if op.exists(ref_txt):
                        try:
                            dict_df = pd.read_csv(ref_txt, sep="\t", header=None, names=["Index", "Region"])
                            labels = dict_df['Region'].tolist()
                        except:
                            print("WARNING: label names from label reference file failed to populate or are invalid. "
                                  "Attempting AAL naming...")
                            try:
                                labels = nodemaker.AAL_naming(coords)
                            except:
                                print('AAL reference labeling failed!')
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

* in_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002.nii.gz
* parc : True
* ref_txt : DesikanKlein2012.txt
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz
* use_AAL_naming : False

Execution Inputs
----------------


* atlas : None
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
    from pynets import utils, nodemaker
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
            if ref_txt is not None and op.exists(ref_txt):
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                labels = dict_df['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas, '.txt')
                    if op.exists(ref_txt):
                        try:
                            dict_df = pd.read_csv(ref_txt, sep="\t", header=None, names=["Index", "Region"])
                            labels = dict_df['Region'].tolist()
                        except:
                            print("WARNING: label names from label reference file failed to populate or are invalid. "
                                  "Attempting AAL naming...")
                            try:
                                labels = nodemaker.AAL_naming(coords)
                            except:
                                print('AAL reference labeling failed!')
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

* in_file : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/002.nii.gz
* parc : True
* ref_txt : DesikanKlein2012.txt
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz
* use_AAL_naming : False


Execution Outputs
-----------------


* atlas : DesikanKlein2012
* coords : [[-39.40774665  -9.04783285  44.00269397]
 [ -7.43663158 -59.25827368  37.86349474]
 [ -5.29388682  39.69537381   6.23667906]
 [-13.39120945 -16.23295962  14.89086219]
 [-34.         -34.          -5.75      ]
 [-27.66990809 -59.83579899 -38.62715683]
 [-20.38812245 -47.51814791 -35.86346142]
 [-37.63897527  42.55678958  18.29409151]
 [-12.9633117   33.74328278  41.0403874 ]
 [-11.61810386 -18.42696522   6.79313959]
 [-12.63600396   9.12141444  10.67334322]
 [-24.48445284   1.78268339   1.53719292]
 [-19.39536138  -3.38457389  -0.67475728]
 [ -1.1030303  -10.01212121  -1.35151515]
 [ -4.27663331 -39.44496763 -32.73866981]
 [ -6.3191898  -27.34677842 -34.81255457]
 [-25.47759724 -22.06794682 -12.82964057]
 [-21.47991968  -5.22188755 -18.10341365]
 [-36.20065008   1.58607326  -0.25890736]
 [-46.02027559  13.76712598  12.43307087]
 [-46.38786232  32.45126812   5.49402174]
 [ -9.7127572  -81.95144033   6.20534979]
 [ -2.06963788 -25.01671309   9.76044568]
 [ -9.28142077  11.34699454  -7.08469945]
 [ -9.84567901 -15.81265432  -9.73657407]
 [-28.          -2.          -8.        ]
 [-40.52919559  32.92976725 -10.45406288]
 [ 14.35376532 -15.67192061  15.144892  ]
 [ 25.46987952 -15.01204819 -11.63855422]
 [ -1.68587896 -47.99711816 -50.91066282]
 [ -1.74736842 -50.93684211 -26.46315789]
 [ 12.1919403  -17.78547264   7.17114428]
 [ 14.20635674  10.36005693  11.24691651]
 [ 25.62502444   2.43147605   1.5769306 ]
 [ 20.5507772   -2.83523316  -0.70777202]
 [ 26.30807397 -20.53450609 -12.81235904]
 [ 21.93        -4.24888889 -18.41777778]
 [  9.19578313  11.24698795  -6.40060241]
 [-24.55775125 -62.69099855  50.64849169]
 [ 10.77466562 -15.93674272  -9.79040126]
 [-22.03214562 -31.49922541 -17.83152595]
 [-50.25136791 -10.39928915  -6.63124912]
 [-53.12214603 -38.11717278  34.63077387]
 [-44.98768633 -21.46208684   8.45106935]
 [ -3.46808511   2.76595745 -10.91489362]
 [  6.45205479   3.70547945 -11.23287671]
 [ -4.46764253 -50.63995891 -13.20005136]
 [ -3.64342857 -66.68971429 -23.60742857]
 [ -4.04270938 -57.24758091 -38.75942609]
 [ -6.17316943 -25.68163193  57.79975058]
 [-57.46055697 -27.80877621 -12.65871519]
 [  5.4625651   21.54329427  28.04296875]
 [ 37.53156103  12.98349921  47.44722368]
 [  9.32470444 -79.53383612  23.1946596 ]
 [ 22.6098635   -4.56230735 -31.95640687]
 [ 34.79281102 -43.04760528 -20.99951817]
 [ 46.24613187 -65.41561533  30.40984289]
 [ 49.02416244 -28.17928934 -27.29678511]
 [  7.90322581 -44.98354839  19.30645161]
 [ 32.80915227 -86.42231717   2.04755712]
 [ 22.2122093   36.89347337 -17.91757314]
 [ 13.6988191  -66.42386874  -5.03918905]
 [  5.23006135  37.48432175 -17.26993865]
 [ 58.32844866 -25.23297708 -12.50093654]
 [ 24.10399334 -29.37895175 -18.43427621]
 [  7.53763026 -24.85488228  58.69490544]
 [ 48.40639193  15.13254836  13.54583684]
 [ 42.43845252  34.29503712 -11.10746385]
 [ 48.22581281  32.69162562   4.66876847]
 [ 12.23624954 -79.71576227   7.05204873]
 [ 45.29406347 -21.88872143  43.83195177]
 [  5.79049467 -16.93622696  39.24515034]
 [ 41.46036719  -7.7177768   44.17087537]
 [  9.59039273 -57.63299253  37.98847777]
 [ -5.28925825  20.86876706  31.68097246]
 [ 39.24827087  43.88000643  17.95375583]
 [ 13.54483621  34.52466355  38.67653984]
 [ 25.51007813 -60.26367188  53.30015625]
 [ 51.52368964  -6.54267157  -8.07164795]
 [ 54.18698939 -35.04665191  36.22738032]
 [-35.83066274  12.09024484  47.08011793]
 [  5.86218182  37.532        2.95709091]
 [ 46.06022409 -17.97478992   8.11204482]
 [ 37.50006131   2.23874923   0.19264255]
 [ -7.08703156 -81.85428994  21.14423077]
 [-22.87226126  -5.03183134 -32.04257958]
 [-36.15644349 -46.70172542 -20.03937918]
 [-41.55397548 -70.55625079  32.08886361]
 [-48.32543193 -28.42695232 -28.30117484]
 [ -6.3807937  -46.63617086  19.25295365]
 [-29.8504807  -89.22470943   1.50272636]
 [-22.22757966  36.75416755 -17.43680101]
 [-13.29384318 -67.50893365  -5.87891203]
 [ -5.10974742  41.00942725 -16.83137673]
 [-43.53290192 -23.8616872   43.53906128]
 [ -5.21337127 -18.39260313  39.69630156]]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/DesikanKlein2012
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* networks_list : None
* par_max : 96
* parcel_list : [<nibabel.nifti1.Nifti1Image object at 0x105ae3198>, <nibabel.nifti1.Nifti1Image object at 0x105ae3ac8>, <nibabel.nifti1.Nifti1Image object at 0x105ae3b70>, <nibabel.nifti1.Nifti1Image object at 0x105ae3c18>, <nibabel.nifti1.Nifti1Image object at 0x105ae3cc0>, <nibabel.nifti1.Nifti1Image object at 0x105ae3d68>, <nibabel.nifti1.Nifti1Image object at 0x105ae3e10>, <nibabel.nifti1.Nifti1Image object at 0x105ae3eb8>, <nibabel.nifti1.Nifti1Image object at 0x105ae3f60>, <nibabel.nifti1.Nifti1Image object at 0x105adb048>, <nibabel.nifti1.Nifti1Image object at 0x105adb0f0>, <nibabel.nifti1.Nifti1Image object at 0x105adb198>, <nibabel.nifti1.Nifti1Image object at 0x105adb240>, <nibabel.nifti1.Nifti1Image object at 0x105adb2e8>, <nibabel.nifti1.Nifti1Image object at 0x105adb390>, <nibabel.nifti1.Nifti1Image object at 0x105adb438>, <nibabel.nifti1.Nifti1Image object at 0x105adb4e0>, <nibabel.nifti1.Nifti1Image object at 0x105adb588>, <nibabel.nifti1.Nifti1Image object at 0x105adb630>, <nibabel.nifti1.Nifti1Image object at 0x105adb6d8>, <nibabel.nifti1.Nifti1Image object at 0x105adb780>, <nibabel.nifti1.Nifti1Image object at 0x105adb828>, <nibabel.nifti1.Nifti1Image object at 0x105adb8d0>, <nibabel.nifti1.Nifti1Image object at 0x105adb978>, <nibabel.nifti1.Nifti1Image object at 0x105adba20>, <nibabel.nifti1.Nifti1Image object at 0x105adbac8>, <nibabel.nifti1.Nifti1Image object at 0x105adbb70>, <nibabel.nifti1.Nifti1Image object at 0x105adbc18>, <nibabel.nifti1.Nifti1Image object at 0x105adbcc0>, <nibabel.nifti1.Nifti1Image object at 0x105adbd68>, <nibabel.nifti1.Nifti1Image object at 0x105adbe10>, <nibabel.nifti1.Nifti1Image object at 0x105adbeb8>, <nibabel.nifti1.Nifti1Image object at 0x105adbf60>, <nibabel.nifti1.Nifti1Image object at 0x105af9048>, <nibabel.nifti1.Nifti1Image object at 0x105af90f0>, <nibabel.nifti1.Nifti1Image object at 0x105af9198>, <nibabel.nifti1.Nifti1Image object at 0x105af9240>, <nibabel.nifti1.Nifti1Image object at 0x105af92e8>, <nibabel.nifti1.Nifti1Image object at 0x105af9390>, <nibabel.nifti1.Nifti1Image object at 0x105af9438>, <nibabel.nifti1.Nifti1Image object at 0x105af94e0>, <nibabel.nifti1.Nifti1Image object at 0x105af9588>, <nibabel.nifti1.Nifti1Image object at 0x105af9630>, <nibabel.nifti1.Nifti1Image object at 0x105af96d8>, <nibabel.nifti1.Nifti1Image object at 0x105af9780>, <nibabel.nifti1.Nifti1Image object at 0x105af9828>, <nibabel.nifti1.Nifti1Image object at 0x105af98d0>, <nibabel.nifti1.Nifti1Image object at 0x105af9978>, <nibabel.nifti1.Nifti1Image object at 0x105af9a20>, <nibabel.nifti1.Nifti1Image object at 0x105af9ac8>, <nibabel.nifti1.Nifti1Image object at 0x105af9b70>, <nibabel.nifti1.Nifti1Image object at 0x105af9c18>, <nibabel.nifti1.Nifti1Image object at 0x105af9cc0>, <nibabel.nifti1.Nifti1Image object at 0x105af9d68>, <nibabel.nifti1.Nifti1Image object at 0x105af9e10>, <nibabel.nifti1.Nifti1Image object at 0x105af9eb8>, <nibabel.nifti1.Nifti1Image object at 0x105af9f60>, <nibabel.nifti1.Nifti1Image object at 0x105b0a048>, <nibabel.nifti1.Nifti1Image object at 0x105b0a0f0>, <nibabel.nifti1.Nifti1Image object at 0x105b0a198>, <nibabel.nifti1.Nifti1Image object at 0x105b0a240>, <nibabel.nifti1.Nifti1Image object at 0x105b0a2e8>, <nibabel.nifti1.Nifti1Image object at 0x105b0a390>, <nibabel.nifti1.Nifti1Image object at 0x105b0a438>, <nibabel.nifti1.Nifti1Image object at 0x105b0a4e0>, <nibabel.nifti1.Nifti1Image object at 0x105b0a588>, <nibabel.nifti1.Nifti1Image object at 0x105b0a630>, <nibabel.nifti1.Nifti1Image object at 0x105b0a6d8>, <nibabel.nifti1.Nifti1Image object at 0x105b0a780>, <nibabel.nifti1.Nifti1Image object at 0x105b0a828>, <nibabel.nifti1.Nifti1Image object at 0x105b0a8d0>, <nibabel.nifti1.Nifti1Image object at 0x105b0a978>, <nibabel.nifti1.Nifti1Image object at 0x105b0aa20>, <nibabel.nifti1.Nifti1Image object at 0x105b0aac8>, <nibabel.nifti1.Nifti1Image object at 0x105b0ab70>, <nibabel.nifti1.Nifti1Image object at 0x105b0ac18>, <nibabel.nifti1.Nifti1Image object at 0x105b0acc0>, <nibabel.nifti1.Nifti1Image object at 0x105b0ad68>, <nibabel.nifti1.Nifti1Image object at 0x105b0ae10>, <nibabel.nifti1.Nifti1Image object at 0x105b0aeb8>, <nibabel.nifti1.Nifti1Image object at 0x105b0af60>, <nibabel.nifti1.Nifti1Image object at 0x1c26046048>, <nibabel.nifti1.Nifti1Image object at 0x1c260460f0>, <nibabel.nifti1.Nifti1Image object at 0x1c26046198>, <nibabel.nifti1.Nifti1Image object at 0x1c26046240>, <nibabel.nifti1.Nifti1Image object at 0x1c260462e8>, <nibabel.nifti1.Nifti1Image object at 0x1c26046390>, <nibabel.nifti1.Nifti1Image object at 0x1c26046438>, <nibabel.nifti1.Nifti1Image object at 0x1c260464e0>, <nibabel.nifti1.Nifti1Image object at 0x1c26046588>, <nibabel.nifti1.Nifti1Image object at 0x1c26046630>, <nibabel.nifti1.Nifti1Image object at 0x1c260466d8>, <nibabel.nifti1.Nifti1Image object at 0x1c26046780>, <nibabel.nifti1.Nifti1Image object at 0x1c26046828>, <nibabel.nifti1.Nifti1Image object at 0x1c260468d0>, <nibabel.nifti1.Nifti1Image object at 0x1c26046978>]
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 44.641134
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/fmri/wf_single_subject_0021001/wf_single_sub_0021001_981/meta_wf_0021001/fmri_connectometry_0021001/fetch_nodes_and_labels_node


Environment
~~~~~~~~~~~


* ANTSPATH : /Users/derekpisner/bin/ants/bin/
* Apple_PubSub_Socket_Render : /private/tmp/com.apple.launchd.VKfenSaB7x/Render
* CONDA_DEFAULT_ENV : base
* CONDA_EXE : /usr/local/anaconda3/bin/conda
* CONDA_PREFIX : /usr/local/anaconda3
* CONDA_PROMPT_MODIFIER : (base) 
* CONDA_SHLVL : 1
* CPPFLAGS : -I/usr/local/opt/libxml2/include
* DISPLAY : dpys:0.0
* DYLD_LIBRARY_PATH : /Applications/freesurfer/lib/gcc/lib::/opt/X11/lib/flat_namespace
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
* LANG : en_US.UTF-8
* LDFLAGS : -L/usr/local/opt/libxml2/lib
* LOCAL_DIR : /Applications/freesurfer/local
* LOGNAME : derekpisner
* MINC_BIN_DIR : /Applications/freesurfer/mni/bin
* MINC_LIB_DIR : /Applications/freesurfer/mni/lib
* MNI_DATAPATH : /Applications/freesurfer/mni/data
* MNI_DIR : /Applications/freesurfer/mni
* MNI_PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* OLDPWD : /Users/derekpisner/Applications/PyNets/tests
* OS : Darwin
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/Users/derekpisner/anaconda3/bin:/Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin:/Users/derekpisner/abin
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner/Applications/PyNets
* SHELL : /bin/bash
* SHLVL : 3
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.qmAkE8F40f/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.1.1
* TERM_SESSION_ID : 6432F315-D86A-4D51-A77C-DB02F4938E15
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0

