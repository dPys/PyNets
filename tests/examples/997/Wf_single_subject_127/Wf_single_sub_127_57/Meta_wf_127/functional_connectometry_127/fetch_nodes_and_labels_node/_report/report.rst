Node: Meta_wf_127 (functional_connectometry_127 (fetch_nodes_and_labels_node (utility)
======================================================================================


 Hierarchy : Wf_single_sub_127_57.Meta_wf_127.functional_connectometry_127.fetch_nodes_and_labels_node
 Exec ID : fetch_nodes_and_labels_node


Original Inputs
---------------


* atlas_select : atlas_aal
* clustering : <undefined>
* function_str : def fetch_nodes_and_labels(atlas_select, uatlas_select, ref_txt, parc, in_file, use_AAL_naming, clustering=False):
    from pynets import utils, nodemaker
    import pandas as pd
    import time
    from pathlib import Path
    import os.path as op

    base_path = utils.get_file()
    # Test if atlas_select is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009',
                            'atlas_talairach_gyrus', 'atlas_talairach_ba', 'atlas_talairach_lobe']
    nilearn_coords_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
    nilearn_prob_atlases = ['atlas_msdl', 'atlas_pauli_2017']
    if uatlas_select is None and atlas_select in nilearn_parc_atlases:
        [label_names, networks_list, uatlas_select] = nodemaker.nilearn_atlas_helper(atlas_select, parc)
        if uatlas_select:
            if not isinstance(uatlas_select, str):
                nib.save(uatlas_select, "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz'))
                uatlas_select = "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz')
            [coords, _, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas_select)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas_select, ' not found!'))
    elif uatlas_select is None and parc is False and atlas_select in nilearn_coords_atlases:
        print('Fetching coords and labels from nilearn coordsinate-based atlas library...')
        # Fetch nilearn atlas coords
        [coords, _, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        parcel_list = None
        par_max = None
    elif uatlas_select is None and parc is False and atlas_select in nilearn_prob_atlases:
        from nilearn.plotting import find_probabilistic_atlas_cut_coords
        print('Fetching coords and labels from nilearn probabilistic atlas library...')
        # Fetch nilearn atlas coords
        [label_names, networks_list, uatlas_select] = nodemaker.nilearn_atlas_helper(atlas_select, parc)
        coords = find_probabilistic_atlas_cut_coords(maps_img=uatlas_select)
        if uatlas_select:
            if not isinstance(uatlas_select, str):
                nib.save(uatlas_select, "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz'))
                uatlas_select = "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz')
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas_select, ' not found!'))
        par_max = None
    elif uatlas_select:
        if clustering is True:
            while True:
                if op.isfile(uatlas_select):
                    break
                else:
                    print('Waiting for atlas file...')
                    time.sleep(15)
        atlas_select = uatlas_select.split('/')[-1].split('.')[0]
        try:
            # Fetch user-specified atlas coords
            [coords, atlas_select, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas_select)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
            # Describe user atlas coords
            print("%s%s%s%s" % ('\n', atlas_select, ' comes with {0} '.format(par_max), 'parcels\n'))
        except ValueError:
            print('\n\nError: Either you have specified the name of a nilearn atlas that does not exist or you have not '
                  'supplied a 3d atlas parcellation image!\n\n')
            parcel_list = None
            par_max = None
            coords = None
        label_names = None
        networks_list = None
    else:
        networks_list = None
        label_names = None
        parcel_list = None
        par_max = None
        coords = None

    # Labels prep
    if atlas_select:
        if label_names:
            pass
        else:
            if ref_txt is not None and op.exists(ref_txt):
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                label_names = dict_df['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas_select, '.txt')
                    if op.exists(ref_txt):
                        try:
                            dict_df = pd.read_csv(ref_txt, sep="\t", header=None, names=["Index", "Region"])
                            label_names = dict_df['Region'].tolist()
                            #print(label_names)
                        except:
                            print("WARNING: label names from label reference file failed to populate or are invalid. "
                                  "Attempting AAL naming...")
                            try:
                                label_names = nodemaker.AAL_naming(coords)
                                # print(label_names)
                            except:
                                print('AAL reference labeling failed!')
                                label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        if use_AAL_naming is True:
                            try:
                                label_names = nodemaker.AAL_naming(coords)
                                # print(label_names)
                            except:
                                print('AAL reference labeling failed!')
                                label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                        else:
                            print('Using generic numbering labels...')
                            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                except:
                    print("Label reference file not found. Attempting AAL naming...")
                    if use_AAL_naming is True:
                        try:
                            label_names = nodemaker.AAL_naming(coords)
                            #print(label_names)
                        except:
                            print('AAL reference labeling failed!')
                            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        print('Using generic numbering labels...')
                        label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    else:
        print('WARNING: No labels available since atlas name is not specified!')

    print("%s%s" % ('Labels:\n', label_names))
    atlas_name = atlas_select
    dir_path = utils.do_dir_path(atlas_select, in_file)

    return label_names, coords, atlas_name, networks_list, parcel_list, par_max, uatlas_select, dir_path

* in_file : /Users/derekpisner/Applications/PyNets/tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz
* parc : False
* ref_txt : None
* uatlas_select : None
* use_AAL_naming : False

Execution Inputs
----------------


* atlas_select : atlas_aal
* clustering : <undefined>
* function_str : def fetch_nodes_and_labels(atlas_select, uatlas_select, ref_txt, parc, in_file, use_AAL_naming, clustering=False):
    from pynets import utils, nodemaker
    import pandas as pd
    import time
    from pathlib import Path
    import os.path as op

    base_path = utils.get_file()
    # Test if atlas_select is a nilearn atlas. If so, fetch coords, labels, and/or networks.
    nilearn_parc_atlases = ['atlas_harvard_oxford', 'atlas_aal', 'atlas_destrieux_2009',
                            'atlas_talairach_gyrus', 'atlas_talairach_ba', 'atlas_talairach_lobe']
    nilearn_coords_atlases = ['coords_power_2011', 'coords_dosenbach_2010']
    nilearn_prob_atlases = ['atlas_msdl', 'atlas_pauli_2017']
    if uatlas_select is None and atlas_select in nilearn_parc_atlases:
        [label_names, networks_list, uatlas_select] = nodemaker.nilearn_atlas_helper(atlas_select, parc)
        if uatlas_select:
            if not isinstance(uatlas_select, str):
                nib.save(uatlas_select, "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz'))
                uatlas_select = "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz')
            [coords, _, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas_select)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas_select, ' not found!'))
    elif uatlas_select is None and parc is False and atlas_select in nilearn_coords_atlases:
        print('Fetching coords and labels from nilearn coordsinate-based atlas library...')
        # Fetch nilearn atlas coords
        [coords, _, networks_list, label_names] = nodemaker.fetch_nilearn_atlas_coords(atlas_select)
        parcel_list = None
        par_max = None
    elif uatlas_select is None and parc is False and atlas_select in nilearn_prob_atlases:
        from nilearn.plotting import find_probabilistic_atlas_cut_coords
        print('Fetching coords and labels from nilearn probabilistic atlas library...')
        # Fetch nilearn atlas coords
        [label_names, networks_list, uatlas_select] = nodemaker.nilearn_atlas_helper(atlas_select, parc)
        coords = find_probabilistic_atlas_cut_coords(maps_img=uatlas_select)
        if uatlas_select:
            if not isinstance(uatlas_select, str):
                nib.save(uatlas_select, "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz'))
                uatlas_select = "%s%s%s" % ('/tmp/', atlas_select, '.nii.gz')
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
        else:
            raise ValueError("%s%s%s" % ('\nERROR: Atlas file for ', atlas_select, ' not found!'))
        par_max = None
    elif uatlas_select:
        if clustering is True:
            while True:
                if op.isfile(uatlas_select):
                    break
                else:
                    print('Waiting for atlas file...')
                    time.sleep(15)
        atlas_select = uatlas_select.split('/')[-1].split('.')[0]
        try:
            # Fetch user-specified atlas coords
            [coords, atlas_select, par_max] = nodemaker.get_names_and_coords_of_parcels(uatlas_select)
            if parc is True:
                parcel_list = nodemaker.gen_img_list(uatlas_select)
            else:
                parcel_list = None
            # Describe user atlas coords
            print("%s%s%s%s" % ('\n', atlas_select, ' comes with {0} '.format(par_max), 'parcels\n'))
        except ValueError:
            print('\n\nError: Either you have specified the name of a nilearn atlas that does not exist or you have not '
                  'supplied a 3d atlas parcellation image!\n\n')
            parcel_list = None
            par_max = None
            coords = None
        label_names = None
        networks_list = None
    else:
        networks_list = None
        label_names = None
        parcel_list = None
        par_max = None
        coords = None

    # Labels prep
    if atlas_select:
        if label_names:
            pass
        else:
            if ref_txt is not None and op.exists(ref_txt):
                dict_df = pd.read_csv(ref_txt, sep=" ", header=None, names=["Index", "Region"])
                label_names = dict_df['Region'].tolist()
            else:
                try:
                    ref_txt = "%s%s%s%s" % (str(Path(base_path).parent), '/labelcharts/', atlas_select, '.txt')
                    if op.exists(ref_txt):
                        try:
                            dict_df = pd.read_csv(ref_txt, sep="\t", header=None, names=["Index", "Region"])
                            label_names = dict_df['Region'].tolist()
                            #print(label_names)
                        except:
                            print("WARNING: label names from label reference file failed to populate or are invalid. "
                                  "Attempting AAL naming...")
                            try:
                                label_names = nodemaker.AAL_naming(coords)
                                # print(label_names)
                            except:
                                print('AAL reference labeling failed!')
                                label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        if use_AAL_naming is True:
                            try:
                                label_names = nodemaker.AAL_naming(coords)
                                # print(label_names)
                            except:
                                print('AAL reference labeling failed!')
                                label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                        else:
                            print('Using generic numbering labels...')
                            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                except:
                    print("Label reference file not found. Attempting AAL naming...")
                    if use_AAL_naming is True:
                        try:
                            label_names = nodemaker.AAL_naming(coords)
                            #print(label_names)
                        except:
                            print('AAL reference labeling failed!')
                            label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
                    else:
                        print('Using generic numbering labels...')
                        label_names = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()
    else:
        print('WARNING: No labels available since atlas name is not specified!')

    print("%s%s" % ('Labels:\n', label_names))
    atlas_name = atlas_select
    dir_path = utils.do_dir_path(atlas_select, in_file)

    return label_names, coords, atlas_name, networks_list, parcel_list, par_max, uatlas_select, dir_path

* in_file : /Users/derekpisner/Applications/PyNets/tests/examples/997/sub-997_ses-01_task-REST_run-01_bold_space-MNI152NLin2009cAsym_preproc_masked.nii.gz
* parc : False
* ref_txt : None
* uatlas_select : None
* use_AAL_naming : False


Execution Outputs
-----------------


* atlas_select : atlas_aal
* coords : [[ -8.32018561 -26.78731632  69.02088167]
 [  7.12679426 -32.94258373  66.79186603]
 [-25.25536481 -21.96351931 -11.38412017]
 [ 28.94503171 -20.99154334 -11.57716702]
 [-45.89007513  28.66666667  12.58283907]
 [ 50.06136681  28.90190609  12.81729428]
 [-55.868879   -34.98421692  -3.58478349]
 [ 57.1599002  -38.56566115  -2.77523248]
 [-21.49284254 -17.29243354 -21.91820041]
 [ 25.14840989 -16.30035336 -21.74381625]
 [-36.24023669  29.49230769 -13.4591716 ]
 [ 40.91036907  30.99472759 -13.25834798]
 [-36.60266667  13.35466667 -35.40533333]
 [ 43.98141892  13.27533784 -33.55405405]
 [-31.39514731 -41.36221837 -21.6152513 ]
 [ 33.66375199 -40.16057234 -21.54689984]
 [-47.37854251  -9.75101215  12.63562753]
 [ 52.38166792  -7.53719008  13.3298272 ]
 [ -6.81740614  48.22696246  28.80546075]
 [  8.75070291  49.52858482  28.92689784]
 [-35.38532463 -67.94314253 -30.26277372]
 [ 38.21450151 -68.37160121 -30.84969789]
 [ -6.52736318  52.25538972  -8.7628524 ]
 [  7.83411215  50.40654206  -8.51635514]
 [-18.79777159  33.4178273   40.99442897]
 [ 21.60483473  29.9097188   42.51307351]
 [-28.56786102 -74.3713355  -39.69055375]
 [ 32.67359471 -70.39678791 -41.20170052]
 [-43.10666122 -47.01920719  45.4229669 ]
 [ 46.29144981 -47.58810409  48.19182156]
 [ -8.11764706 -38.36764706 -19.77941176]
 [ 13.06280193 -35.71014493 -20.59903382]
 [-16.82450675  45.91900312 -14.71858775]
 [ 18.17853561  46.61785356 -15.49448345]
 [-56.14649682 -34.93630573  29.11942675]
 [ 57.27116067 -32.82209833  33.13228586]
 [-14.34311111 -44.66311111 -18.26133333]
 [ 17.88850174 -44.16492451 -19.35423926]
 [-44.42491468 -62.08532423  34.278157  ]
 [ 45.2283105  -61.23858447  37.31506849]
 [-32.639289   -81.99387067  14.78884462]
 [ 37.10644391 -81.00143198  18.09546539]
 [-22.52538371 -60.25855962 -23.48051948]
 [ 25.42952646 -59.53760446 -24.94038997]
 [-11.76715177   9.72349272   8.07068607]
 [ 14.49496982  10.77665996   8.14889336]
 [-31.58561644 -61.07876712 -46.76369863]
 [ 33.92883895 -64.4082397  -49.76029963]
 [ -6.80973451   3.12942478  60.53539823]
 [  8.24630957  -1.09320962  60.51286377]
 [-24.20812686   2.60059465   1.0703667 ]
 [ 27.48496241   3.67857143   1.19360902]
 [-25.02914679 -55.77106518 -49.054584  ]
 [ 25.7729636  -57.63084922 -50.79809359]
 [-23.50909091  -1.94545455 -18.45454545]
 [ 27.05645161  -0.57258065 -18.80645161]
 [-50.03625    -29.291875   -24.519375  ]
 [ 53.37158931 -32.13445851 -23.73783404]
 [-18.06143345  -1.35836177  -1.0443686 ]
 [ 20.92142857  -1.05714286  -1.08571429]
 [-10.32444959 -50.1622248  -47.25144844]
 [ 10.19035847 -50.75154512 -47.62422744]
 [-42.90441932 -23.77543679  47.45683453]
 [ 41.15825268 -26.7726916   51.25398901]
 [-21.79166667 -35.         -43.06944444]
 [ 26.80503145 -35.01886792 -42.64150943]
 [-10.05665722 -77.5490085    5.47308782]
 [ 15.70983342 -74.44922085   8.01934444]
 [ -2.36363636 -40.90909091 -21.81818182]
 [ -6.69614836  35.04992867 -19.29243937]
 [  8.00268456  34.45100671 -19.43624161]
 [ -8.39831224 -80.54345992  26.1721519 ]
 [ 13.12288136 -80.70621469  26.87146893]
 [ -2.         -41.28205128 -13.33333333]
 [-33.76013166  31.49763423  34.12836865]
 [ 37.39502058  31.77141737  32.80180357]
 [ -8.48628193 -56.90192368  47.13213497]
 [  9.68443627 -57.31188725  42.36519608]
 [-15.35929029 -68.67422376  -6.27402661]
 [ 16.0773913  -68.14173913  -5.21565217]
 [ -2.22516556 -53.45695364  -7.61589404]
 [ -5.82479031  34.46598322  12.06523765]
 [-30.88513514  49.14189189 -10.98648649]
 [ 32.89458128  51.27487685 -12.14187192]
 [  8.11728865  35.72277228  14.43259711]
 [-42.34666667 -20.03555556   8.67555556]
 [ 45.70040486 -18.31578947   9.08502024]
 [ -2.24096386 -68.24096386 -16.1686747 ]
 [ -6.99755501 -16.58557457  40.26283619]
 [  7.66409442 -10.19700409  38.43758511]
 [-53.43554007 -22.00958188   5.84059233]
 [ 57.83253741 -23.01050621   5.41865648]
 [ -2.47368421 -71.63157895 -26.84210526]
 [-36.48246546 -79.5685441   -9.16471838]
 [ 37.87057634 -83.19110212  -8.99292214]
 [ -6.18829517 -44.16793893  22.46310433]
 [  7.14114114 -43.09309309  20.51651652]
 [-35.41011841   5.44025834   2.1722282 ]
 [ 38.72131148   5.02769927   0.80158282]
 [-40.1540856   13.86770428 -21.43657588]
 [ 47.93721973  13.4573991  -18.18983558]
 [-11.30091743 -18.86972477   6.60366972]
 [ 12.7038789  -18.76821192   6.72847682]
 [ -2.15686275 -65.1372549  -34.94117647]
 [ -8.9         13.93076923 -13.00769231]
 [ 10.12456747  14.66435986 -12.56055363]
 [ -2.63157895 -56.89473684 -35.31578947]
 [-38.92822695  -6.96        49.64652482]
 [ 41.10085773  -9.54983733  50.80981958]
 [ -2.64285714 -46.85714286 -33.        ]
 [-23.71996124 -60.7994186   57.66375969]
 [ 25.83881135 -60.43674021  60.71949572]
 [-16.75695461 -85.61054173  26.85358712]
 [ 23.9631728  -82.20396601  29.28470255]
 [-48.79576108  11.49132948  17.80154143]
 [ 49.92709078  13.67548249  20.19585418]]
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/997/atlas_aal
* label_names : ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R', 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R', 'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R', 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10']
* networks_list : None
* par_max : 116
* parcel_list : None
* uatlas_select : /Users/derekpisner/nilearn_data/aal_SPM12/aal/atlas/AAL.nii


Runtime info
------------


* duration : 4.945774
* hostname : dpys
* prev_wd : /Users/derekpisner/Applications/PyNets
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/997/Wf_single_subject_127/Wf_single_sub_127_57/Meta_wf_127/functional_connectometry_127/fetch_nodes_and_labels_node


Environment
~~~~~~~~~~~


* ANTSPATH : /Users/derekpisner/bin/ants/bin/
* Apple_PubSub_Socket_Render : /private/tmp/com.apple.launchd.LEz8QPGeOM/Render
* CONDA_DEFAULT_ENV : base
* CONDA_EXE : /usr/local/anaconda3/bin/conda
* CONDA_PREFIX : /usr/local/anaconda3
* CONDA_PROMPT_MODIFIER : (base) 
* CONDA_SHLVL : 1
* CPPFLAGS : -I/usr/local/opt/libxml2/include
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
* OLDPWD : /Users/derekpisner
* OS : Darwin
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner/Applications/PyNets
* SHELL : /bin/bash
* SHLVL : 2
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.0lGeOlHWzb/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.1.1
* TERM_SESSION_ID : FE8A7C24-4E2F-49CF-AFB1-E40646E27050
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0

