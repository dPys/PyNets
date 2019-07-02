Node: meta_wf_0021001 (dmri_connectometry_0021001 (streams2graph_node (utility)
===============================================================================


 Hierarchy : wf_single_sub_0021001_dmri_42.meta_wf_0021001.dmri_connectometry_0021001.streams2graph_node
 Exec ID : streams2graph_node.b0.c0


Original Inputs
---------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_model : csd
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* function_str : def streams2graph(atlas_mni, streams, overlap_thr, dir_path, track_type, target_samples, conn_model, network, node_size,
                  dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels,
                  coords, norm, binary, voxel_size='2mm'):
    '''
    Use tracked streamlines as a basis for estimating a structural connectome.

    Parameters
    ----------
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to streamline array sequence in .trk format.
    overlap_thr : int
        Number of voxels for which a given streamline must intersect with an ROI
        for an edge to be counted.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    voxel_size : str
        Target isotropic voxel resolution of all input Nifti1Image files.

    Returns
    -------
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to streamline array sequence in .trk format.
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
    import networkx as nx
    from itertools import combinations
    from collections import defaultdict
    import time

    # Read Streamlines
    streamlines_mni = nib.streamlines.load(streams)
    streamlines = Streamlines(streamlines_mni.streamlines)

    # Load parcellation
    atlas_data = nib.load(atlas_mni).get_fdata()

    # Instantiate empty networkX graph object & dictionary
    # Create voxel-affine mapping
    lin_T, offset = _mapping_to_voxel(np.eye(4), voxel_size)
    mx = len(np.unique(atlas_data.astype(np.int64)))
    g = nx.Graph(ecount=0, vcount=mx)
    edge_dict = defaultdict(int)
    node_dict = dict(zip(np.unique(atlas_data), np.arange(mx)))

    # Add empty vertices
    for node in range(mx):
        g.add_node(node)

    # Build graph
    start_time = time.time()
    for s in streamlines:
        # Map the streamlines coordinates to voxel coordinates
        points = _to_voxel_coordinates(s, lin_T, offset)

        # get labels for label_volume
        i, j, k = points.T
        lab_arr = atlas_data[i, j, k]
        endlabels = []
        for lab in np.unique(lab_arr):
            if lab > 0:
                if np.sum(lab_arr == lab) >= overlap_thr:
                    endlabels.append(node_dict[lab])

        edges = combinations(endlabels, 2)
        for edge in edges:
            lst = tuple([int(node) for node in edge])
            edge_dict[tuple(sorted(lst))] += 1

        edge_list = [(k[0], k[1], v) for k, v in edge_dict.items()]
        g.add_weighted_edges_from(edge_list)
    print("%s%s%s" % ('Graph construction runtime: ',
    np.round(time.time() - start_time, 1), 's'))

    # Convert to numpy matrix
    conn_matrix_raw = nx.to_numpy_matrix(g)

    # Enforce symmetry
    conn_matrix_symm = np.maximum(conn_matrix_raw, conn_matrix_raw.T)

    # Remove background label
    conn_matrix = conn_matrix_symm[1:, 1:]

    return atlas_mni, streams, conn_matrix, track_type, target_samples, dir_path, conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* overlap_thr : 1
* parc : True
* prune : 1
* roi : None
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_mni_csd_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk
* target_samples : 100000
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz

Execution Inputs
----------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_model : csd
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* function_str : def streams2graph(atlas_mni, streams, overlap_thr, dir_path, track_type, target_samples, conn_model, network, node_size,
                  dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels,
                  coords, norm, binary, voxel_size='2mm'):
    '''
    Use tracked streamlines as a basis for estimating a structural connectome.

    Parameters
    ----------
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to streamline array sequence in .trk format.
    overlap_thr : int
        Number of voxels for which a given streamline must intersect with an ROI
        for an edge to be counted.
    dir_path : str
        Path to directory containing subject derivative data for a given pynets run.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    voxel_size : str
        Target isotropic voxel resolution of all input Nifti1Image files.

    Returns
    -------
    atlas_mni : str
        File path to atlas parcellation Nifti1Image in T1w-warped MNI space.
    streams : str
        File path to streamline array sequence in .trk format.
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    track_type : str
        Tracking algorithm used (e.g. 'local' or 'particle').
    target_samples : int
        Total number of streamline samples specified to generate streams.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    conn_model : str
        Connectivity reconstruction method (e.g. 'csa', 'tensor', 'csd').
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default')
        used to filter nodes in the study of brain subgraphs.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    ID : str
        A subject id or other unique identifier.
    roi : str
        File path to binarized/boolean region-of-interest Nifti1Image file.
    min_span_tree : bool
        Indicates whether local thresholding from the Minimum Spanning Tree
        should be used.
    disp_filt : bool
        Indicates whether local thresholding using a disparity filter and
        'backbone network' should be used.
    parc : bool
        Indicates whether to use parcels instead of coordinates as ROI nodes.
    prune : bool
        Indicates whether to prune final graph of disconnected nodes/isolates.
    atlas : str
        Name of atlas parcellation used.
    uatlas : str
        File path to atlas parcellation Nifti1Image in MNI template space.
    labels : list
        List of string labels corresponding to graph nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
    import networkx as nx
    from itertools import combinations
    from collections import defaultdict
    import time

    # Read Streamlines
    streamlines_mni = nib.streamlines.load(streams)
    streamlines = Streamlines(streamlines_mni.streamlines)

    # Load parcellation
    atlas_data = nib.load(atlas_mni).get_fdata()

    # Instantiate empty networkX graph object & dictionary
    # Create voxel-affine mapping
    lin_T, offset = _mapping_to_voxel(np.eye(4), voxel_size)
    mx = len(np.unique(atlas_data.astype(np.int64)))
    g = nx.Graph(ecount=0, vcount=mx)
    edge_dict = defaultdict(int)
    node_dict = dict(zip(np.unique(atlas_data), np.arange(mx)))

    # Add empty vertices
    for node in range(mx):
        g.add_node(node)

    # Build graph
    start_time = time.time()
    for s in streamlines:
        # Map the streamlines coordinates to voxel coordinates
        points = _to_voxel_coordinates(s, lin_T, offset)

        # get labels for label_volume
        i, j, k = points.T
        lab_arr = atlas_data[i, j, k]
        endlabels = []
        for lab in np.unique(lab_arr):
            if lab > 0:
                if np.sum(lab_arr == lab) >= overlap_thr:
                    endlabels.append(node_dict[lab])

        edges = combinations(endlabels, 2)
        for edge in edges:
            lst = tuple([int(node) for node in edge])
            edge_dict[tuple(sorted(lst))] += 1

        edge_list = [(k[0], k[1], v) for k, v in edge_dict.items()]
        g.add_weighted_edges_from(edge_list)
    print("%s%s%s" % ('Graph construction runtime: ',
    np.round(time.time() - start_time, 1), 's'))

    # Convert to numpy matrix
    conn_matrix_raw = nx.to_numpy_matrix(g)

    # Enforce symmetry
    conn_matrix_symm = np.maximum(conn_matrix_raw, conn_matrix_raw.T)

    # Remove background label
    conn_matrix = conn_matrix_symm[1:, 1:]

    return atlas_mni, streams, conn_matrix, track_type, target_samples, dir_path, conn_model, network, node_size, dens_thresh, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, norm, binary

* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* overlap_thr : 1
* parc : True
* prune : 1
* roi : None
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_mni_csd_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk
* target_samples : 100000
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Execution Outputs
-----------------


* ID : 0021001
* atlas : DesikanKlein2012
* atlas_mni : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/reg_dmri/anat_reg/DesikanKlein2012_t1w_mni.nii.gz
* binary : False
* conn_matrix : [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 1. 1.]
 [0. 0. 0. ... 1. 0. 1.]
 [0. 0. 0. ... 1. 1. 0.]]
* conn_model : csd
* coords : [(41.460367190926746, -7.717776803057831, 44.17087536813084), (22.609863496257162, -4.5623073535887215, -31.956406869220608), (46.24613187336348, -65.41561532968342, 30.409842894548916), (-3.643428571428572, -66.68971428571429, -23.60742857142857), (-4.467642526964568, -50.63995891114536, -13.200051361068311), (-50.25136790908666, -10.39928915493617, -6.6312491231352055), (25.510078125000007, -60.263671875, 53.30015625), (-29.850480700243935, -89.22470942746449, 1.5027263595924865), (32.80915226949037, -86.42231717134189, 2.0475571242801465), (-57.46055697326084, -27.808776212138028, -12.658715189171197), (5.862181818181824, 37.53200000000001, 2.9570909090909083), (45.29406346814034, -21.888721426777195, 43.83195177091183), (-6.173169428113312, -25.681631925886336, 57.79975057901299), (10.774665617623924, -15.936742722265933, -9.790401258851297), (46.06022408963585, -17.974789915966383, 8.11204481792717), (13.69881910335458, -66.42386874281534, -5.0391890479673975), (49.02416243654821, -28.17928934010152, -27.296785109983077), (-13.293843177421195, -67.50893365198588, -5.878912029133943), (26.308073973838518, -20.534506089309872, -12.81235904375282), (5.462565104166671, 21.543294270833343, 28.04296875), (-48.32543192812716, -28.426952315134756, -28.301174844505873), (7.903225806451616, -44.983548387096775, 19.30645161290323)]
* dens_thresh : True
* dir_path : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012
* disp_filt : False
* labels : [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
* min_span_tree : False
* network : None
* node_size : None
* norm : 0
* parc : True
* prune : 1
* roi : None
* streams : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/DesikanKlein2012/streamlines_mni_csd_100000_Nonemm_curv[60_30_10]_step[0.2_0.3_0.4_0.5]_warped.trk
* target_samples : 100000
* track_type : particle
* uatlas : /Users/derekpisner/Applications/PyNets/pynets/atlases/DesikanKlein2012.nii.gz


Runtime info
------------


* duration : 0.637298
* hostname : dpys
* prev_wd : /Users/derekpisner
* working_dir : /Users/derekpisner/Applications/PyNets/tests/examples/002/dmri/wf_single_subject_dmri_0021001/wf_single_sub_0021001_dmri_42/meta_wf_0021001/dmri_connectometry_0021001/_network_Default/_conn_model_csd/streams2graph_node


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
* OS : Darwin
* PATH : /Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/Users/derekpisner/anaconda3/bin:/Users/derekpisner/bin/ants/bin/:/usr/local/opt/libxml2/bin:/Applications/freesurfer/bin:/Applications/freesurfer/fsfast/bin:/Applications/freesurfer/tktools:/usr/local/fsl/bin:/Applications/freesurfer/mni/bin:/usr/local/fsl/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Users/derekpisner/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/opt/X11/bin:/Users/derekpisner/abin:/Users/derekpisner/abin
* PERL5LIB : /Applications/freesurfer/mni/lib/../Library/Perl/Updates/5.12.3
* PWD : /Users/derekpisner
* SHELL : /bin/bash
* SHLVL : 3
* SSH_AUTH_SOCK : /private/tmp/com.apple.launchd.qmAkE8F40f/Listeners
* SUBJECTS_DIR : /Applications/freesurfer/subjects
* TERM : xterm-256color
* TERM_PROGRAM : Apple_Terminal
* TERM_PROGRAM_VERSION : 421.1.1
* TERM_SESSION_ID : AFAF5DB1-79BD-4BC9-B7BB-C754B1B9AAB6
* TMPDIR : /var/folders/r1/p8kclf5j3v74m4l5l4__jty00000gn/T/
* USER : derekpisner
* XPC_FLAGS : 0x0
* XPC_SERVICE_NAME : 0
* _ : /usr/local/anaconda3/bin/pynets_run.py
* _CE_CONDA : 
* _CE_M : 
* __CF_USER_TEXT_ENCODING : 0x1F5:0x0:0x0

