#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner (dPys)
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def get_conn_matrix(time_series, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree,
                    disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary,
                    hpass):
    """
    Computes a functional connectivity matrix based on a node-extracted time-series array.
    Includes a library of routines across Nilearn, scikit-learn, and skggm packages, among others.

    Parameters
    ----------
    time_series : array
        2D m x n array consisting of the time-series signal for each ROI node where m = number of scans and
        n = number of ROI's.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
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
        List of string labels corresponding to ROI nodes.
    coords : list
        List of (x, y, z) tuples corresponding to a coordinate atlas used or
        which represent the center-of-mass of each parcellation node.
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.

    Returns
    -------
    conn_matrix : array
        Adjacency matrix stored as an m x n array of nodes and edges.
    conn_model : str
       Connectivity estimation model (e.g. corr for correlation, cov for covariance, sps for precision covariance,
       partcorr for partial correlation). sps type is used by default.
    dir_path : str
        Path to directory containing subject derivative data for given run.
    node_size : int
        Spherical centroid node size in the case that coordinate-based centroids
        are used as ROI's for tracking.
    smooth : int
        Smoothing width (mm fwhm) to apply to time-series when extracting signal from ROI's.
    dens_thresh : bool
        Indicates whether a target graph density is to be used as the basis for
        thresholding.
    network : str
        Resting-state network based on Yeo-7 and Yeo-17 naming (e.g. 'Default') used to filter nodes in the study of
        brain subgraphs.
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
    c_boot : int
        Number of bootstraps if user specified circular-block bootstrapped resampling of the node-extracted time-series.
    norm : int
        Indicates method of normalizing resulting graph.
    binary : bool
        Indicates whether to binarize resulting graph edges to form an
        unweighted graph.
    hpass : bool
        High-pass filter values (Hz) to apply to node-extracted time-series.
    """
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import GraphicalLassoCV

    conn_matrix = None
    if conn_model == 'corr' or conn_model == 'cor' or conn_model == 'correlation':
        # credit: nilearn
        print('\nComputing correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'partcorr' or conn_model == 'parcorr' or conn_model == 'partialcorrelation':
        # credit: nilearn
        print('\nComputing partial correlation matrix...\n')
        conn_measure = ConnectivityMeasure(kind='partial correlation')
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif conn_model == 'cov' or conn_model == 'covariance' or conn_model == 'covar' or conn_model == 'sps' or conn_model == 'sparse' or conn_model == 'precision':
        # Fit estimator to matrix to get sparse matrix
        estimator_shrunk = None
        estimator = GraphicalLassoCV(cv=5)
        try:
            print('\nComputing covariance...\n')
            estimator.fit(time_series)
        except:
            print('Unstable Lasso estimation--Attempting to re-run by first applying shrinkage...')
            try:
                from sklearn.covariance import GraphicalLasso, empirical_covariance, shrunk_covariance
                emp_cov = empirical_covariance(time_series)
                for i in np.arange(0.8, 0.99, 0.01):
                    shrunk_cov = shrunk_covariance(emp_cov, shrinkage=i)
                    alphaRange = 10.0 ** np.arange(-8, 0)
                    for alpha in alphaRange:
                        try:
                            estimator_shrunk = GraphicalLasso(alpha)
                            estimator_shrunk.fit(shrunk_cov)
                            print("Retrying covariance matrix estimate with alpha=%s" % alpha)
                            if estimator_shrunk is None:
                                pass
                            else:
                                break
                        except:
                            print("Covariance estimation failed with shrinkage at alpha=%s" % alpha)
                            continue
            except ValueError:
                print('Unstable Lasso estimation! Shrinkage failed. A different connectivity model may be needed.')
        if estimator is None and estimator_shrunk is None:
            raise RuntimeError('\nERROR: Covariance estimation failed.')
        if conn_model == 'sps' or conn_model == 'sparse' or conn_model == 'precision':
            if estimator_shrunk is None:
                print('\nFetching precision matrix from covariance estimator...\n')
                conn_matrix = -estimator.precision_
            else:
                print('\nFetching shrunk precision matrix from covariance estimator...\n')
                conn_matrix = -estimator_shrunk.precision_
        elif conn_model == 'cov' or conn_model == 'covariance' or conn_model == 'covar':
            if estimator_shrunk is None:
                print('\nFetching covariance matrix from covariance estimator...\n')
                conn_matrix = estimator.covariance_
            else:
                conn_matrix = estimator_shrunk.covariance_
    elif conn_model == 'QuicGraphicalLasso':

        try:
            from inverse_covariance import QuicGraphicalLasso
        except ImportError:
            print('Cannot run QuicGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLasso
        # credit: skggm
        model = QuicGraphicalLasso(
            init_method='cov',
            lam=0.5,
            mode='default',
            verbose=1)
        print('\nCalculating QuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphicalLassoCV':
        try:
            from inverse_covariance import QuicGraphicalLassoCV
        except ImportError:
            print('Cannot run QuicGraphLassoCV. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoCV
        # credit: skggm
        model = QuicGraphicalLassoCV(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoCV precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'QuicGraphicalLassoEBIC':
        try:
            from inverse_covariance import QuicGraphicalLassoEBIC
        except ImportError:
            print('Cannot run QuicGraphLassoEBIC. Skggm not installed!')

        # Compute the sparse inverse covariance via QuicGraphLassoEBIC
        # credit: skggm
        model = QuicGraphicalLassoEBIC(
            init_method='cov',
            verbose=1)
        print('\nCalculating QuicGraphLassoEBIC precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.precision_
    elif conn_model == 'AdaptiveQuicGraphicalLasso':
        try:
            from inverse_covariance import AdaptiveQuicGraphicalLasso, QuicGraphicalLassoEBIC
        except ImportError:
            print('Cannot run AdaptiveGraphLasso. Skggm not installed!')

        # Compute the sparse inverse covariance via
        # AdaptiveGraphLasso + QuicGraphLassoEBIC + method='binary'
        # credit: skggm
        model = AdaptiveQuicGraphicalLasso(
            estimator=QuicGraphicalLassoEBIC(
                init_method='cov',
            ),
            method='binary',
        )
        print('\nCalculating AdaptiveQuicGraphLasso precision matrix using skggm...\n')
        model.fit(time_series)
        conn_matrix = -model.estimator_.precision_
    else:
        raise ValueError('\nERROR! No connectivity model specified at runtime. Select a valid estimator using the '
                         '-mod flag.')

    # Enforce symmetry
    conn_matrix = np.maximum(conn_matrix, conn_matrix.T)

    if conn_matrix.shape < (2, 2):
        raise RuntimeError('\nERROR! Matrix estimation selection yielded an empty or 1-dimensional graph. '
                           'Check time-series for errors or try using a different atlas')

    coords = np.array(coords)
    labels = np.array(labels)

    del time_series

    return conn_matrix, conn_model, dir_path, node_size, smooth, dens_thresh, network, ID, roi, min_span_tree, disp_filt, parc, prune, atlas, uatlas, labels, coords, c_boot, norm, binary, hpass


def timeseries_bootstrap(tseries, block_size):
    """
    Generates a bootstrap sample derived from the input time-series.
    Utilizes Circular-block-bootstrap method described in [1]_.

    Parameters
    ----------
    tseries : array_like
        A matrix of shapes (`M`, `N`) with `M` timepoints and `N` variables
    block_size : integer
        Size of the bootstrapped blocks

    Returns
    -------
    bseries : array_like
        Bootstrap sample of the input timeseries

    References
    ----------
    .. [1] P. Bellec; G. Marrelec; H. Benali, A bootstrap test to investigate
       changes in brain connectivity for functional MRI. Statistica Sinica,
       special issue on Statistical Challenges and Advances in Brain Science,
       2008, 18: 1253-1268.
    """
    np.random.seed(int(42))

    # calculate number of blocks
    k = int(np.ceil(float(tseries.shape[0]) / block_size))

    # generate random indices of blocks
    r_ind = np.floor(np.random.rand(1, k) * tseries.shape[0])
    blocks = np.dot(np.arange(0, block_size)[:, np.newaxis], np.ones([1, k]))

    block_offsets = np.dot(np.ones([block_size, 1]), r_ind)
    block_mask = (blocks + block_offsets).flatten('F')[:tseries.shape[0]]
    block_mask = np.mod(block_mask, tseries.shape[0])

    return tseries[block_mask.astype('uint8'), :], block_mask.astype('uint8')


def fill_confound_nans(confounds, dir_path):
    """Fill the NaN values of a confounds dataframe with mean values"""
    import uuid
    import os
    from time import strftime
    run_uuid = '%s_%s' % (strftime('%Y%m%d_%H%M%S'), uuid.uuid4())
    print('Warning: NaN\'s detected in confound regressor file. Filling these with mean values, but the '
          'regressor file should be checked manually.')
    confounds_nonan = confounds.apply(lambda x: x.fillna(x.mean()), axis=0)
    os.makedirs("%s%s" % (dir_path, '/confounds_tmp'), exist_ok=True)
    conf_corr = "%s%s%s%s" % (dir_path, '/confounds_tmp/confounds_mean_corrected_', run_uuid, '.tsv')
    confounds_nonan.to_csv(conf_corr, sep='\t')
    return conf_corr


class TimeseriesExtraction(object):
    """
    Class for implementing various time-series extracting routines.
    """
    def __init__(self, net_parcels_nii_path, node_size, conf, func_file, coords, roi, dir_path, ID, network, smooth,
                 atlas, uatlas, labels, c_boot, block_size, hpass, mask):
        self.net_parcels_nii_path = net_parcels_nii_path
        self.node_size = node_size
        self.conf = conf
        self.func_file = func_file
        self.coords = coords
        self.roi = roi
        self.dir_path = dir_path
        self.ID = ID
        self.network = network
        self.smooth = smooth
        self.atlas = atlas
        self.uatlas = uatlas
        self.labels = labels
        self.c_boot = c_boot
        self.block_size = block_size
        self.mask = mask
        self.hpass = hpass
        self.ts_within_nodes = None
        self._mask_img = None
        self._mask_path = None
        self._func_img = None
        self._t_r = None
        self._detrending = True
        self._net_parcels_nii_temp_path = None
        self._net_parcels_map_nifti = None
        self._spheres_masker = None
        self._parcel_masker = None

    def prepare_inputs(self):
        """Helper function to creating temporary nii's and prepare inputs from time-series extraction"""
        import os.path as op
        import nibabel as nib
        from nilearn.image import math_img
        if not op.isfile(self.func_file):
            raise ValueError('\nERROR: Functional data input not found! Check that the file(s) specified with the -i '
                             'flag exist(s)')

        if self.conf:
            if not op.isfile(self.conf):
                raise ValueError('\nERROR: Confound regressor file not found! Check that the file(s) specified with '
                                 'the -conf flag exist(s)')

        self._func_img = nib.load(self.func_file)
        self._func_img.set_data_dtype(np.float32)
        hdr = self._func_img.header

        if self.hpass:
            if len(hdr.get_zooms()) == 4:
                self._t_r = float(hdr.get_zooms()[-1])
            else:
                self._t_r = None
        else:
            self._t_r = None

        if self.hpass is not None:
            if float(self.hpass) > 0:
                self.hpass = float(self.hpass)
                self._detrending = False
            else:
                self.hpass = None
                self._detrending = True
        else:
            self.hpass = None
            self._detrending = True

        if self.mask is not None:
            # Ensure mask is binary
            self._mask_img = math_img('img > 0', img=nib.load(self.mask))
            self._mask_img.set_data_dtype(np.uint8)
        else:
            self._mask_img = None

        if self.smooth:
            if float(self.smooth) > 0:
                print("%s%s%s" % ('Smoothing FWHM: ', self.smooth, ' mm\n'))

        if self.hpass:
            print("%s%s%s" % ('Applying high-pass filter: ', self.hpass, ' Hz\n'))

        return

    def extract_ts_coords(self):
        """
        API for employing Nilearn's NiftiSpheresMasker to extract fMRI time-series data from spherical ROI's based on a
        given list of seed coordinates. The resulting time-series can then optionally be resampled using circular-block
        bootrapping. The final 2D m x n array is ultimately saved to file in .npy format.
        """
        from nilearn import input_data
        from pynets.fmri.estimation import fill_confound_nans

        print("%s%s%s" % ('Using node radius: ', self.node_size, ' mm'))
        self._spheres_masker = input_data.NiftiSpheresMasker(seeds=self.coords, radius=float(self.node_size),
                                                             allow_overlap=True, standardize=True,
                                                             smoothing_fwhm=float(self.smooth),
                                                             high_pass=self.hpass, detrend=self._detrending,
                                                             t_r=self._t_r, verbose=2, dtype='auto',
                                                             mask_img=self._mask_img)
        if self.conf is not None:
            import pandas as pd
            confounds = pd.read_csv(self.conf, sep='\t')
            if confounds.isnull().values.any():
                conf_corr = fill_confound_nans(confounds, self.dir_path)
                self.ts_within_nodes = self._spheres_masker.fit_transform(self._func_img, confounds=conf_corr)
            else:
                self.ts_within_nodes = self._spheres_masker.fit_transform(self._func_img, confounds=self.conf)
        else:
            self.ts_within_nodes = self._spheres_masker.fit_transform(self._func_img)

        self._func_img.uncache()

        if self.ts_within_nodes is None:
            raise RuntimeError('\nERROR: Time-series extraction failed!')
        else:
            print("%s%s%d%s" % ('\nTime series has {0} samples'.format(self.ts_within_nodes.shape[0]),
                                ' mean extracted from ', len(self.coords), ' coordinate ROI\'s'))

        return

    def extract_ts_parc(self):
        """
        API for employing Nilearn's NiftiLabelsMasker to extract fMRI time-series data from spherical ROI's based on a
        given 3D atlas image of integer-based voxel intensities. The resulting time-series can then optionally be
        resampled using circular-block bootrapping. The final 2D m x n array is ultimately saved to file in .npy format.
        """
        import nibabel as nib
        from nilearn import input_data
        from pynets.fmri.estimation import fill_confound_nans

        self._net_parcels_map_nifti = nib.load(self.net_parcels_nii_path)
        self._net_parcels_map_nifti.set_data_dtype(np.uint8)
        self._parcel_masker = input_data.NiftiLabelsMasker(labels_img=self._net_parcels_map_nifti, background_label=0,
                                                           standardize=True, smoothing_fwhm=float(self.smooth),
                                                           high_pass=self.hpass, detrend=self._detrending,
                                                           t_r=self._t_r, verbose=2, resampling_target='data',
                                                           dtype='auto', mask_img=self._mask_img)
        if self.conf is not None:
            import pandas as pd
            confounds = pd.read_csv(self.conf, sep='\t')
            if confounds.isnull().values.any():
                conf_corr = fill_confound_nans(confounds, self.dir_path)
                self.ts_within_nodes = self._parcel_masker.fit_transform(self._func_img, confounds=conf_corr)
            else:
                self.ts_within_nodes = self._parcel_masker.fit_transform(self._func_img, confounds=self.conf)
        else:
            self.ts_within_nodes = self._parcel_masker.fit_transform(self._func_img)

        self._func_img.uncache()

        if self.ts_within_nodes is None:
            raise RuntimeError('\nERROR: Time-series extraction failed!')
        else:
            self.node_size = 'parc'

        return

    def bootstrap_timeseries(self):
        """Perform circular-block bootstrapping of the extracted time-series"""
        print("%s%s%s" % ('Performing circular block bootstrapping with ', self.c_boot, ' iterations...'))
        self.ts_within_nodes = timeseries_bootstrap(self.ts_within_nodes, self.block_size)[0]
        return

    def save_and_cleanup(self):
        """Save the extracted time-series and clean cache"""
        from pynets.core import utils

        # Save time series as file
        utils.save_ts_to_file(self.roi, self.network, self.ID, self.dir_path, self.ts_within_nodes, self.c_boot,
                              self.smooth, self.hpass, self.node_size)

        if self._mask_path is not None:
            self._mask_img.uncache()

        if self._spheres_masker is not None:
            del self._spheres_masker

        if self._parcel_masker is not None:
            del self._parcel_masker
            self._net_parcels_map_nifti.uncache()
        return
