# first line: 348
def clean(signals, sessions=None, detrend=True, standardize=True,
          confounds=None, low_pass=None, high_pass=None, t_r=2.5,
          ensure_finite=False):
    """Improve SNR on masked fMRI signals.

    This function can do several things on the input signals, in
    the following order:

    - detrend
    - standardize
    - remove confounds
    - low- and high-pass filter

    Low-pass filtering improves specificity.

    High-pass filtering should be kept small, to keep some
    sensitivity.

    Filtering is only meaningful on evenly-sampled signals.

    Parameters
    ----------
    signals: numpy.ndarray
        Timeseries. Must have shape (instant number, features number).
        This array is not modified.

    sessions : numpy array, optional
        Add a session level to the cleaning process. Each session will be
        cleaned independently. Must be a 1D array of n_samples elements.

    confounds: numpy.ndarray, str or list of
        Confounds timeseries. Shape must be
        (instant number, confound number), or just (instant number,)
        The number of time instants in signals and confounds must be
        identical (i.e. signals.shape[0] == confounds.shape[0]).
        If a string is provided, it is assumed to be the name of a csv file
        containing signals as columns, with an optional one-line header.
        If a list is provided, all confounds are removed from the input
        signal, as if all were in the same array.

    t_r: float
        Repetition time, in second (sampling period).

    low_pass, high_pass: float
        Respectively low and high cutoff frequencies, in Hertz.

    detrend: bool
        If detrending should be applied on timeseries (before
        confound removal)

    standardize: bool
        If True, returned signals are set to unit variance.

    ensure_finite: bool
        If True, the non-finite values (NANs and infs) found in the data
        will be replaced by zeros.

    Returns
    -------
    cleaned_signals: numpy.ndarray
        Input signals, cleaned. Same shape as `signals`.

    Notes
    -----
    Confounds removal is based on a projection on the orthogonal
    of the signal space. See `Friston, K. J., A. P. Holmes,
    K. J. Worsley, J.-P. Poline, C. D. Frith, et R. S. J. Frackowiak.
    "Statistical Parametric Maps in Functional Imaging: A General
    Linear Approach". Human Brain Mapping 2, no 4 (1994): 189-210.
    <http://dx.doi.org/10.1002/hbm.460020402>`_

    See Also
    --------
        nilearn.image.clean_img
    """

    if isinstance(low_pass, bool):
        raise TypeError("low pass must be float or None but you provided "
                        "low_pass='{0}'".format(low_pass))
    if isinstance(high_pass, bool):
        raise TypeError("high pass must be float or None but you provided "
                        "high_pass='{0}'".format(high_pass))

    if not isinstance(confounds,
                      (list, tuple, _basestring, np.ndarray, type(None))):
        raise TypeError("confounds keyword has an unhandled type: %s"
                        % confounds.__class__)

    if not isinstance(ensure_finite, bool):
        raise ValueError("'ensure_finite' must be boolean type True or False "
                         "but you provided ensure_finite={0}".format(ensure_finite))

    if not isinstance(signals, np.ndarray):
        signals = as_ndarray(signals)

    if ensure_finite:
        signals[np.logical_not(np.isfinite(signals))] = 0

    # Read confounds
    if confounds is not None:
        if not isinstance(confounds, (list, tuple)):
            confounds = (confounds, )

        all_confounds = []
        for confound in confounds:
            if isinstance(confound, _basestring):
                filename = confound
                confound = csv_to_array(filename)
                if np.isnan(confound.flat[0]):
                    # There may be a header
                    if NP_VERSION >= [1, 4, 0]:
                        confound = csv_to_array(filename, skip_header=1)
                    else:
                        confound = csv_to_array(filename, skiprows=1)
                if confound.shape[0] != signals.shape[0]:
                    raise ValueError("Confound signal has an incorrect length")

            elif isinstance(confound, np.ndarray):
                if confound.ndim == 1:
                    confound = np.atleast_2d(confound).T
                elif confound.ndim != 2:
                    raise ValueError("confound array has an incorrect number "
                                     "of dimensions: %d" % confound.ndim)

                if confound.shape[0] != signals.shape[0]:
                    raise ValueError("Confound signal has an incorrect length")
            else:
                raise TypeError("confound has an unhandled type: %s"
                                % confound.__class__)
            all_confounds.append(confound)

        # Restrict the signal to the orthogonal of the confounds
        confounds = np.hstack(all_confounds)
        del all_confounds

    if sessions is not None:
        if not len(sessions) == len(signals):
            raise ValueError(('The length of the session vector (%i) '
                              'does not match the length of the signals (%i)')
                              % (len(sessions), len(signals)))
        for s in np.unique(sessions):
            session_confounds = None
            if confounds is not None:
                session_confounds = confounds[sessions == s]
            signals[sessions == s, :] = \
                clean(signals[sessions == s],
                      detrend=detrend, standardize=standardize,
                      confounds=session_confounds, low_pass=low_pass,
                      high_pass=high_pass, t_r=2.5)

    # detrend
    signals = _ensure_float(signals)
    signals = _standardize(signals, normalize=False, detrend=detrend)

    # Remove confounds
    if confounds is not None:
        confounds = _ensure_float(confounds)
        confounds = _standardize(confounds, normalize=standardize,
                                 detrend=detrend)
        if not standardize:
            # Improve numerical stability by controlling the range of
            # confounds. We don't rely on _standardize as it removes any
            # constant contribution to confounds.
            confound_max = np.max(np.abs(confounds), axis=0)
            confound_max[confound_max == 0] = 1
            confounds /= confound_max

        if (LooseVersion(scipy.__version__) > LooseVersion('0.9.0')):
            # Pivoting in qr decomposition was added in scipy 0.10
            Q, R, _ = linalg.qr(confounds, mode='economic', pivoting=True)
            Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float).eps * 100.]
            signals -= Q.dot(Q.T).dot(signals)
        else:
            Q, R = linalg.qr(confounds, mode='economic')
            non_null_diag = np.abs(np.diag(R)) > np.finfo(np.float).eps * 100.
            if np.all(non_null_diag):
                signals -= Q.dot(Q.T).dot(signals)
            elif np.any(non_null_diag):
                R = R[:, non_null_diag]
                confounds = confounds[:, non_null_diag]
                inv = scipy.linalg.inv(np.dot(R.T, R))
                signals -= confounds.dot(inv).dot(confounds.T).dot(signals)

    if low_pass is not None or high_pass is not None:
        if t_r is None:
            raise ValueError("Repetition time (t_r) must be specified for "
                             "filtering")

        signals = butterworth(signals, sampling_rate=1. / t_r,
                              low_pass=low_pass, high_pass=high_pass)

    if standardize:
        signals = _standardize(signals, normalize=True, detrend=False)
        signals *= np.sqrt(signals.shape[0])  # for unit variance

    return signals
