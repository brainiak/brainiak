#  Copyright 2017 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Intersubject correlation (ISC) analysis

Functions for computing intersubject correlation (ISC) and related
analyses (e.g., intersubject funtional correlations; ISFC), as well
as statistical tests designed specifically for ISC analyses.

The implementation is based on the work in [Hasson2004]_, [Kauppi2014]_,
[Simony2016]_, and [Chen2016]_.

.. [Chen2016] "Untangling the relatedness among correlations, part I:
   nonparametric approaches to inter-subject correlation analysis at the
   group level.", G. Chen, Y. W. Shin, P. A. Taylor, D. R. Glen, R. C.
   Reynolds, R. B. Israel, R. W. Cox, 2016, NeuroImage, 142, 248-259.
   https://doi.org/10.1016/j.neuroimage.2016.05.023

.. [Hasson2004] "Intersubject synchronization of cortical activity
   during natural vision.", U. Hasson, Y. Nir, I. Levy, G. Fuhrmann,
   R. Malach, 2004, Science, 303, 1634-1640.
   https://doi.org/10.1126/science.1089506

.. [Kauppi2014] "A versatile software package for inter-subject
   correlation based analyses of fMRI.", J. P. Kauppi, J. Pajula,
   J. Tohka, 2014, Frontiers in Neuroinformatics, 8, 2.
   https://doi.org/10.3389/fninf.2014.00002

.. [Simony2016] "Dynamic reconfiguration of the default mode network
   during narrative comprehension.", E. Simony, C. J. Honey, J. Chen, O.
   Lositsky, Y. Yeshurun, A. Wiesel, U. Hasson, 2016, Nature Communications,
   7, 12141. https://doi.org/10.1038/ncomms12141
"""

# Authors: Sam Nastase, Christopher Baldassano, Qihong Lu,
#          Mai Nguyen, and Mor Regev
# Princeton University, 2018

import numpy as np
import logging
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from scipy.fftpack import fft, ifft
import itertools as it
from brainiak.fcma.util import compute_correlation
from brainiak.utils.utils import compute_p_from_null_distribution

logger = logging.getLogger(__name__)

__all__ = [
    "isc",
    "isfc",
    "check_timeseries_input",
    "check_isc_input",
    "compute_summary_statistic",
    "bootstrap_isc",
    "check_group_assignment",
    "get_group_parameters",
    "permute_one_sample_iscs",
    "permute_two_sample_iscs",
    "permutation_isc",
    "timeshift_isc",
    "phaseshift_isc"
]


MAX_RANDOM_SEED = 2**32 - 1


def isc(data, pairwise=False, summary_statistic=None):
    """Intersubject correlation

    For each voxel or ROI, compute the Pearson correlation between each
    subject's response time series and other subjects' response time series.
    If pairwise is False (default), use the leave-one-out approach, where
    correlation is computed between each subject and the average of the other
    subjects. If pairwise is True, compute correlations between all pairs of
    subjects. If summary_statistic is None, return N ISC values for N subjects
    (leave-one-out) or N(N-1)/2 ISC values for each pair of N subjects,
    corresponding to the upper triangle of the pairwise correlation matrix
    (see scipy.spatial.distance.squareform). Alternatively, use either
    'mean' or 'median' to compute summary statistic of ISCs (Fisher Z will
    be applied if using mean). Input data should be a n_TRs by n_voxels by
    n_subjects array (e.g., brainiak.image.MaskedMultiSubjectData) or a list
    where each item is a n_TRs by n_voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. If only two
    subjects are supplied, simply compute Pearson correlation (precludes
    averaging in leave-one-out approach, and does not apply summary statistic).
    Output is an ndarray where the first dimension is the number of subjects
    or pairs and the second dimension is the number of voxels (or ROIs).

    The implementation is based on the work in [Hasson2004]_.

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISC

    pairwise : bool, default:False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : None or str, default:None
        Return all ISCs or collapse using 'mean' or 'median'

    Returns
    -------
    iscs : subjects or pairs by voxels ndarray
        ISC for each subject or pair (or summary statistic) per voxel

    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = check_timeseries_input(data)

    # No summary statistic if only two subjects
    if n_subjects == 2:
        logger.info("Only two subjects! Simply computing Pearson correlation.")
        summary_statistic = None

    # Loop over each voxel or ROI
    voxel_iscs = []
    for v in np.arange(n_voxels):
        voxel_data = data[:, v, :].T
        if n_subjects == 2:
            iscs = pearsonr(voxel_data[0, :], voxel_data[1, :])[0]
        elif pairwise:
            iscs = squareform(np.corrcoef(voxel_data), checks=False)
        elif not pairwise:
            iscs = np.array([pearsonr(subject,
                                      np.mean(np.delete(voxel_data,
                                                        s, axis=0),
                                              axis=0))[0]
                             for s, subject in enumerate(voxel_data)])
        voxel_iscs.append(iscs)
    iscs = np.column_stack(voxel_iscs)

    # Summarize results (if requested)
    if summary_statistic:
        iscs = compute_summary_statistic(iscs,
                                         summary_statistic=summary_statistic,
                                         axis=0)[np.newaxis, :]

    return iscs


def isfc(data, pairwise=False, summary_statistic=None):

    """Intersubject functional correlation (ISFC)

    For each voxel or ROI, compute the Pearson correlation between each
    subject's response time series and other subjects' response time series
    for all voxels or ROIs. If pairwise is False (default), use the
    leave-one-out approach, where correlation is computed between each
    subject and the average of the other subjects. If pairwise is True,
    compute correlations between all pairs of subjects. If summary_statistic
    is None, return N ISFC values for N subjects (leave-one-out) or N(N-1)/2
    ISFC values for each pair of N subjects, corresponding to the upper
    triangle of the correlation matrix (see scipy.spatial.distance.squareform).
    Alternatively, use either 'mean' or 'median' to compute summary statistic
    of ISFCs (Fisher Z is applied if using mean). Input should be n_TRs by
    n_voxels by n_subjects array (e.g., brainiak.image.MaskedMultiSubjectData)
    or a list where each item is a n_TRs by n_voxels ndarray per subject.
    Multiple input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. If only two
    subjects are supplied, simply compute ISFC between these two subjects
    (precludes averaging in leave-one-out approach, and does not apply summary
    statistic). Output is n_voxels by n_voxels array if summary_statistic is
    supplied; otherwise output is n_voxels by n_voxels by n_subjects (or
    n_pairs) array.

    The implementation is based on the work in [Simony2016]_.

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : None or str, default:None
        Return all ISFCs or collapse using 'mean' or 'median'

    Returns
    -------
    isfcs : subjects or pairs by voxels ndarray
        ISFC for each subject or pair (or summary statistic) per voxel

    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = check_timeseries_input(data)

    # Handle just two subjects properly
    if n_subjects == 2:
        isfcs = compute_correlation(np.ascontiguousarray(data[..., 0].T),
                                    np.ascontiguousarray(data[..., 1].T))
        isfcs = (isfcs + isfcs.T) / 2
        assert isfcs.shape == (n_voxels, n_voxels)
        summary_statistic = None
        logger.info("Only two subjects! Computing ISFC between them.")

    # Compute all pairwise ISFCs
    elif pairwise:
        isfcs = []
        for pair in it.combinations(np.arange(n_subjects), 2):
            isfc_pair = compute_correlation(np.ascontiguousarray(
                                                data[..., pair[0]].T),
                                            np.ascontiguousarray(
                                                data[..., pair[1]].T))
            isfc_pair = (isfc_pair + isfc_pair.T) / 2
            isfcs.append(isfc_pair)
        isfcs = np.dstack(isfcs)
        assert isfcs.shape == (n_voxels, n_voxels,
                               n_subjects * (n_subjects - 1) / 2)

    # Compute ISFCs using leave-one-out approach
    elif not pairwise:

        # Roll subject axis for loop
        data = np.rollaxis(data, 2, 0)

        # Compute leave-one-out ISFCs
        isfcs = [compute_correlation(np.ascontiguousarray(subject.T),
                                     np.ascontiguousarray(np.mean(
                                         np.delete(data, s, axis=0),
                                         axis=0).T))
                 for s, subject in enumerate(data)]

        # Transpose and average ISFC matrices for both directions
        isfcs = np.dstack([(isfc_matrix + isfc_matrix.T) / 2
                           for isfc_matrix in isfcs])
        assert isfcs.shape == (n_voxels, n_voxels, n_subjects)

    # Summarize results (if requested)
    if summary_statistic:
        isfcs = compute_summary_statistic(isfcs,
                                          summary_statistic=summary_statistic,
                                          axis=2)

    return isfcs


def check_timeseries_input(data):

    """Checks response time series input data for ISC analysis

    Input data should be a n_TRs by n_voxels by n_subjects ndarray
    (e.g., brainiak.image.MaskedMultiSubjectData) or a list where each
    item is a n_TRs by n_voxels ndarray for a given subject. Multiple
    input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. This
    function is only intended to be used internally by other
    functions in this module (e.g., isc, isfc).

    Parameters
    ----------
    data : ndarray or list
        Time series data

    Returns
    -------
    iscs : ndarray
        Array of ISC values

    n_TRs : int
        Number of time points (TRs)

    n_voxels : int
        Number of voxels (or ROIs)

    n_subjects : int
        Number of subjects

    """

    # Convert list input to 3d and check shapes
    if type(data) == list:
        data_shape = data[0].shape
        for i, d in enumerate(data):
            if d.shape != data_shape:
                raise ValueError("All ndarrays in input list "
                                 "must be the same shape!")
            if d.ndim == 1:
                data[i] = d[:, np.newaxis]
        data = np.dstack(data)

    # Convert input ndarray to 3d and check shape
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data[:, np.newaxis, :]
        elif data.ndim == 3:
            pass
        else:
            raise ValueError("Input ndarray should have 2 "
                             "or 3 dimensions (got {0})!".format(data.ndim))

    # Infer subjects, TRs, voxels and log for user to check
    n_TRs, n_voxels, n_subjects = data.shape
    logger.info("Assuming {0} subjects with {1} time points "
                "and {2} voxel(s) or ROI(s) for ISC analysis.".format(
                    n_subjects, n_TRs, n_voxels))

    return data, n_TRs, n_voxels, n_subjects


def check_isc_input(iscs, pairwise=False):

    """Checks ISC inputs for statistical tests

    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array or a 1D
    array (or list) of ISC values for a single voxel or ROI. This
    function is only intended to be used internally by other
    functions in this module (e.g., bootstrap_isc, permutation_isc).

    Parameters
    ----------
    iscs : ndarray or list
        ISC values

    Returns
    -------
    iscs : ndarray
        Array of ISC values

    n_subjects : int
        Number of subjects

    n_voxels : int
        Number of voxels (or ROIs)
    """

    # Standardize structure of input data
    if type(iscs) == list:
        iscs = np.array(iscs)[:, np.newaxis]

    elif isinstance(iscs, np.ndarray):
        if iscs.ndim == 1:
            iscs = iscs[:, np.newaxis]

    # Check if incoming pairwise matrix is vectorized triangle
    if pairwise:
        try:
            test_square = squareform(iscs[:, 0])
            n_subjects = test_square.shape[0]
        except ValueError:
            raise ValueError("For pairwise input, ISCs must be the "
                             "vectorized triangle of a square matrix.")
    elif not pairwise:
        n_subjects = iscs.shape[0]

    # Infer subjects, voxels and print for user to check
    n_voxels = iscs.shape[1]
    logger.info("Assuming {0} subjects with and {1} "
                "voxel(s) or ROI(s) in bootstrap ISC test.".format(n_subjects,
                                                                   n_voxels))

    return iscs, n_subjects, n_voxels


def compute_summary_statistic(iscs, summary_statistic='mean', axis=None):

    """Computes summary statistics for ISCs

    Computes either the 'mean' or 'median' across a set of ISCs. In the
    case of the mean, ISC values are first Fisher Z transformed (arctanh),
    averaged, then inverse Fisher Z transformed (tanh).

    The implementation is based on the work in [SilverDunlap1987]_.

    .. [SilverDunlap1987] "Averaging corrlelation coefficients: should
       Fisher's z transformation be used?", N. C. Silver, W. P. Dunlap, 1987,
       Journal of Applied Psychology, 72, 146-148.
       https://doi.org/10.1037/0021-9010.72.1.146

    Parameters
    ----------
    iscs : list or ndarray
        ISC values

    summary_statistic : str, default:'mean'
        Summary statistic, 'mean' or 'median'

    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

    Returns
    -------
    statistic : float or ndarray
        Summary statistic of ISC values

    """

    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Compute summary statistic
    if summary_statistic == 'mean':
        statistic = np.tanh(np.nanmean(np.arctanh(iscs), axis=axis))
    elif summary_statistic == 'median':
        statistic = np.nanmedian(iscs, axis=axis)

    return statistic


def bootstrap_isc(iscs, pairwise=False, summary_statistic='median',
                  n_bootstraps=1000, ci_percentile=95, random_state=None):

    """One-sample group-level bootstrap hypothesis test for ISCs

    For ISCs from one more voxels or ROIs, resample subjects with replacement
    to construct a bootstrap distribution. Input is a list or ndarray of
    ISCs for a single voxel/ROI, or an ISCs-by-voxels ndarray. ISC values
    should be either N ISC values for N subjects in the leave-one-out appraoch
    (pairwise=False), N(N-1)/2 ISC values for N subjects in the pairwise
    approach (pairwise=True). In the pairwise approach, ISC values should
    correspond to the vectorized upper triangle of a square corrlation matrix
    (see scipy.stats.distance.squareform). Shifts bootstrap distribution by
    actual summary statistic (effectively to zero) for two-tailed null
    hypothesis test (Hall & Wilson, 1991). Uses subject-wise (not pair-wise)
    resampling in the pairwise approach. Returns the observed ISC, the
    confidence interval, and a p-value for the bootstrap hypothesis test, as
    well as the bootstrap distribution of summary statistics. According to
    Chen et al., 2016, this is the preferred nonparametric approach for
    controlling false positive rates (FPR) for one-sample tests in the pairwise
    approach.

    The implementation is based on the work in [Chen2016]_ and
    [HallWilson1991]_.

    .. [HallWilson1991] "Two guidelines for bootstrap hypothesis testing.",
       P. Hall, S. R., Wilson, 1991, Biometrics, 757-762.
       https://doi.org/10.2307/2532163

    Parameters
    ----------
    iscs : list or ndarray, ISCs by voxels array
        ISC values for one or more voxels

    pairwise : bool, default:False
        Indicator of pairwise or leave-one-out, should match ISCs structure

    summary_statistic : str, default:'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_bootstraps : int, default:1000
        Number of bootstrap samples (subject-level with replacement)

    ci_percentile : int, default:95
         Percentile for computing confidence intervals

    random_state = int or None, default:None
        Initial random seed

    Returns
    -------
    observed : float, median (or mean) ISC value
        Summary statistic for actual ISCs

    ci : tuple, bootstrap confidence intervals
        Confidence intervals generated from bootstrap distribution

    p : float, p-value
        p-value based on bootstrap hypothesis test

    distribution : ndarray, bootstraps by voxels (optional)
        Bootstrap distribution if return_bootstrap=True

    """

    # Standardize structure of input data
    iscs, n_subjects, n_voxels = check_isc_input(iscs, pairwise=pairwise)

    # Check for valid summary statistic
    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Compute summary statistic for observed ISCs
    observed = compute_summary_statistic(iscs,
                                         summary_statistic=summary_statistic,
                                         axis=0)[np.newaxis, :]

    # Set up an empty list to build our bootstrap distribution
    distribution = []

    # Loop through n bootstrap iterations and populate distribution
    for i in np.arange(n_bootstraps):

        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # Randomly sample subject IDs with replacement
        subject_sample = sorted(prng.choice(np.arange(n_subjects),
                                            size=n_subjects))

        # Squareform and shuffle rows/columns of pairwise ISC matrix to
        # to retain correlation structure among ISCs, then get triangle
        if pairwise:

            # Loop through voxels
            isc_sample = []
            for voxel_iscs in iscs.T:

                # Square the triangle and fill diagonal
                voxel_iscs = squareform(voxel_iscs)
                np.fill_diagonal(voxel_iscs, 1)

                # Check that pairwise ISC matrix is square and symmetric
                assert voxel_iscs.shape[0] == voxel_iscs.shape[1]
                assert np.allclose(voxel_iscs, voxel_iscs.T)

                # Shuffle square correlation matrix and get triangle
                voxel_sample = voxel_iscs[subject_sample, :][:, subject_sample]
                voxel_sample = squareform(voxel_sample, checks=False)

                # Censor off-diagonal 1s for same-subject pairs
                voxel_sample[voxel_sample == 1.] = np.NaN

                isc_sample.append(voxel_sample)

            isc_sample = np.column_stack(isc_sample)

        # Get simple bootstrap sample if not pairwise
        elif not pairwise:
            isc_sample = iscs[subject_sample, :]

        # Compute summary statistic for bootstrap ISCs per voxel
        # (alternatively could construct distribution for all voxels
        # then compute statistics, but larger memory footprint)
        distribution.append(compute_summary_statistic(
                                isc_sample,
                                summary_statistic=summary_statistic,
                                axis=0))

        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.array(distribution)
    assert distribution.shape == (n_bootstraps, n_voxels)

    # Compute CIs of median from bootstrap distribution (default: 95%)
    ci = (np.percentile(distribution, (100 - ci_percentile)/2, axis=0),
          np.percentile(distribution, ci_percentile + (100 - ci_percentile)/2,
                        axis=0))

    # Shift bootstrap distribution to 0 for hypothesis test
    shifted = distribution - observed

    # Get p-value for actual median from shifted distribution
    p = compute_p_from_null_distribution(observed, shifted,
                                         side='two-sided', exact=False,
                                         axis=0)

    # Reshape p-values to fit with data shape
    p = p[np.newaxis, :]

    return observed, ci, p, distribution


def check_group_assignment(group_assignment, n_subjects):
    if type(group_assignment) == list:
        pass
    elif type(group_assignment) == np.ndarray:
        group_assignment = group_assignment.tolist()
    else:
        logger.info("No group assignment provided, "
                    "performing one-sample test.")

    if group_assignment and len(group_assignment) != n_subjects:
        raise ValueError("Group assignments ({0}) "
                         "do not match number of subjects ({1})!".format(
                                len(group_assignment), n_subjects))
    return group_assignment


def get_group_parameters(group_assignment, n_subjects, pairwise=False):

    # Set up dictionary to contain group info
    group_parameters = {'group_assignment': group_assignment,
                        'n_subjects': n_subjects,
                        'group_labels': None, 'groups': None,
                        'sorter': None, 'unsorter': None,
                        'group_matrix': None, 'group_selector': None}

    # Set up group selectors for two-group scenario
    if group_assignment and len(np.unique(group_assignment)) == 2:
        group_parameters['n_groups'] = 2

        # Get group labels and counts
        group_labels = np.unique(group_assignment)
        groups = {group_labels[0]: group_assignment.count(group_labels[0]),
                  group_labels[1]: group_assignment.count(group_labels[1])}

        # For two-sample pairwise approach set up selector from matrix
        if pairwise:
            # Sort the group_assignment variable if it came in shuffled
            # so it's easier to build group assignment matrix
            sorter = np.array(group_assignment).argsort()
            unsorter = np.array(group_assignment).argsort().argsort()

            # Populate a matrix with group assignments
            upper_left = np.full((groups[group_labels[0]],
                                  groups[group_labels[0]]),
                                 group_labels[0])
            upper_right = np.full((groups[group_labels[0]],
                                   groups[group_labels[1]]),
                                  np.nan)
            lower_left = np.full((groups[group_labels[1]],
                                  groups[group_labels[0]]),
                                 np.nan)
            lower_right = np.full((groups[group_labels[1]],
                                   groups[group_labels[1]]),
                                  group_labels[1])
            group_matrix = np.vstack((np.hstack((upper_left, upper_right)),
                                      np.hstack((lower_left, lower_right))))
            np.fill_diagonal(group_matrix, np.nan)
            group_parameters['group_matrix'] = group_matrix

            # Unsort matrix and squareform to create selector
            group_parameters['group_selector'] = squareform(
                                        group_matrix[unsorter, :][:, unsorter],
                                        checks=False)
            group_parameters['sorter'] = sorter
            group_parameters['unsorter'] = unsorter

        # If leave-one-out approach, just user group assignment as selector
        else:
            group_parameters['group_selector'] = group_assignment

        # Save these parameters for later
        group_parameters['groups'] = groups
        group_parameters['group_labels'] = group_labels

    # Manage one-sample and incorrect group assignments
    elif not group_assignment or len(np.unique(group_assignment)) == 1:
        group_parameters['n_groups'] = 1

        # If pairwise initialize matrix of ones for sign-flipping
        if pairwise:
            group_parameters['group_matrix'] = np.ones((
                                            group_parameters['n_subjects'],
                                            group_parameters['n_subjects']))

    elif len(np.unique(group_assignment)) > 2:
        raise ValueError("This test is not valid for more than "
                         "2 groups! (got {0})".format(
                                len(np.unique(group_assignment))))
    else:
        raise ValueError("Invalid group assignments!")

    return group_parameters


def permute_one_sample_iscs(iscs, group_parameters, i, pairwise=False,
                            summary_statistic='median', group_matrix=None,
                            exact_permutations=None, prng=None):

    """Applies one-sample permutations to ISC data

    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array.
    This function is only intended to be used internally by the
    permutation_isc function in this module.

    Parameters
    ----------
    iscs : ndarray or list
        ISC values

    group_parameters : dict
        Dictionary of group parameters

    i : int
        Permutation iteration

    pairwise : bool, default:False
        Indicator of pairwise or leave-one-out, should match ISCs variable

    summary_statistic : str, default:'median'
        Summary statistic, either 'median' (default) or 'mean'

    exact_permutations : list
        List of permutations

    prng = None or np.random.RandomState, default:None
        Initial random seed

    Returns
    -------
    isc_sample : ndarray
        Array of permuted ISC values

    """

    # Randomized sign-flips
    if exact_permutations:
        sign_flipper = np.array(exact_permutations[i])
    else:
        sign_flipper = prng.choice([-1, 1],
                                   size=group_parameters['n_subjects'],
                                   replace=True)

    # If pairwise, apply sign-flips by rows and columns
    if pairwise:
        matrix_flipped = (group_parameters['group_matrix'] * sign_flipper
                                                           * sign_flipper[
                                                                :, np.newaxis])
        sign_flipper = squareform(matrix_flipped, checks=False)

    # Apply flips along ISC axis (same across voxels)
    isc_flipped = iscs * sign_flipper[:, np.newaxis]

    # Get summary statistics on sign-flipped ISCs
    isc_sample = compute_summary_statistic(
                    isc_flipped,
                    summary_statistic=summary_statistic,
                    axis=0)

    return isc_sample


def permute_two_sample_iscs(iscs, group_parameters, i, pairwise=False,
                            summary_statistic='median',
                            exact_permutations=None, prng=None):

    """Applies two-sample permutations to ISC data

    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array.
    This function is only intended to be used internally by the
    permutation_isc function in this module.

    Parameters
    ----------
    iscs : ndarray or list
        ISC values

    group_parameters : dict
        Dictionary of group parameters

    i : int
        Permutation iteration

    pairwise : bool, default:False
        Indicator of pairwise or leave-one-out, should match ISCs variable

    summary_statistic : str, default:'median'
        Summary statistic, either 'median' (default) or 'mean'

    exact_permutations : list
        List of permutations

    prng = None or np.random.RandomState, default:None
        Initial random seed
        Indicator of pairwise or leave-one-out, should match ISCs variable

    Returns
    -------
    isc_sample : ndarray
        Array of permuted ISC values

    """

    # Shuffle the group assignments
    if exact_permutations:
        group_shuffler = np.array(exact_permutations[i])
    elif not exact_permutations and pairwise:
        group_shuffler = prng.permutation(np.arange(
            len(np.array(group_parameters['group_assignment'])[
                            group_parameters['sorter']])))
    elif not exact_permutations and not pairwise:
        group_shuffler = prng.permutation(np.arange(
            len(group_parameters['group_assignment'])))

    # If pairwise approach, convert group assignments to matrix
    if pairwise:

        # Apply shuffler to group matrix rows/columns
        group_shuffled = group_parameters['group_matrix'][
                            group_shuffler, :][:, group_shuffler]

        # Unsort shuffled matrix and squareform to create selector
        group_selector = squareform(group_shuffled[
                                    group_parameters['unsorter'], :]
                                    [:, group_parameters['unsorter']],
                                    checks=False)

    # Shuffle group assignments in leave-one-out two sample test
    elif not pairwise:

        # Apply shuffler to group matrix rows/columns
        group_selector = np.array(
                    group_parameters['group_assignment'])[group_shuffler]

    # Get difference of within-group summary statistics
    # with group permutation
    isc_sample = (compute_summary_statistic(
                    iscs[group_selector == group_parameters[
                                            'group_labels'][0], :],
                    summary_statistic=summary_statistic,
                    axis=0) -
                  compute_summary_statistic(
                    iscs[group_selector == group_parameters[
                                            'group_labels'][1], :],
                    summary_statistic=summary_statistic,
                    axis=0))

    return isc_sample


def permutation_isc(iscs, group_assignment=None, pairwise=False,  # noqa: C901
                    summary_statistic='median', n_permutations=1000,
                    random_state=None):

    """Group-level permutation test for ISCs

    For ISCs from one or more voxels or ROIs, permute group assignments to
    construct a permutation distribution. Input is a list or ndarray of
    ISCs  for a single voxel/ROI, or an ISCs-by-voxels ndarray. If two groups,
    ISC values should stacked along first dimension (vertically), and a
    group_assignment list (or 1d array) of same length as the number of
    subjects should be provided to indicate groups. If no group_assignment
    is provided, one-sample test is performed using a sign-flipping procedure.
    Performs exact test if number of possible permutations (2**N for one-sample
    sign-flipping, N! for two-sample shuffling) is less than or equal to number
    of requested permutation; otherwise, performs approximate permutation test
    using Monte Carlo resampling. ISC values should either be N ISC values for
    N subjects in the leave-one-out approach (pairwise=False) or N(N-1)/2 ISC
    values for N subjects in the pairwise approach (pairwise=True). In the
    pairwise approach, ISC values should correspond to the vectorized upper
    triangle of a square corrlation matrix (scipy.stats.distance.squareform).
    Note that in the pairwise approach, group_assignment order should match the
    row/column order of the subject-by-subject square ISC matrix even though
    the input ISCs should be supplied as the vectorized upper triangle of the
    square ISC matrix. Returns the observed ISC and permutation-based p-value
    (two-tailed test), as well as the permutation distribution of summary
    statistic. According to Chen et al., 2016, this is the preferred
    nonparametric approach for controlling false positive rates (FPR) for
    two-sample tests. This approach may yield inflated FPRs for one-sample
    tests.

    The implementation is based on the work in [Chen2016]_.

    Parameters
    ----------
    iscs : list or ndarray, correlation matrix of ISCs
        ISC values for one or more voxels

    group_assignment : list or ndarray, group labels
        Group labels matching order of ISC input

    pairwise : bool, default:False
        Indicator of pairwise or leave-one-out, should match ISCs variable

    summary_statistic : str, default:'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_permutations : int, default:1000
        Number of permutation iteration (randomizing group assignment)

    random_state = int, None, or np.random.RandomState, default:None
        Initial random seed

    Returns
    -------
    observed : float, ISC summary statistic or difference
        Actual ISC or group difference (excluding between-group ISCs)

    p : float, p-value
        p-value based on permutation test

    distribution : ndarray, permutations by voxels (optional)
        Permutation distribution if return_bootstrap=True
    """

    # Standardize structure of input data
    iscs, n_subjects, n_voxels = check_isc_input(iscs, pairwise=pairwise)

    # Check for valid summary statistic
    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Check match between group labels and ISCs
    group_assignment = check_group_assignment(group_assignment,
                                              n_subjects)

    # Get group parameters
    group_parameters = get_group_parameters(group_assignment, n_subjects,
                                            pairwise=pairwise)

    # Set up permutation type (exact or Monte Carlo)
    if group_parameters['n_groups'] == 1:
        if n_permutations < 2**n_subjects:
            logger.info("One-sample approximate permutation test using "
                        "sign-flipping procedure with Monte Carlo resampling.")
            exact_permutations = None
        else:
            logger.info("One-sample exact permutation test using "
                        "sign-flipping procedure with 2**{0} "
                        "({1}) iterations.".format(n_subjects,
                                                   2**n_subjects))
            exact_permutations = list(it.product([-1, 1], repeat=n_subjects))
            n_permutations = 2**n_subjects

    # Check for exact test for two groups
    else:
        if n_permutations < np.math.factorial(n_subjects):
            logger.info("Two-sample approximate permutation test using "
                        "group randomization with Monte Carlo resampling.")
            exact_permutations = None
        else:
            logger.info("Two-sample exact permutation test using group "
                        "randomization with {0}! "
                        "({1}) iterations.".format(
                                n_subjects,
                                np.math.factorial(n_subjects)))
            exact_permutations = list(it.permutations(
                np.arange(len(group_assignment))))
            n_permutations = np.math.factorial(n_subjects)

    # If one group, just get observed summary statistic
    if group_parameters['n_groups'] == 1:
        observed = compute_summary_statistic(
                        iscs,
                        summary_statistic=summary_statistic,
                        axis=0)[np.newaxis, :]

    # If two groups, get the observed difference
    else:
        observed = (compute_summary_statistic(
                        iscs[group_parameters['group_selector'] ==
                             group_parameters['group_labels'][0], :],
                        summary_statistic=summary_statistic,
                        axis=0) -
                    compute_summary_statistic(
                        iscs[group_parameters['group_selector'] ==
                             group_parameters['group_labels'][1], :],
                        summary_statistic=summary_statistic,
                        axis=0))
        observed = np.array(observed)[np.newaxis, :]

    # Set up an empty list to build our permutation distribution
    distribution = []

    # Loop through n permutation iterations and populate distribution
    for i in np.arange(n_permutations):

        # Random seed to be deterministically re-randomized at each iteration
        if exact_permutations:
            prng = None
        elif isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # If one group, apply sign-flipping procedure
        if group_parameters['n_groups'] == 1:
            isc_sample = permute_one_sample_iscs(
                            iscs, group_parameters, i,
                            pairwise=pairwise,
                            summary_statistic=summary_statistic,
                            exact_permutations=exact_permutations,
                            prng=prng)

        # If two groups, set up group matrix get the observed difference
        else:
            isc_sample = permute_two_sample_iscs(
                            iscs, group_parameters, i,
                            pairwise=pairwise,
                            summary_statistic=summary_statistic,
                            exact_permutations=exact_permutations,
                            prng=prng)

        # Tack our permuted ISCs onto the permutation distribution
        distribution.append(isc_sample)

        # Update random state for next iteration
        if not exact_permutations:
            random_state = np.random.RandomState(prng.randint(
                                                    0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.array(distribution)
    assert distribution.shape == (n_permutations, n_voxels)

    # Get p-value for actual median from shifted distribution
    if exact_permutations:
        p = compute_p_from_null_distribution(observed, distribution,
                                             side='two-sided', exact=True,
                                             axis=0)
    else:
        p = compute_p_from_null_distribution(observed, distribution,
                                             side='two-sided', exact=False,
                                             axis=0)

    # Reshape p-values to fit with data shape
    p = p[np.newaxis, :]

    return observed, p, distribution


def timeshift_isc(data, pairwise=False, summary_statistic='median',
                  n_shifts=1000, random_state=None):

    """Circular time-shift randomization for one-sample ISC test

    For each voxel or ROI, compute the actual ISC and p-values
    from a null distribution of ISCs where response time series
    are first circularly shifted by random intervals. If pairwise,
    apply time-shift randomization to each subjects and compute pairwise
    ISCs. If leave-one-out approach is used (pairwise=False), apply
    the random time-shift to only the left-out subject in each iteration
    of the leave-one-out procedure. Input data should be a list where
    each item is a time-points by voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a single ndarray is
    supplied, the last dimension is assumed to correspond to subjects.
    Returns the observed ISC and p-values (two-tailed test), as well as
    the null distribution of ISCs computed on randomly time-shifted data.

    The implementation is based on the work in [Kauppi2010]_ and
    [Kauppi2014]_.

    .. [Kauppi2010] "Inter-subject correlation of brain hemodynamic
       responses during watching a movie: localization in space and
       frequency.", J. P. Kauppi, I. P. Jääskeläinen, M. Sams, J. Tohka,
       2010, Frontiers in Neuroinformatics, 4, 5.
       https://doi.org/10.3389/fninf.2010.00005

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : str, default:'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_shifts : int, default:1000
        Number of randomly shifted samples

    random_state = int, None, or np.random.RandomState, default:None
        Initial random seed

    Returns
    -------
    observed : float, observed ISC (without time-shifting)
        Actual ISCs

    p : float, p-value
        p-value based on time-shifting randomization test

    distribution : ndarray, time-shifts by voxels (optional)
        Time-shifted null distribution if return_bootstrap=True
    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = check_timeseries_input(data)

    # Get actual observed ISC
    observed = isc(data, pairwise=pairwise,
                   summary_statistic=summary_statistic)

    # Roll axis to get subjects in first dimension for loop
    if pairwise:
        data = np.rollaxis(data, 2, 0)

    # Iterate through randomized shifts to create null distribution
    distribution = []
    for i in np.arange(n_shifts):

        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # Get a random set of shifts based on number of TRs,
        shifts = prng.choice(np.arange(n_TRs), size=n_subjects,
                             replace=True)

        # In pairwise approach, apply all shifts then compute pairwise ISCs
        if pairwise:

            # Apply circular shift to each subject's time series
            shifted_data = []
            for subject, shift in zip(data, shifts):
                shifted_data.append(np.concatenate(
                                        (subject[-shift:, :],
                                         subject[:-shift, :])))
            shifted_data = np.dstack(shifted_data)

            # Compute null ISC on shifted data for pairwise approach
            shifted_isc = isc(shifted_data, pairwise=pairwise,
                              summary_statistic=summary_statistic)

        # In leave-one-out, apply shift only to each left-out participant
        elif not pairwise:

            shifted_isc = []
            for s, shift in enumerate(shifts):
                shifted_subject = np.concatenate((data[-shift:, :, s],
                                                  data[:-shift, :, s]))
                nonshifted_mean = np.mean(np.delete(data, s, 2), axis=2)
                loo_isc = isc(np.dstack((shifted_subject, nonshifted_mean)),
                              pairwise=False,
                              summary_statistic=None)
                shifted_isc.append(loo_isc)

            # Get summary statistics across left-out subjects
            shifted_isc = compute_summary_statistic(
                                np.dstack(shifted_isc),
                                summary_statistic=summary_statistic,
                                axis=2)

        distribution.append(shifted_isc)

        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.vstack(distribution)
    assert distribution.shape == (n_shifts, n_voxels)

    # Get p-value for actual median from shifted distribution
    p = compute_p_from_null_distribution(observed, distribution,
                                         side='two-sided', exact=False,
                                         axis=0)

    # Reshape p-values to fit with data shape
    p = p[np.newaxis, :]

    return observed, p, distribution


def phaseshift_isc(data, pairwise=False, summary_statistic='median',
                   n_shifts=1000, random_state=None):

    """Phase randomization for one-sample ISC test

    For each voxel or ROI, compute the actual ISC and p-values
    from a null distribution of ISCs where response time series
    are phase randomized prior to computing ISC. If pairwise,
    apply phase randomization to each subject and compute pairwise
    ISCs. If leave-one-out approach is used (pairwise=False), only
    apply phase randomization to the left-out subject in each iteration
    of the leave-one-out procedure. Input data should be a list where
    each item is a time-points by voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a single ndarray is
    supplied, the last dimension is assumed to correspond to subjects.
    Returns the observed ISC and p-values (two-tailed test), as well as
    the null distribution of ISCs computed on phase-randomized data.

    The implementation is based on the work in [Lerner2011]_ and
    [Simony2016]_.

    .. [Lerner2011] "Topographic mapping of a hierarchy of temporal
       receptive windows using a narrated story.", Y. Lerner, C. J. Honey,
       L. J. Silbert, U. Hasson, 2011, Journal of Neuroscience, 31, 2906-2915.
       https://doi.org/10.1523/jneurosci.3684-10.2011

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : str, default:'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_shifts : int, default:1000
        Number of randomly shifted samples

    random_state = int, None, or np.random.RandomState, default:None
        Initial random seed

    Returns
    -------
    observed : float, observed ISC (without time-shifting)
        Actual ISCs

    p : float, p-value
        p-value based on time-shifting randomization test

    distribution : ndarray, time-shifts by voxels (optional)
        Time-shifted null distribution if return_bootstrap=True
    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = check_timeseries_input(data)

    # Get actual observed ISC
    observed = isc(data, pairwise=pairwise,
                   summary_statistic=summary_statistic)

    # Iterate through randomized shifts to create null distribution
    distribution = []
    for i in np.arange(n_shifts):

        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # Get randomized phase shifts
        if n_TRs % 2 == 0:
            # Why are we indexing from 1 not zero here? n_TRs / -1 long?
            pos_freq = np.arange(1, data.shape[0] // 2)
            neg_freq = np.arange(data.shape[0] - 1, data.shape[0] // 2, -1)
        else:
            pos_freq = np.arange(1, (data.shape[0] - 1) // 2 + 1)
            neg_freq = np.arange(data.shape[0] - 1,
                                 (data.shape[0] - 1) // 2, -1)

        phase_shifts = prng.rand(len(pos_freq), 1, n_subjects) * 2 * np.math.pi

        # In pairwise approach, apply all shifts then compute pairwise ISCs
        if pairwise:

            # Fast Fourier transform along time dimension of data
            fft_data = fft(data, axis=0)

            # Shift pos and neg frequencies symmetrically, to keep signal real
            fft_data[pos_freq, :, :] *= np.exp(1j * phase_shifts)
            fft_data[neg_freq, :, :] *= np.exp(-1j * phase_shifts)

            # Inverse FFT to put data back in time domain for ISC
            shifted_data = np.real(ifft(fft_data, axis=0))

            # Compute null ISC on shifted data for pairwise approach
            shifted_isc = isc(shifted_data, pairwise=True,
                              summary_statistic=summary_statistic)

        # In leave-one-out, apply shift only to each left-out participant
        elif not pairwise:

            # Roll subject axis in phaseshifts for loop
            phase_shifts = np.rollaxis(phase_shifts, 2, 0)

            shifted_isc = []
            for s, shift in enumerate(phase_shifts):

                # Apply FFT to left-out subject
                fft_subject = fft(data[:, :, s], axis=0)

                # Shift pos and neg frequencies symmetrically, keep signal real
                fft_subject[pos_freq, :] *= np.exp(1j * shift)
                fft_subject[neg_freq, :] *= np.exp(-1j * shift)

                # Inverse FFT to put data back in time domain for ISC
                shifted_subject = np.real(ifft(fft_subject, axis=0))

                # ISC of shifted left-out subject vs mean of N-1 subjects
                nonshifted_mean = np.mean(np.delete(data, s, 2), axis=2)
                loo_isc = isc(np.dstack((shifted_subject, nonshifted_mean)),
                              pairwise=False, summary_statistic=None)
                shifted_isc.append(loo_isc)

            # Get summary statistics across left-out subjects
            shifted_isc = compute_summary_statistic(
                            np.dstack(shifted_isc),
                            summary_statistic=summary_statistic, axis=2)
        distribution.append(shifted_isc)

        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.vstack(distribution)
    assert distribution.shape == (n_shifts, n_voxels)

    # Get p-value for actual median from shifted distribution
    p = compute_p_from_null_distribution(observed, distribution,
                                         side='two-sided', exact=False,
                                         axis=0)

    # Reshape p-values to fit with data shape
    p = p[np.newaxis, :]

    return observed, p, distribution
