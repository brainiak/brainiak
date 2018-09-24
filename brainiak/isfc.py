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

Functions for computing intersubject correlation (ISC) and variations
including intersubject fucntional correlations (ISFC)

Paper references:
ISC: Hasson, U., Nir, Y., Levy, I., Fuhrmann, G. & Malach, R. Intersubject
synchronization of cortical activity during natural vision. Science 303,
1634–1640 (2004).

ISFC: Simony E, Honey CJ, Chen J, Lositsky O, Yeshurun Y, Wiesel A, Hasson U
(2016) Dynamic reconfiguration of the default mode network during narrative
comprehension. Nat Commun 7.
"""

# Authors: Sam Nastase, Christopher Baldassano, Mai Nguyen, and Mor Regev
# Princeton University, 2018

from brainiak.fcma.util import compute_correlation
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, zscore

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
    (see scipy.spatial.distance.squareform). Alternatively, supply either
    np.mean or np.median to compute summary statistic of ISCs (Fisher Z will
    be applied and inverted if using mean). Input data should be a list 
    where each item is a time-points by voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a single ndarray is
    supplied, the last dimension is assumed to correspond to subjects. If 
    only two subjects are supplied, simply compute Pearson correlation
    (precludes averaging in leave-one-out approach, and does not apply
    summary statistic.) Output is an ndarray where the first dimension is
    the number of subjects or pairs and the second dimension is the number
    of voxels (or ROIs).
        
    The implementation is based on the following publication:
    
    .. [Hasson2004] "Intersubject synchronization of cortical activity 
    during natural vision.", U. Hasson, Y. Nir, I. Levy, G. Fuhrmann,
    R. Malach, 2004, Science, 303, 1634-1640.

    Parameters
    ----------
    data : list or ndarray
        fMRI data for which to compute ISC
        
    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach
        
    summary_statistic : None
        Return all ISCs or collapse using np.mean or np.median

    Returns
    -------
    iscs : subjects or pairs by voxels ndarray
        ISC for each subject or pair (or summary statistic) per voxel

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
    elif type(data) == np.ndarray:
        if data.ndim == 2:
            data = data[:, np.newaxis, :]            
        elif data.ndim == 3:
            pass
        else:
            raise ValueError("Input ndarray should have 2 "
                             f"or 3 dimensions (got {data.ndim})!")

    # Infer subjects, TRs, voxels and print for user to check
    n_subjects = data.shape[2]
    n_TRs = data.shape[0]
    n_voxels = data.shape[1]
    print(f"Assuming {n_subjects} subjects with {n_TRs} time points "
          f"and {n_voxels} voxel(s) or ROI(s).")
    
    # Loop over each voxel or ROI
    voxel_iscs = []
    for v in np.arange(n_voxels):
        voxel_data = data[:, v, :].T
        if n_subjects == 2:
            iscs = pearsonr(voxel_data[0, :], voxel_data[1, :])[0]
            summary_statistic = None
            print("Only two subjects! Simply computing Pearson correlation.")
        elif pairwise:
            iscs = squareform(np.corrcoef(voxel_data), checks=False)
        elif not pairwise:
            iscs = np.array([pearsonr(subject,
                                      np.mean([s for s in voxel_data
                                               if s is not subject],
                                              axis=0))[0]
                    for subject in voxel_data])
        voxel_iscs.append(iscs)
    iscs = np.column_stack(voxel_iscs)
    
    # Summarize results (if requested)
    if summary_statistic == np.mean:
        iscs = np.tanh(summary_statistic(np.arctanh(iscs), axis=0))[np.newaxis, :]
    elif summary_statistic == np.median:    
        iscs = summary_statistic(iscs, axis=0)[np.newaxis, :]
    elif not summary_statistic:
        pass
    else:
        raise TypeError("Unrecognized summary_statistic! Use None, np.median, or np.mean.")
    return iscs


def isfc(D, collapse_subj=True, return_p=False, num_perm=1000,
         two_sided=False, random_state=0, float_type=np.float64):
    """Intersubject functional correlation

    Computes the correlation between the timecoure of each voxel in each
    subject with the average of all other subjects' timecourses in *all*
    voxels. By default the result is averaged across subjects, unless
    collapse_subj is set to False. A null distribution can optionally be
    computed using phase randomization, to compute a p value for each voxel-to-
    voxel correlation.

    Uses the high performance compute_correlation routine from fcma.util

    Parameters
    ----------
    D : voxel by time by subject ndarray
        fMRI data for which to compute ISFC

    collapse_subj : bool, default:True
        Whether to average across subjects before returning result

    return_p : bool, default:False
        Whether to use phase randomization to compute a p value for each voxel

    num_perm : int, default:1000
        Number of null samples to use for computing p values

    two_sided : bool, default:False
        Whether the p value should be one-sided (testing only for being
        above the null) or two-sided (testing for both significantly positive
        and significantly negative values)

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance to define the state of the
        random permutations generator.

    float_type : either float16, float32, or float64
        Depends on the required precision
        and available memory in the system.
        All the arrays generated during the execution will be cast
        to specified float type in order to save memory.

    Returns
    -------
    ISFC : voxel by voxel ndarray
        (or voxel by voxel by subject ndarray, if collapse_subj=False)
        pearson correlation between all pairs of voxels, across subjects

    p : ndarray the same shape as ISC (if return_p = True)
        p values for each ISC value under the null distribution
    """

    n_vox = D.shape[0]
    n_subj = D.shape[2]

    n_perm = num_perm*int(return_p)
    max_null = -np.ones(n_perm, dtype=float_type)
    min_null = np.ones(n_perm, dtype=float_type)

    ISFC = np.zeros((n_vox, n_vox, n_subj), dtype=float_type)

    for loo_subj in range(D.shape[2]):
        group = np.mean(D[:, :, np.arange(n_subj) != loo_subj], axis=2)
        subj = D[:, :, loo_subj]
        tmp_ISFC = compute_correlation(group, subj).astype(float_type)
        # Symmetrize matrix
        tmp_ISFC = (tmp_ISFC+tmp_ISFC.T)/2
        ISFC[:, :, loo_subj] = tmp_ISFC
    if collapse_subj:
        ISFC = np.mean(ISFC, axis=2)

    for p in range(n_perm):
        # Randomize phases of D to create next null dataset
        D = phase_randomize(D, random_state)
        # Loop across choice of leave-one-out subject
        ISFC_null = np.zeros((n_vox, n_vox), dtype=float_type)
        for loo_subj in range(D.shape[2]):
            group = np.mean(D[:, :, np.arange(n_subj) != loo_subj], axis=2)
            subj = D[:, :, loo_subj]
            tmp_ISFC = compute_correlation(group, subj).astype(float_type)
            # Symmetrize matrix
            tmp_ISFC = (tmp_ISFC+tmp_ISFC.T)/2

            if not collapse_subj:
                max_null[p] = max(np.max(tmp_ISFC), max_null[p])
                min_null[p] = min(np.min(tmp_ISFC), min_null[p])
            ISFC_null = ISFC_null + tmp_ISFC/n_subj

        if collapse_subj:
            max_null[p] = np.max(ISFC_null)
            min_null[p] = np.min(ISFC_null)

    if return_p:
        p = p_from_null(ISFC, two_sided,
                        max_null_input=max_null,
                        min_null_input=min_null)
        return ISFC, p
    else:
        return ISFC

'''    two_sided : bool, default:False
        Whether the p value should be one-sided (testing only for being
        above the null) or two-sided (testing for both significantly positive
        and significantly negative values)

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance to define the state of the
        random permutations generator.'''
    
    
def bootstrap_isc(iscs, pairwise=False, summary_statistic=np.median,
                  n_bootstraps=1000, ci_percentile=95,
                  return_distribution=False):
    
    """One-sample group-level bootstrap hypothesis test for iscs

    For ISCs from one more voxels or ROIs, resample subjects with replacement
    to construct a bootstrap distribution. Input is either a list or ndarray
    of ISCs for a single voxel/ROI, or an ISCs-by-voxels ndarray. ISC values
    should be either N ISC values for N subjects in the leave-one-out appraoch
    (pairwise=False), N(N-1)/2 ISC values for N subjects in the pairwise
    approach (pairwise=True). In the pairwise approach, ISC values should 
    correspond to the vectorized upper triangle of a square corrlation matrix
    (see scipy.stats.distance.squareform). Shifts bootstrap by actual median
    (effectively to zero) for two-tailed null hypothesis test (Hall & Wilson,
    1991). Uses subject-wise (not pair-wise) resampling in the pairwise approach.
    Returns the observed ISC, the confidence interval, and a p-value for the
    bootstrap hypothesis test. According to Chen et al., 2016, this is the
    preferred nonparametric approach for controlling false positive rates (FPR)
    for one-sample tests in the pairwise approach. Optionally, you can return
    the bootstrap distribution of summary statistics (memory intensive).
    
    The implementation is based on the following publications:
    
    .. [Chen2016] "Untangling the relatedness among correlations, part I: 
    nonparametric approaches to inter-subject correlation analysis at the
    group level.", G. Chen, Y. W. Shin, P. A. Taylor, D. R. Glen, R. C. 
    Reynolds, R. B. Israel, R. W. Cox, 2016, NeuroImage, 142, 248-259.
    
    .. [HallWilson1991] "Two guidelines for bootstrap hypothesis testing.",
    P. Hall, S. R., Wilson, 1991, Biometrics, 757-762.

    Parameters
    ----------
    iscs : list or ndarray, ISCs by voxels array
        ISC values for one or more voxels

    pairwise : bool, default:False
        Indicator of pairwise or leave-one-out, should match ISCs structure

    summary_statistic : numpy function, default:np.median
        Summary statistic, either np.median (default) or np.mean

    n_bootstraps : int, default:1000
        Number of bootstrap samples (subject-level with replacement)

    ci_percentile : int, default:95
         Percentile for computing confidence intervals

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
    if type(iscs) == list:
        iscs = np.array(iscs)[:, np.newaxis]
        
    elif type(iscs) == np.ndarray:
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
    
    # Infer subjects, TRs, voxels and print for user to check
    n_voxels = iscs.shape[1]
    print(f"Assuming {n_subjects} subjects with and {n_voxels} "
           "voxel(s) or ROI(s).")
    
    # Compute summary statistic for observed ISCs
    if summary_statistic == np.mean:
        observed = np.tanh(np.mean(np.arctanh(iscs), axis=0))[np.newaxis, :]
    elif summary_statistic == np.median:
        observed = summary_statistic(iscs, axis=0)[np.newaxis, :]
    else:
        raise TypeError("Unrecognized summary_statistic! Use np.median or np.mean.")
    
    # Set up an empty list to build our bootstrap distribution
    distribution = []
    
    # Loop through n bootstrap iterations and populate distribution
    for i in np.arange(n_bootstraps):

        # Randomly sample subject IDs with replacement
        subject_sample = sorted(np.random.choice(np.arange(n_subjects),
                                                 size=n_subjects))
        
        # Loop through voxels
        voxel_statistics = []
        for voxel_iscs in iscs.T:

            # Squareform and shuffle rows/columns of pairwise ISC matrix to
            # to retain correlation structure among ISCs, then get triangle
            if pairwise:

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

            # Get simple bootstrap sample of not pairwise
            elif not pairwise:
                voxel_sample = voxel_iscs[subject_sample]
                
            # Compute summary statistic for bootstrap ISCs per voxel
            # (alternatively could construct distrubtion for all voxels
            # then compute statistics, but larger memory footprint)
            if summary_statistic == np.mean:
                voxel_statistics.append(np.tanh(np.nanmean(np.arctanh(voxel_sample), axis=0)))
            elif summary_statistic == np.median:
                voxel_statistics.append(np.nanmedian(voxel_sample, axis=0))
            
        distribution.append(voxel_statistics)
        
    # Convert distribution to numpy array
    distribution = np.array(distribution)
    assert distribution.shape == (n_bootstraps, n_voxels)

    # Compute CIs of median from bootstrap distribution (default: 95%)
    ci = (np.percentile(distribution, (100 - ci_percentile)/2, axis=0),
          np.percentile(distribution, ci_percentile + (100 - ci_percentile)/2, axis=0))
    
    # Shift bootstrap distribution to 0 for hypothesis test
    shifted = distribution - observed
    
    # Get p-value for actual median from shifted distribution
    p = ((np.sum(np.abs(shifted) >= np.abs(observed), axis=0) + 1) /
          float((len(shifted) + 1)))[np.newaxis, :]
    
    if return_distribution:
        return observed, ci, p, distribution
    elif not return_distribution:
        return observed, ci, p


