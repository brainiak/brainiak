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

#from brainiak.fcma.util import compute_correlation
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, zscore
import itertools as it

def isc(data, pairwise=False, summary_statistic=None, verbose=True):
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
    if verbose:
        print(f"Assuming {n_subjects} subjects with {n_TRs} time points "
              f"and {n_voxels} voxel(s) or ROI(s).")
    
    # Loop over each voxel or ROI
    voxel_iscs = []
    for v in np.arange(n_voxels):
        voxel_data = data[:, v, :].T
        if n_subjects == 2:
            iscs = pearsonr(voxel_data[0, :], voxel_data[1, :])[0]
            summary_statistic = None
            if verbose:
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
        raise ValueError("Unrecognized summary_statistic! Use None, np.median, or np.mean.")
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
                  return_distribution=False, random_state=None):
    
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
    resampling in the pairwise approach. Returns the observed ISC, the confidence
    interval, and a p-value for the bootstrap hypothesis test. Optionally returns
    the bootstrap distribution of summary statistics.According to Chen et al.,
    2016, this is the preferred nonparametric approach for controlling false
    positive rates (FPR) for one-sample tests in the pairwise approach.
    
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
         
    return_distribution : bool, default:False
        Optionally return the bootstrap distribution of summary statistics
        
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
        
    if n_subjects < 2:
        raise ValueError("Input data seems to contain only one subject! "
                         "Needs two or more subjects. Check that input is "
                         "not summary statistic.")
    
    # Infer subjects, voxels and print for user to check
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
        # (alternatively could construct distrubtion for all voxels
        # then compute statistics, but larger memory footprint)
        if summary_statistic == np.mean:
            distribution.append(np.tanh(np.nanmean(np.arctanh(isc_sample), axis=0)))
        elif summary_statistic == np.median:
            distribution.append(np.nanmedian(isc_sample, axis=0))
                    
        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, 2**32 - 1))
        
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


def permutation_isc(iscs, group_assignment=None, pairwise=False,
                    summary_statistic=np.median, n_permutations=1000,
                    return_distribution=False, random_state=None):
    
    """Group-level permutation test for ISCs

    For ISCs from one or more voxels or ROIs, permute group assignments to
    construct a permutation distribution. Input is a list or ndarray of
    ISCs  for a single voxel/ROI, or an ISCs-by-voxels ndarray. If two groups,
    ISC values should stacked along first dimension (vertically), and a
    group_assignment list (or 1d array) of same length as the number of
    subjects should be provided to indicate group labels. If no group_assignment
    is provided, a one-sample test is performed using a sign-flipping procedure.
    Performs exact test if number of possible permutations (2**N for one-sample
    sign-flipping, N! for two-sample shuffling) is less than or equal to number
    of requested permutation; otherwise, performs approximate permutation test
    using Monte Carlo resampling. ISC values should either be N ISC values for
    N subjects in the leave-one-out approach (pairwise=False) or N(N-1)/2 ISC
    values for N subjects in the pairwise approach (pairwise=True). In the
    pairwise approach, ISC values should correspond to the vectorized upper
    triangle of a square corrlation matrix (see scipy.stats.distance.squareform).
    Note that in the pairwise approach, group_assignment order should match the
    row/column order of the subject-by-subject square ISC matrix even though the
    input ISCs should be supplied as the vectorized upper triangle of the square
    ISC matrix. Returns the observed ISC and permutation-based p-value (two-tailed
    test). Optionall  returns the permutation distribution of summary statistics.
    According to Chen et al., 2016, this is the preferred nonparametric approach
    for controlling false positive rates (FPR) for two-sample tests. This approach
    may yield inflated FPRs for one-sample tests.
    
    The implementation is based on the following publications:
    
    .. [Chen2016] "Untangling the relatedness among correlations, part I: 
    nonparametric approaches to inter-subject correlation analysis at the
    group level.", G. Chen, Y. W. Shin, P. A. Taylor, D. R. Glen, R. C. 
    Reynolds, R. B. Israel, R. W. Cox, 2016, NeuroImage, 142, 248-259.
    
    .. [PhipsonSmyth2010] "Permutation p-values should never be zero:
    calculating exact p-values when permutations are randomly drawn.",
    B. Phipson, G. K., Smyth, 2010, Statistical Applications in Genetics
    and Molecular Biology, 9, 1544-6115.

    Parameters
    ----------
    iscs : list or ndarray, correlation matrix of iscs
        ISC values for one or more voxels

    group_assignment : list or ndarray, group labels
        Group labels matching order of ISC input
        
    pairwise : bool, default:False
        Indicator of pairwise or leave-one-out, should match ISCs variable
        
    summary_statistic : numpy function, default:np.median
        Summary statistic, either np.median (default) or np.mean

    n_permutations : int, default:1000
        Number of permutation iteration (randomizing group assignment)
        
    return_distribution : bool, default:False
        Optionally return the bootstrap distribution of summary statistics
        
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
            
    # Check match between group labels and ISCs
    if type(group_assignment) == list:
        pass
    elif type(group_assignment) == np.ndarray:
        group_assignment = group_assignment.tolist()
    else:
        print("No group assignment provided, performing one-sample test.")
    
    if group_assignment and len(group_assignment) != n_subjects:
        raise ValueError(f"Group assignments ({len(group_assignment)}) "
                         f"do not match number of subjects ({n_subjects})!")
    
    # Set up group selectors for two-group scenario
    if group_assignment and len(np.unique(group_assignment)) == 2:
        n_groups = 2
        
        # Get group labels and counts
        group_labels = np.unique(group_assignment)
        groups = {group_labels[0]: group_assignment.count(group_labels[0]),
                  group_labels[1]: group_assignment.count(group_labels[1])}

        # For two-sample pairwise approach set up selector from matrix
        if pairwise == True:
            # Sort the group_assignment variable if it came in shuffled
            # so it's easier to build group assignment matrix
            sorter = np.array(group_assignment).argsort()
            unsorter = np.array(group_assignment).argsort().argsort()
            
            # Populate a matrix with group assignments                                 
            group_matrix = np.vstack((np.hstack((np.full((groups[group_labels[0]],
                                                          groups[group_labels[0]]),
                                                         group_labels[0]),
                                                 np.full((groups[group_labels[0]],
                                                          groups[group_labels[1]]),
                                                        np.nan))),
                                      np.hstack((np.full((groups[group_labels[1]],
                                                          groups[group_labels[0]]),
                                                         np.nan),
                                                 np.full((groups[group_labels[1]],
                                                          groups[group_labels[1]]),
                                                         group_labels[1])))))
            np.fill_diagonal(group_matrix, np.nan)
            
            # Unsort matrix and squareform to create selector
            group_selector = squareform(group_matrix[unsorter, :][:, unsorter],
                                        checks=False)
            
        # If leave-one-out approach, just user group assignment as selector
        elif pairwise == False:
            group_selector = group_assignment
    
    # Manage one-sample and incorrect group assignments
    elif not group_assignment or len(np.unique(group_assignment)) == 1:
        n_groups = 1
        
        # If pairwise initialize matrix of ones for sign-flipping
        ones_matrix = np.ones((n_subjects, n_subjects))
        
    elif len(np.unique(group_assignment)) > 2:
        raise ValueError("This test is not valid for more than "
                         f"2 groups! (got {n_groups})")
    else:
        raise ValueError("Invalid group assignments!")
    
    # Infer subjects, groups, voxels and print for user to check
    n_voxels = iscs.shape[1]
    print(f"Assuming {n_subjects} subjects, {n_groups} group(s), "
          f"and {n_voxels} voxel(s) or ROI(s).")
    
    # Set up permutation type (exact or Monte Carlo)
    if n_groups == 1:
        if n_permutations < 2**n_subjects:
            print("One-sample approximate permutation test using sign-flipping "
                  "procedure with Monte Carlo resampling.")
            exact_permutations = None
        elif n_permutations >= 2**n_subjects:
            print("One-sample exact permutation test using sign-flipping "
                  f"procedure with 2**{n_subjects} ({2**n_subjects}) iterations.")
            exact_permutations = list(it.product([-1, 1], repeat=n_subjects))
            n_permutations = 2**n_subjects
    elif n_groups == 2:
        if n_permutations < np.math.factorial(n_subjects):
            print("Two-sample approximate permutation test using "
                  "group randomization with Monte Carlo resampling.")
            exact_permutations = None
        elif n_permutations >= np.math.factorial(n_subjects):
            print("Two-sample exact permutation test using group "
                  f"randomization with {n_subjects}! "
                  f"({np.math.factorial(n_subjects)}) "
                  "iterations.")
            exact_permutations = list(it.permutations(
                np.arange(len(group_assignment))))
            n_permutations = np.math.factorial(n_subjects)
    
    # If one group, just get observed summary statistic
    if n_groups == 1:
        if summary_statistic == np.mean:
            observed = np.tanh(np.mean(np.arctanh(iscs), axis=0))
        elif summary_statistic == np.median:
            observed = np.median(iscs, axis=0)
    
    # If two groups, get the observed difference
    elif n_groups == 2:
        
        if summary_statistic == np.mean:
            observed = (np.tanh(np.mean(np.arctanh(
                iscs[group_selector == group_labels[0], :]), axis=0)) - 
                        np.tanh(np.mean(np.arctanh(
                iscs[group_selector == group_labels[1], :]), axis=0)))
        elif summary_statistic == np.median:
            observed = (np.median(
                iscs[group_selector == group_labels[0], :], axis=0) - 
                        np.median(
                iscs[group_selector == group_labels[1], :], axis=0))
        observed = np.array(observed)[np.newaxis, :]
        
    # Set up an empty list to build our permutation distribution
    distribution = []
    
    # Loop through n permutation iterations and populate distribution
    for i in np.arange(n_permutations):
        
        # Random seed to be deterministically re-randomized at each iteration
        if exact_permutations:
            pass
        elif isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)
        
        # If one group, apply sign-flipping procedure
        if n_groups == 1:
            
            # Randomized sign-flips
            if exact_permutations:
                sign_flipper = np.array(exact_permutations[i])
            elif not exact_permutations:
                sign_flipper = prng.choice([-1, 1], size=n_subjects, replace=True)
            
            # If pairwise, apply sign-flips by rows and columns
            if pairwise:
                matrix_flipped = (ones_matrix * sign_flipper
                                              * sign_flipper[:, np.newaxis])
                sign_flipper = squareform(matrix_flipped, checks=False)
            
            # Apply flips along ISC axis (same across voxels)
            isc_flipped = iscs * sign_flipper[:, np.newaxis]
            
            # Get summary statistics on sign-flipped ISCs
            if summary_statistic == np.mean:
                isc_sample = np.tanh(np.mean(np.arctanh(isc_flipped), axis=0))
            elif summary_statistic == np.median:
                isc_sample = np.median(isc_flipped, axis=0)            
        
        # If two groups, set up group matrix get the observed difference
        elif n_groups == 2:

            # Shuffle the group assignments
            if exact_permutations:
                group_shuffler = np.array(exact_permutations[i])
            elif not exact_permutations and pairwise:
                group_shuffler = prng.permutation(np.arange(
                    len(np.array(group_assignment)[sorter])))
            elif not exact_permutations and not pairwise:
                group_shuffler = prng.permutation(np.arange(
                    len(group_assignment)))
            
            # If pairwise approach, convert group assignments to matrix
            if pairwise:
                
                # Apply shuffler to group matrix rows/columns
                group_shuffled = group_matrix[group_shuffler, :][:, group_shuffler]
                
                # Unsort shuffled matrix and squareform to create selector
                group_selector = squareform(group_shuffled[unsorter, :][:, unsorter],
                                            checks=False)
                
            # Shuffle group assignments in leave-one-out two sample test
            elif not pairwise:
                
                # Apply shuffler to group matrix rows/columns
                group_selector = np.array(group_assignment)[group_shuffler]
                
            # Get difference of within-group summary statistics
            # with group permutation        
            if summary_statistic == np.mean:
                isc_sample = (np.tanh(np.mean(np.arctanh(
                    iscs[group_selector == group_labels[0], :]), axis=0)) - 
                            np.tanh(np.mean(np.arctanh(
                    iscs[group_selector == group_labels[1], :]), axis=0)))
            elif summary_statistic == np.median:
                isc_sample = (np.median(
                    iscs[group_selector == group_labels[0], :], axis=0) - 
                            np.median(
                    iscs[group_selector == group_labels[1], :], axis=0))
        
        # Tack our permuted ISCs onto the permutation distribution
        distribution.append(isc_sample) 
        
        # Update random state for next iteration
        if not exact_permutations:
            random_state = np.random.RandomState(prng.randint(0, 2**32 - 1))
        
    # Convert distribution to numpy array
    distribution = np.array(distribution)
    assert distribution.shape == (n_permutations, n_voxels)

    # Get p-value for actual median from shifted distribution
    p = ((np.sum(np.abs(distribution) >= np.abs(observed), axis=0) + 1) /
          float((len(distribution) + 1)))[np.newaxis, :]
    
    if return_distribution:
        return observed, p, distribution
    elif not return_distribution:
        return observed, p


def timeshift_isc(data, pairwise=False, summary_statistic=np.median,
                  n_shifts=1000, return_distribution=False, random_state=None):
    
    """Circular time-shift randomization for one-sample ISC test
    
    For a single voxel/ROI, take in response time series for multiple
    subjects and apply a random temporal shift interval to each subject
    prior to computing ISCs. Input should be list or dictionary of
    n_samples x n_voxels time series data where each item in the list
    or value in the dictionary corresponds to one subject's data.
    
    This implementation is based on the following publications:

    .. [Kauppi2010] "Inter-subject correlation of brain hemodynamic 
    responses during watching a movie: localization in space and
    frequency.", J. P. Kauppi, I. P. Jääskeläinen, M. Sams, J. Tohka,
    2010, Frontiers in Neuroinformatics, 4, 5.

    .. [Kauppi2014] "A versatile software package for inter-subject
    correlation based analyses of fMRI.", J. P. Kauppi, J. Pajula, 
    J. Tohka, 2014, Frontiers in Neuroinformatics, 8, 2.

    Parameters
    ----------
    data : list or dict, time series data for multiple subjects
        List or dictionary of response time series for multiple subjects

    pairwise : bool, default:False
        Indicator of pairwise or leave-one-out, should match iscs variable

    summary_statistic : numpy function, default:np.median
        Summary statistic, either np.median (default) or np.mean
        
    n_shifts : int, default:1000
        Number of randomly shifted samples
        
    return_distribution : bool, default:False
        Optionally return the bootstrap distribution of summary statistics
        
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
    
    # Get actual observed ISC
    observed = isc(data, pairwise=pairwise, summary_statistic=summary_statistic)
    
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
                                        (subject[-shift:, :], subject[:-shift, :])))
            shifted_data = np.dstack(shifted_data)

            # Compute null ISC on shifted data for pairwise approach
            shifted_isc = isc(shifted_data, pairwise=pairwise,
                              summary_statistic=summary_statistic, verbose=False)
        
        # In leave-one-out, apply shift only to each left-out participant
        elif not pairwise:
            
            shifted_isc = []
            for s, shift in enumerate(shifts):
                shifted_subject = np.concatenate((data[-shift:, :, s], data[:-shift, :, s]))
                nonshifted_mean = np.mean(np.delete(data, s, 2), axis=2)
                loo_isc = isc(np.dstack((shifted_subject, nonshifted_mean)), pairwise=False,
                              summary_statistic=None, verbose=False)
                shifted_isc.append(loo_isc)
            if summary_statistic == np.mean:
                shifted_isc = np.tanh(np.mean(np.arctanh(np.dstack(shifted_isc)), axis=2))
            elif summary_statistic == np.median:
                shifted_isc = np.median(np.dstack(shifted_isc), axis=2)
                
        distribution.append(shifted_isc)
        
        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, 2**32 - 1))
        
    # Convert distribution to numpy array
    distribution = np.vstack(distribution)
    assert distribution.shape == (n_shifts, n_voxels)

    # Get p-value for actual median from shifted distribution
    p = ((np.sum(np.abs(distribution) >= np.abs(observed), axis=0) + 1) /
          float((len(distribution) + 1)))[np.newaxis, :]
    
    if return_distribution:
        return observed, p, distribution
    elif not return_distribution:
        return observed, p