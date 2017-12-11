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
"""Intersubject analyses (ISC/ISFC)

Functions for computing intersubject correlation (ISC) and intersubject
functional correlation (ISFC)

Paper references:
ISC: Hasson, U., Nir, Y., Levy, I., Fuhrmann, G. & Malach, R. Intersubject
synchronization of cortical activity during natural vision. Science 303,
1634–1640 (2004).

ISFC: Simony E, Honey CJ, Chen J, Lositsky O, Yeshurun Y, Wiesel A, Hasson U
(2016) Dynamic reconfiguration of the default mode network during narrative
comprehension. Nat Commun 7.
"""

# Authors: Christopher Baldassano and Mor Regev
# Princeton University, 2017

from brainiak.fcma.util import compute_correlation
import numpy as np
from scipy import stats
from scipy.fftpack import fft, ifft
import math


def isc(D, collapse_subj=True, return_p=False, num_perm=1000, two_sided=False):
    """Intersubject correlation

    For each voxel, computes the correlation of each subject's timecourse with
    the mean of all other subjects' timecourses. By default the result is
    averaged across subjects, unless collapse_subj is set to False. A null
    distribution can optionally be computed using phase randomization, to
    compute a p value for each voxel.

    Parameters
    ----------
    D : voxel by time by subject ndarray
        fMRI data for which to compute ISC

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

    Returns
    -------
    ISC : voxel ndarray (or voxel by subject ndarray, if collapse_subj=False)
        pearson correlation for each voxel, across subjects

    p : ndarray the same shape as ISC (if return_p = True)
        p values for each ISC value under the null distribution
    """

    n_vox = D.shape[0]
    n_subj = D.shape[2]

    if return_p:
        n_perm = num_perm
    else:
        n_perm = 0

    ISC = np.zeros((n_vox, n_subj, n_perm + 1))

    for p in range(n_perm + 1):
        # Loop across choice of leave-one-out subject
        for loo_subj in range(n_subj):
            group = np.mean(D[:, :, np.arange(n_subj) != loo_subj], axis=2)
            subj = D[:, :, loo_subj]
            for v in range(n_vox):
                ISC[v, loo_subj, p] = stats.pearsonr(group[v, :],
                                                     subj[v, :])[0]

        # Randomize phases of D to create next null dataset
        D = phase_randomize(D)

    if collapse_subj:
        ISC = np.mean(ISC, axis=1)

    if not return_p:
        return np.squeeze(ISC)

    # Compute maximum/minimum ISC in each null dataset
    if collapse_subj:
        max_null = np.max(ISC[:, 1:], axis=0)
        min_null = np.min(ISC[:, 1:], axis=0)
        ISC = ISC[:, 0]
    else:
        max_null = np.max(ISC[:, :, 1:], axis=(0, 1))
        min_null = np.min(ISC[:, :, 1:], axis=(0, 1))
        ISC = ISC[:, :, 0]

    # Compute where the true values fall on the null distribution
    max_null_ecdf = ecdf(max_null)
    if two_sided:
        min_null_ecdf = ecdf(min_null)
        p = 2 * np.minimum(1 - max_null_ecdf(ISC), min_null_ecdf(ISC))
        p = np.minimum(p, 1)
    else:
        p = 1 - max_null_ecdf(ISC)
    return ISC, p


def isfc(D, collapse_subj=True, return_p=False,
         num_perm=1000, two_sided=False):
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

    if return_p:
        n_perm = num_perm
    else:
        n_perm = 0

    ISFC = np.zeros((n_vox, n_vox, n_subj, n_perm + 1))

    for p in range(n_perm + 1):
        # Loop across choice of leave-one-out subject
        for loo_subj in range(D.shape[2]):
            group = np.mean(D[:, :, np.arange(n_subj) != loo_subj], axis=2)
            subj = D[:, :, loo_subj]
            ISFC[:, :, loo_subj, p] = compute_correlation(group, subj)

            # Symmetrize matrix
            ISFC[:, :, loo_subj, p] = (ISFC[:, :, loo_subj, p] +
                                       ISFC[:, :, loo_subj, p].T) / 2

        # Randomize phases of D to create next null dataset
        D = phase_randomize(D)

    if collapse_subj:
        ISFC = np.mean(ISFC, axis=2)

    if not return_p:
        return np.squeeze(ISFC)

    # Compute maximum/minimum ISFC in each null dataset
    if collapse_subj:
        max_null = np.max(ISFC[:, :, 1:], axis=(0, 1))
        min_null = np.min(ISFC[:, :, 1:], axis=(0, 1))
        ISFC = ISFC[:, :, 0]
    else:
        max_null = np.max(ISFC[:, :, :, 1:], axis=(0, 1, 2))
        min_null = np.min(ISFC[:, :, 1:], axis=(0, 1))
        ISFC = ISFC[:, :, :, 0]

    # Compute where the true values fall on the null distribution
    max_null_ecdf = ecdf(max_null)
    if two_sided:
        min_null_ecdf = ecdf(min_null)
        p = 2 * np.minimum(1 - max_null_ecdf(ISFC), min_null_ecdf(ISFC))
        p = np.minimum(p, 1)
    else:
        p = 1 - max_null_ecdf(ISFC)
    return ISFC, p


def phase_randomize(D):
    """Randomly shift signal phases

    For each timecourse (from each voxel and each subject), computes its DFT
    and then randomly shifts the phase of each frequency before inverting
    back into the time domain. This yields timecourses with the same power
    spectrum (and thus the same autocorrelation) as the original timecourses,
    but will remove any meaningful temporal relationships between the
    timecourses.

    Parameters
    ----------
    D : voxel by time by subject ndarray
        fMRI data to be phase randomized

    Returns
    ----------
    ndarray of same shape as D
        phase randomized timecourses
    """

    F = fft(D, axis=1)
    if D.shape[1] % 2 == 0:
        pos_freq = np.arange(1, D.shape[1] // 2)
        neg_freq = np.arange(D.shape[1] - 1, D.shape[1] // 2, -1)
    else:
        pos_freq = np.arange(1, (D.shape[1] - 1) // 2 + 1)
        neg_freq = np.arange(D.shape[1] - 1, (D.shape[1] - 1) // 2, -1)

    shift = np.random.rand(D.shape[0], len(pos_freq), D.shape[2]) * 2 * math.pi

    # Shift pos and neg frequencies symmetrically, to keep signal real
    F[:, pos_freq, :] *= np.exp(1j * shift)
    F[:, neg_freq, :] *= np.exp(-1j * shift)

    return np.real(ifft(F, axis=1))


def ecdf(x):
    """Empirical cumulative distribution function

    Given a 1D array of values, returns a function f(q) that outputs the
    fraction of values less than or equal to q.

    Parameters
    ----------
    x : 1D array
        values for which to compute CDF

    Returns
    ----------
    ecdf_fun: Callable[[float], float]
        function that returns the value of the CDF at a given point
    """
    xp = np.sort(x)
    yp = np.arange(len(xp) + 1) / len(xp)

    def ecdf_fun(q):
        return yp[np.searchsorted(xp, q, side="right")]
    return ecdf_fun
