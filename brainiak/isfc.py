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
from .utils.utils import phase_randomize, p_from_null


def isc(D, collapse_subj=True, return_p=False, num_perm=1000,
        two_sided=False, random_state=0, float_type=np.float64):
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

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance to define the state of the
        random permutations generator.

    float_type : either float16, float32, or float64,
        Depends on the required precision
        and available memory in the system.
        All the arrays generated during the execution will be cast
        to specified float type in order to save memory.

    Returns
    -------
    ISC : voxel ndarray (or voxel by subject ndarray, if collapse_subj=False)
        pearson correlation for each voxel, across subjects

    p : ndarray the same shape as ISC (if return_p = True)
        p values for each ISC value under the null distribution
    """

    n_vox = D.shape[0]
    n_subj = D.shape[2]

    n_perm = num_perm*int(return_p)
    max_null = np.zeros(n_perm, dtype=float_type)
    min_null = np.zeros(n_perm, dtype=float_type)
    ISC = np.zeros((n_vox, n_subj), dtype=float_type)

    for loo_subj in range(n_subj):
        group = np.mean(D[:, :, np.arange(n_subj) != loo_subj], axis=2)
        subj = D[:, :, loo_subj]
        for v in range(n_vox):
            ISC[v, loo_subj] = stats.pearsonr(group[v, :],
                                              subj[v, :])[0]
    if collapse_subj:
        ISC = np.mean(ISC, axis=1)

    for p in range(n_perm):
        # Randomize phases of D to create next null dataset
        D = phase_randomize(D, random_state)
        # Loop across choice of leave-one-out subject
        tmp_ISC = np.zeros((n_vox, n_subj), dtype=float_type)
        for loo_subj in range(n_subj):
            group = np.mean(D[:, :, np.arange(n_subj) != loo_subj], axis=2)
            subj = D[:, :, loo_subj]
            for v in range(n_vox):
                tmp_ISC[v, loo_subj] = stats.pearsonr(group[v, :],
                                                      subj[v, :])[0]
        if collapse_subj:
            tmp_ISC = np.mean(tmp_ISC, axis=1)
        max_null[p] = np.max(tmp_ISC)
        min_null[p] = np.min(tmp_ISC)

    if return_p:
        p = p_from_null(ISC, two_sided,
                        max_null_input=max_null,
                        min_null_input=min_null)
        return ISC, p
    else:
        return ISC


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
