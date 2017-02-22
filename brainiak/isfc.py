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
functional correlation (ISFC), and utility functions for loading subject data
files and masks

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
import nibabel as nib
import numpy as np


def isc(D, collapse_subj=True):
    """Intersubject correlation

    For each voxel, computes the correlation of each subject's timecourse with
    the mean of all other subjects' timecourses. By default the result is
    averaged across subjects, unless collapse_subj is set to False.

    Parameters
    ----------
    D : voxel by time by subject ndarray
        fMRI data for which to compute ISC

    collapse_subj : bool, default:True
        Whether to average across subjects before returning result

    Returns
    -------
    ISC : voxel ndarray (or voxel by subject ndarray, if collapse_subj=False)
        pearson correlation for each voxel, across subjects
    """

    n_vox = D.shape[0]
    n_subj = D.shape[2]
    ISC = np.zeros((n_vox, n_subj))

    # Loop across choice of leave-one-out subject
    for loo_subj in range(n_subj):
        group = np.mean(D[:, :, np.arange(n_subj) != loo_subj], axis=2)
        subj = D[:, :, loo_subj]
        for v in range(n_vox):
            ISC[v, loo_subj] = compute_correlation(group[v, :], subj[v, :])

    if collapse_subj:
        ISC = np.mean(ISC, axis=1)
    return ISC


def isfc(D, collapse_subj=True):
    """Intersubject functional correlation

    Computes the correlation between the timecoure of each voxel in each
    subject with the average of all other subjects' timecourses in *all*
    voxels. By default the result is averaged across subjects, unless
    collapse_subj is set to False.

    Uses the high performance compute_correlation routine from fcma.util

    Parameters
    ----------
    D : voxel by time by subject ndarray
        fMRI data for which to compute ISFC

    collapse_subj : bool, default:True
        Whether to average across subjects before returning result

    Returns
    -------
    ISFC : voxel by voxel ndarray
        (or voxel by voxel by subject ndarray, if collapse_subj=False)
        pearson correlation between all pairs of voxels, across subjects
    """

    n_vox = D.shape[0]
    n_subj = D.shape[2]
    ISFC = np.zeros((n_vox, n_vox, n_subj))

    # Loop across choice of leave-one-out subject
    for loo_subj in range(D.shape[2]):
        group = np.mean(D[:, :, np.arange(n_subj) != loo_subj], axis=2)
        subj = D[:, :, loo_subj]
        ISFC[:, :, loo_subj] = compute_correlation(group, subj)

        # Symmetrize matrix
        ISFC[:, :, loo_subj] = (ISFC[:, :, loo_subj] +
                                ISFC[:, :, loo_subj].T) / 2

    if collapse_subj:
        ISFC = np.mean(ISFC, axis=2)
    return ISFC


def load_subjects_nii(data_files, mask_file, mask_func=None):
    """Loading masked nifti data into matrix

    Given a list of subject data files and a mask, loads voxel timecourses
    inside mask into a matrix, which is returned along with the voxel
    coordinates of the mask. If mask_func is given, it specifies which mask
    values should be included (e.g. for thresholding a continuous-valued mask)

    Parameters
    ----------
    data_files : list of filenames of subject nii files

    mask_file : filename of mask nii

    mask_func : Callable[[ndarray], bool] : default x>0

    Returns
    -------
    D : voxel by time by subject ndarray
        all data within mask, from all subjects
    coords : tuple of 3 ndarrays
        x,y,z (as provided by nibabel) coordinates of mask voxel locations
    """

    mask_nii = nib.load(mask_file)
    mask = mask_nii.get_data()
    if mask_func is not None:
        mask = mask_func(mask)
    else:
        mask = mask > 0
    mask_shape = mask_nii.shape
    coords = np.where(mask)

    data_shape = nib.load(data_files[0]).shape
    D = np.zeros((np.sum(mask), data_shape[3], len(data_files)))
    for s in range(len(data_files)):
        nii = nib.load(data_files[s])
        if nii.shape != data_shape:
            raise ValueError("Data has different shapes across subjects")
        if nii.shape[:3] != mask_shape:
            raise ValueError("Data and mask have different shapes")
        D[:, :, s] = nii.get_data()[mask]

    return (D, coords)
