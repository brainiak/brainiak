#  Copyright 2016 Intel Corporation
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

"""Topological Data Analysis

Prepare the steps for TDA analyses:

Preprocess the volumes involving binarizing, smoothing and normalizing

Run a correlation of every voxel against every other voxel

Takes in a time series, ignoring the conditions.
It does some voxel selection on this time series.
It then runs a correlation on all of these selected voxels.
Now each voxel can be represented as existing in a higher dimensional space now

volume is organized as an Voxel x Timepoint matrix
voxel_number describes how many voxels are to be used in the correlation
selection contains the procedure for selecting voxels: Ttest, Variance
distance is the procedure for calculating the distance matrix:
Dist, InverseCor, InverseAbsCor or none

 Authors: Cameron Ellis (Princeton) 2016
"""
import logging

import numpy as np
import scipy.ndimage as ndimage
from scipy import spatial
from scipy import stats

__all__ = [
    "convert_space",
    "DistanceFuncs",
    "preprocess",
    "SelectionFuncs",
]

logger = logging.getLogger(__name__)


def preprocess(volume, t_score=True, gauss_size=0, normalize=False):
    """Preprocess the data for TDA relevant features

    Parameters
    ----------

    volume : 2d array, float
        fMRI data, voxel by TR.

    t_score : boolean, default: True
       Do you use the t values of the data or do you binarize based on some
       threshold?

    gauss_size: float, default: 0
       Sigma for the 3d smoothing kernel.

    normalize : boolean, default: True
       The power value for the normalization procedure.

    Returns
    ----------

    2d array, float
        Preprocessed fMRI data, voxel by TR.
    """
    # Handle exceptions in the values of the volume input
    if len(volume.shape) != 2:
        logging.exception('Volume is only {} dimensions, requires 2 '
                          'dimensional data'.format(len(volume.shape)))
        quit()

    # Binarize the data
    if t_score is False:
        volume[abs(volume) > 0] = 1

    # Smooth the data using a given kernel
    if gauss_size > 0:
        volume = ndimage.filters.gaussian_filter(volume, gauss_size)

    # Normalize the data to a given power, 0 means nothing is changed
    if normalize is True:
        volume = (volume - np.mean(volume)) / np.std(volume)

    return(volume)


class SelectionFuncs:
    """Evaluate the voxels according to the given metric."""

    # Perform a one sample t test against zero for each voxels across time
    def ttest_score(volume):
        return stats.ttest_1samp(volume, 0, axis=1)[1]

    # Calculate the variance for each voxels across time
    def variance_score(volume):
        return np.var(volume, axis=1)


def _select_voxels(volume, voxel_number=1000,
                   selectionfunc=SelectionFuncs.ttest_score):
    """Select voxels that perform best according to some function

    Parameters
    ----------

    volume : 2d array, float
        fMRI data, voxel by TR.

    voxel_number : int, default: 1000
       How many voxels are you going to use.

    selectionfunc: Option[Callable[[ndarray], ndarray],
            SelectionFuncs.ttest_score]
       What function are you going to use to select the top voxels.

    Returns
    ----------

    Iterable[bool]
        The voxels that have been selected

    """
    # TODO: use more advanced voxel selection procedures
    # Reduce the number of voxels to be considered if it exceeds the limit
    if voxel_number > volume.shape[0]:
        voxel_number = volume.shape[0]

    # Run the one function that was test
    altered_voxel = selectionfunc(volume)

    # Only keep voxels over the threshold
    threshold = sorted(altered_voxel)[volume.shape[0] - voxel_number]

    # Which voxels are best, needed for later
    selected_voxels = altered_voxel >= threshold

    return selected_voxels

    logging.info('Voxel Selection complete')


def _mds_conversion(volume, selected_voxels, dist_metric, dimensions=2):
    """Select voxels that perform best according to some function

    Parameters
    ----------

    volume : 2d array, float
        fMRI data, voxel by TR.

    selected_voxels : int
       Which indexes, according to volume, are selected

    dist_metric: 2d array, float
       The distance matrix, the same size as len(selected_voxels) by len(
       selected_voxels)

    Returns
    ----------

    ndarray[float].shape(selected_voxels.shape(0),dimensions)
        The coordinates, listed as voxel by dimension, as the output of MDS
    """
    # Run classical MDS, project into dimensions

    import sklearn.manifold
    # Make the object for the mds
    mds = sklearn.manifold.MDS(n_components=dimensions)
    mds_coords = mds.fit(dist_metric)

    # Specify the coordinates
    selected_coordinates = np.empty((volume.shape[0], dimensions))  # Preset
    # Put the MDS coordinates where they are supposed to go
    selected_coordinates[selected_voxels, ] = mds_coords.embedding_

    return selected_coordinates


class DistanceFuncs:
    """Collection of distance functions"""
    # TODO: Get distance metrics from the Han lab
    # Calculate the correlation
    def compute_corr(cor_matrix):
        return cor_matrix

    # Calculate the euclidean distance between
    def compute_euclidean_distance(cor_matrix):
        return spatial.distance.squareform(spatial.distance.pdist(cor_matrix))

    # Take the inverse of the correlation matrix (kind of)
    def compute_inverse_corr_distance(cor_matrix):
        return 1 - cor_matrix

    # Take the inverse of the abs correlation matrix (kind of)
    def compute_inverse_abs_corr_distance(cor_matrix):
        return 1 - abs(cor_matrix)


def convert_space(volume, voxel_number=1000,
                  selectionfunc=SelectionFuncs.ttest_score,
                  distancefunc=DistanceFuncs.compute_corr,
                  run_mds=False, dimensions=2):
    """Correlate  voxels and  process the correlation matrix

    Parameters
    ----------

    volume : list of 4d array, float
        fMRI volumes, by TR.

    voxel_number : int, default: 1000
       How many voxels are you going to use

    selectionfunc : Option[Callable[[ndarray], ndarray],
            SelectionFuncs.ttest_score]
       What function are you going to use to select the top voxels

    run_mds : bool, default: False
        Lower the dimensionality of the correlation matrix

    dimensions : int, default: 2
        How many dimensions are you reducing the correlation matrix to

    distancefunc : Option[Callable[[ndarray], ndarray],
            DistanceFuncs.compute_euclidean_distance]
       What function are you going to use to convert the correlation matrix

    Returns
    ----------

    ndarray, float
        The coordinates, listed as voxel by dimension, as the output of MDS
    """
    # Handle exceptions in the values of the volume input
    if len(volume.shape) != 2:
        logging.exception('Volume is only {} dimensions, requires 2 '
                          'dimensional data'.format(len(volume.shape)))
        quit()

    selected_voxels = _select_voxels(volume, voxel_number, selectionfunc)

    # TODO: use fcma toolbox to calculate the correlation matrix
    cor_matrix = np.corrcoef(volume[selected_voxels, ])

    dist_metric = distancefunc(cor_matrix)

    if run_mds == 1:
        converted_space = _mds_conversion(volume, selected_voxels,
                                          dist_metric, dimensions)
    else:
        converted_space = dist_metric

    return converted_space
