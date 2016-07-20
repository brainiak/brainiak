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

Takes in a time series, ignoring the conditions. It does some voxel selection on this time series.
It then runs a correlation on all of these selected voxels.
Now each voxel can be represented as existing in a higher dimensional space now

volume is organized as an Voxel x Timepoint matrix
voxel_number describes how many voxels are to be used in the correlation
selection contains the procedure for selecting voxels: Ttest, Variance
distance is the procedure for calculating the distance matrix: Dist, InverseCor, InverseAbsCor or none

 Authors: Cameron Ellis (Princeton) 2016
"""
import logging

import numpy as np
import scipy.ndimage as ndimage
from scipy import spatial
from scipy import stats

__all__ = [
    "preprocess",
    "convert_space",
]

logger = logging.getLogger(__name__)

def preprocess(volume, t_score=1, gauss_size=0, norm_power=0):
    """Preprocess the data for TDA relevant features

    Takes in a specified volume (3d or 4d) for a participant's block,
    determines whether it should be binary,
    applies smoothing to the volume and returns the result
    TODO: fix to be like tfa
    """

    #TODO: exception flag

    # Binarize the data
    if t_score == 0:
        volume[abs(volume) > 0] = 1

    # Smooth the data using a given kernel
    if gauss_size > 0:
        volume = ndimage.filters.gaussian_filter(volume, gauss_size)

    # Normalize the data to a given power, 0 means nothing is changed
    if norm_power > 0:
        volume = volume / np.power(np.sum(np.power(volume, norm_power)), 1 / norm_power)

    return(volume)

def _ttest_score(volume):
    #TODO: Doc string
    altered_voxel = abs(stats.ttest_1samp(volume, 0, axis=1)[1])
    return altered_voxel

def _variance_score(volume):
    # TODO: Doc string
    altered_voxel = np.var(volume, axis=1)
    return altered_voxel


def _select_voxels(volume, voxel_number=100, selectionfunc=_ttest_score):
    # TODO: Doc string
    #Reduce the number of voxels to be considered if it exceeds the limit
    if voxel_number>volume.shape[0]:
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
    # TODO: Doc string
    # Run classical MDS, project into dimensions

    import sklearn.manifold
    # Make the object for the mds
    mds = sklearn.manifold.MDS(n_components=dimensions)
    mds_coords = mds.fit(dist_metric)

    # Specify the coordinates
    selected_coordinates = np.empty((volume.shape[0], dimensions))  # Preset
    selected_coordinates[selected_voxels,] = mds_coords.embedding_  # Put the MDS coordinates where they are supposed to go

    return selected_coordinates

#Define how to transform the neural data, if at all

#Calculate the euclidean distance between
_compute_euclidean_distance = \
    lambda x: spatial.distance.squareform(spatial.distance.pdist(x))

#TODO: lines
_compute_inverse_corr_distance = lambda x: 1 - x #Take the inverse of the correlation matrix (kind of)
_compute_inverse_abs_corr_distance = lambda x: 1 - abs(x) #Take the inverse of the abs correlation matrix (kind of)


#TODO: Fix, put below func name
""" Correlate all functions with all other functions

Run a correlation of every voxel against every other voxel

Takes in a time series, ignoring the conditions. It does some voxel selection on this time series.
It then runs a correlation on all of these selected voxels.
Now each voxel can be represented as existing in a higher dimensional space now

volume is organized as an Voxel x Timepoint matrix
voxel_number describes how many voxels are to be used in the correlation
selection contains the procedure for selecting voxels: Ttest, Variance
distance is the procedure for calculating the distance matrix: Dist, InverseCor, InverseAbsCor or none

 Authors: Cameron Ellis (Princeton) 2016

"""

def convert_space(volume, voxel_number=100, selectionfunc=_ttest_score, distancefunc=_compute_euclidean_distance):

    #TODO: use more advanced voxel selection procedures
    selected_voxels = _select_voxels(volume, voxel_number=voxel_number, selectionfunc=selectionfunc)

    #TODO: use fcma toolbox to calculate the correlation matrix
    cor_matrix = np.corrcoef(volume[selected_voxels,])

    #TODO: Use the distance metrics from the Han lab to calculate more interesting distrance functions
    dist_metric = _compute_euclidean_distance(cor_matrix)
    selected_coordinates = _mds_conversion(volume, selected_voxels, dist_metric)

    return selected_coordinates
