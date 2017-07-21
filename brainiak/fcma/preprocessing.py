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
"""FCMA preprocessing."""

# Authors: Yida Wang
# (Intel Labs), 2017

import math
import time
import numpy as np
import logging
from scipy.stats.mstats import zscore
from mpi4py import MPI
from enum import Enum

from ..image import mask_images, multimask_images


logger = logging.getLogger(__name__)

__all__ = [
    "RandomType",
    "prepare_fcma_data",
    "generate_epochs_info",
    "prepare_mvpa_data",
    "prepare_searchlight_mvpa_data",
]


def _separate_epochs(activity_data, epoch_list):
    """ create data epoch by epoch

    Separate data into epochs of interest specified in epoch_list
    and z-score them for computing correlation

    Parameters
    ----------
    activity_data: list of 2D array in shape [nVoxels, nTRs]
        the masked activity data organized in voxel*TR formats of all subjects
    epoch_list: list of 3D array in shape [condition, nEpochs, nTRs]
        specification of epochs and conditions
        assuming all subjects have the same number of epochs
        len(epoch_list) equals the number of subjects

    Returns
    -------
    raw_data: list of 2D array in shape [epoch length, nVoxels]
        the data organized in epochs
        and z-scored in preparation of correlation computation
        len(raw_data) equals the number of epochs
    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs
    """
    time1 = time.time()
    raw_data = []
    labels = []
    for sid in range(len(epoch_list)):
        epoch = epoch_list[sid]
        for cond in range(epoch.shape[0]):
            sub_epoch = epoch[cond, :, :]
            for eid in range(epoch.shape[1]):
                r = np.sum(sub_epoch[eid, :])
                if r > 0:   # there is an epoch in this condition
                    # mat is row-major
                    # regardless of the order of acitvity_data[sid]
                    mat = activity_data[sid][:, sub_epoch[eid, :] == 1]
                    mat = np.ascontiguousarray(mat.T)
                    mat = zscore(mat, axis=0, ddof=0)
                    # if zscore fails (standard deviation is zero),
                    # set all values to be zero
                    mat = np.nan_to_num(mat)
                    mat = mat / math.sqrt(r)
                    raw_data.append(mat)
                    labels.append(cond)
    time2 = time.time()
    logger.debug(
        'epoch separation done, takes %.2f s' %
        (time2 - time1)
    )
    return raw_data, labels


def _randomize_single_subject(data, seed=None):
    """Randomly permute the voxels of the subject.

     The subject is organized as Voxel x TR,
     this method shuffles the voxel dimension in place.

    Parameters
    ----------
    data: 2D array in shape [nVoxels, nTRs]
        Activity data to be shuffled.
    seed: Optional[int]
        Seed for random state used implicitly for shuffling.

    Returns
    -------
    None.
    """
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(data)


def _randomize_subject_list(data_list, random):
    """Randomly permute the voxels of a subject list.

     The method shuffles the subject one by one in place according to
     the random type. If RandomType.NORANDOM, return the original list.

    Parameters
    ----------
    data_list: list of 2D array in shape [nVxels, nTRs]
        Activity data list to be shuffled.
    random: RandomType
        Randomization type.

    Returns
    -------
    None.
    """
    if random == RandomType.REPRODUCIBLE:
        for i in range(len(data_list)):
            _randomize_single_subject(data_list[i], seed=i)
    elif random == RandomType.UNREPRODUCIBLE:
        for data in data_list:
            _randomize_single_subject(data)


class RandomType(Enum):
    """Define the random types as enumeration

    NORANDOM means do not randomize the data;
    REPRODUCIBLE means randomize the data with a fixed seed so that the
    permutation holds between different runs;
    UNREPRODUCIBLE means truly randomize the data which returns different
    results in different runs.
    """
    NORANDOM = 0
    REPRODUCIBLE = 1
    UNREPRODUCIBLE = 2


def prepare_fcma_data(images, conditions, mask1, mask2=None,
                      random=RandomType.NORANDOM, comm=MPI.COMM_WORLD):
    """Prepare data for correlation-based computation and analysis.

    Generate epochs of interests, then broadcast to all workers.

    Parameters
    ----------
    images: Iterable[SpatialImage]
        Data.
    conditions: List[UniqueLabelConditionSpec]
        Condition specification.
    mask1: np.ndarray
        Mask to apply to each image.
    mask2: Optional[np.ndarray]
        Mask to apply to each image.
        If it is not specified, the method will assign None to the returning
        variable raw_data2 and the self-correlation on raw_data1 will be
        computed
    random: Optional[RandomType]
        Randomize the data within subject or not.
        Default NORANDOM
    comm: MPI.Comm
        MPI communicator to use for MPI operations.

    Returns
    -------
    raw_data1: list of 2D array in shape [epoch length, nVoxels]
        the data organized in epochs, specified by the first mask.
        len(raw_data) equals the number of epochs
    raw_data2: Optional, list of 2D array in shape [epoch length, nVoxels]
        the data organized in epochs, specified by the second mask if any.
        len(raw_data2) equals the number of epochs
    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs
    """
    rank = comm.Get_rank()
    labels = []
    raw_data1 = []
    raw_data2 = []
    if rank == 0:
        logger.info('start to apply masks and separate epochs')
        if mask2 is not None:
            masks = (mask1, mask2)
            activity_data1, activity_data2 = zip(*multimask_images(images,
                                                                   masks,
                                                                   np.float32))
            _randomize_subject_list(activity_data2, random)
            raw_data2, _ = _separate_epochs(activity_data2, conditions)
        else:
            activity_data1 = list(mask_images(images, mask1, np.float32))
        _randomize_subject_list(activity_data1, random)
        raw_data1, labels = _separate_epochs(activity_data1, conditions)
        time1 = time.time()
    raw_data_length = len(raw_data1)
    raw_data_length = comm.bcast(raw_data_length)
    # broadcast the data subject by subject to prevent size overflow
    for i in range(raw_data_length):
        if rank != 0:
            raw_data1.append(None)
            if mask2 is not None:
                raw_data2.append(None)
        raw_data1[i] = comm.bcast(raw_data1[i], root=0)
        if mask2 is not None:
            raw_data2[i] = comm.bcast(raw_data2[i], root=0)

    if comm.Get_size() > 1:
        labels = comm.bcast(labels, root=0)
        if rank == 0:
            time2 = time.time()
            logger.info(
                'data broadcasting done, takes %.2f s' %
                (time2 - time1)
            )
    if mask2 is None:
        raw_data2 = None
    return raw_data1, raw_data2, labels


def generate_epochs_info(epoch_list):
    """ use epoch_list to generate epoch_info defined below

    Parameters
    ----------
    epoch\_list: list of 3D (binary) array in shape [condition, nEpochs, nTRs]
        Contains specification of epochs and conditions, assuming
        1. all subjects have the same number of epochs;
        2. len(epoch_list) equals the number of subjects;
        3. an epoch is always a continuous time course.

    Returns
    -------
    epoch_info: list of tuple (label, sid, start, end).
        label is the condition labels of the epochs;
        sid is the subject id, corresponding to the index of raw_data;
        start is the start TR of an epoch (inclusive);
        end is the end TR of an epoch(exclusive).
        Assuming len(labels) labels equals the number of epochs and
        the epochs of the same sid are adjacent in epoch_info
    """
    time1 = time.time()
    epoch_info = []
    for sid, epoch in enumerate(epoch_list):
        for cond in range(epoch.shape[0]):
            sub_epoch = epoch[cond, :, :]
            for eid in range(epoch.shape[1]):
                r = np.sum(sub_epoch[eid, :])
                if r > 0:   # there is an epoch in this condition
                    start = np.nonzero(sub_epoch[eid, :])[0][0]
                    epoch_info.append((cond, sid, start, start + r))
    time2 = time.time()
    logger.debug(
        'epoch separation done, takes %.2f s' %
        (time2 - time1)
    )
    return epoch_info


def prepare_mvpa_data(images, conditions, mask):
    """Prepare data for activity-based model training and prediction.

    Average the activity within epochs and z-scoring within subject.

    Parameters
    ----------
    images: Iterable[SpatialImage]
        Data.
    conditions: List[UniqueLabelConditionSpec]
        Condition specification.
    mask: np.ndarray
        Mask to apply to each image.

    Returns
    -------
    processed_data: 2D array in shape [num_voxels, num_epochs]
        averaged epoch by epoch processed data
    labels: 1D array
        contains labels of the data
    """
    activity_data = list(mask_images(images, mask, np.float32))
    epoch_info = generate_epochs_info(conditions)
    num_epochs = len(epoch_info)
    (d1, _) = activity_data[0].shape
    processed_data = np.empty([d1, num_epochs])
    labels = np.empty(num_epochs)
    subject_count = [0]  # counting the epochs per subject for z-scoring
    cur_sid = -1
    # averaging
    for idx, epoch in enumerate(epoch_info):
        labels[idx] = epoch[0]
        if cur_sid != epoch[1]:
            subject_count.append(0)
            cur_sid = epoch[1]
        subject_count[-1] += 1
        processed_data[:, idx] = \
            np.mean(activity_data[cur_sid][:, epoch[2]:epoch[3]],
                    axis=1)
    # z-scoring
    cur_epoch = 0
    for i in subject_count:
        if i > 1:
            processed_data[:, cur_epoch:cur_epoch + i] = \
                zscore(processed_data[:, cur_epoch:cur_epoch + i],
                       axis=1, ddof=0)
        cur_epoch += i
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    processed_data = np.nan_to_num(processed_data)

    return processed_data, labels


def prepare_searchlight_mvpa_data(images, conditions, data_type=np.float32,
                                  random=RandomType.NORANDOM):
    """ obtain the data for activity-based voxel selection using Searchlight

    Average the activity within epochs and z-scoring within subject,
    while maintaining the 3D brain structure. In order to save memory,
    the data is processed subject by subject instead of reading all in before
    processing. Assuming all subjects live in the identical cube.

    Parameters
    ----------
    images: Iterable[SpatialImage]
        Data.
    conditions: List[UniqueLabelConditionSpec]
        Condition specification.
    data_type
        Type to cast image to.
    random: Optional[RandomType]
        Randomize the data within subject or not.

    Returns
    -------
    processed_data: 4D array in shape [brain 3D + epoch]
        averaged epoch by epoch processed data

    labels: 1D array
        contains labels of the data
    """
    time1 = time.time()
    epoch_info = generate_epochs_info(conditions)
    num_epochs = len(epoch_info)
    processed_data = None
    logger.info(
        'there are %d subjects, and in total %d epochs' %
        (len(conditions), num_epochs)
    )
    labels = np.empty(num_epochs)
    # assign labels
    for idx, epoch in enumerate(epoch_info):
        labels[idx] = epoch[0]
    # counting the epochs per subject for z-scoring
    subject_count = np.zeros(len(conditions), dtype=np.int32)

    logger.info('start to apply masks and separate epochs')
    for sid, f in enumerate(images):
        data = f.get_data().astype(data_type)
        [d1, d2, d3, d4] = data.shape
        if random == RandomType.REPRODUCIBLE:
            data = data.reshape((d1 * d2 * d3, d4))
            _randomize_single_subject(data, seed=sid)
            data = data.reshape((d1, d2, d3, d4))
        elif random == RandomType.UNREPRODUCIBLE:
            data = data.reshape((d1 * d2 * d3, d4))
            _randomize_single_subject(data)
            data = data.reshape((d1, d2, d3, d4))
        if processed_data is None:
            processed_data = np.empty([d1, d2, d3, num_epochs],
                                      dtype=data_type)
        # averaging
        for idx, epoch in enumerate(epoch_info):
            if sid == epoch[1]:
                subject_count[sid] += 1
                processed_data[:, :, :, idx] = \
                    np.mean(data[:, :, :, epoch[2]:epoch[3]], axis=3)

        logger.debug(
            'file %s is loaded and processed, with data shape %s',
            f.get_filename(), data.shape
        )
    # z-scoring
    cur_epoch = 0
    for i in subject_count:
        if i > 1:
            processed_data[:, :, :, cur_epoch:cur_epoch + i] = \
                zscore(processed_data[:, :, :, cur_epoch:cur_epoch + i],
                       axis=3, ddof=0)
        cur_epoch += i
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    processed_data = np.nan_to_num(processed_data)
    time2 = time.time()
    logger.info(
        'data processed for activity-based voxel selection, takes %.2f s' %
        (time2 - time1)
    )

    return processed_data, labels
