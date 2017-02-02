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
"""Full Correlation Matrix Analysis (FCMA)

FCMA related IO routines
"""

# Authors: Yida Wang
# (Intel Labs), 2017

import nibabel as nib
from nibabel.nifti1 import Nifti1Pair
import os
import math
import time
import numpy as np
import logging
from scipy.stats.mstats import zscore
from mpi4py import MPI

logger = logging.getLogger(__name__)


def _read_one_nifti_file(filename, mask1=None, mask2=None):
    """ read data from an NIfTI file, apply the masks if provided

    Parameters
    ----------
    filename: str
        the absolute path of the NIfTI file
    mask1: Optional[3D array]
        This is a 3D binary file indicating which voxels
        are selected (labeled 1).
        If it is not specified, the data will not be masked and remain in 4D
    mask2: Optional[3D array]
        This is a 3D binary file indicating which voxels
        are selected (labeled 1).
        It is only allowed when mask is not None.
        If it is not specified, the returning variable data2 is None

    Returns
    -------
    data1: 2D/4D array
        if masked, in shape [nVoxels, nTRs], organized in voxel*TR formats;
        if not masked, in shape [x, y, z, t] or [brain 3D, nTRs];
        len(activity_data) equals the number of subjects;
        the data type is float32.
    data2: Optional[2D array]
        It is None if mask2 is None;
        otherwise, in shape [nVoxels, nTRs], organized in voxel*TR formats.
        The data type is float32.

    """
    if mask1 is None and mask2 is not None:
        raise ValueError('the second mask only takes effect when the first '
                         'mask is specified')
    img = nib.load(filename)
    raw_data = img.get_data()
    data2 = None
    if mask1 is None:
        data1 = raw_data.astype(np.float32)
    else:
        data1 = raw_data[mask1].astype(np.float32)
        if mask2 is not None:
            data2 = raw_data[mask2].astype(np.float32)

    return data1, data2


def read_activity_data(dir, file_extension, mask_file1=None, mask_file2=None):
    """ read data in NIfTI from a dir and apply the spatial mask to them

    Parameters
    ----------
    dir: str
        the path to all subject files
    file_extension: str
        the file extension, usually nii.gz or nii
    mask_file1: Optional[str]
        the absolute path of the mask file, we apply the mask right after
        reading a file for saving memory
        if it is not specified, the data will not be masked and remain in 4D
    mask_file2: Optional[str]
        the absolute path of another mask file, we apply the mask right after
        reading a file for saving memory.
        This mask is only valid when mask_file is not None

    Returns
    -------
    activity_data1: list of array of brain activity data
        if masked, in shape [nVoxels, nTRs], organized in voxel*TR formats;
        if not masked, in shape [x, y, z, t] or [brain 3D, nTRs];
        len(activity_data) equals the number of subjects;
        the data type is float32.
    activity_data2: Optional[list of array of brain activity data]
        in shape [nVoxels, nTRs], organized in voxel*TR formats;
        it is None if mask_file2 is None;
        len(activity_data2) equals the number of subjects;
        the data type is float32.
    """
    time1 = time.time()
    mask1 = None
    if mask_file1:
        mask_img = nib.load(mask_file1)
        mask1 = mask_img.get_data().astype(np.bool)
        logger.info(
            'first mask size: %d' %
            np.sum(mask1)
        )
    mask2 = None
    if mask_file2:
        mask_img = nib.load(mask_file2)
        mask2 = mask_img.get_data().astype(np.bool)
        logger.info(
            'second mask size: %d' %
            np.sum(mask2)
        )
    files = [f for f in sorted(os.listdir(dir))
             if os.path.isfile(os.path.join(dir, f))
             and f.endswith(file_extension)]
    activity_data1 = []
    activity_data2 = []
    for f in files:
        filename = os.path.join(dir, f)
        masked_data, masked_data2 = _read_one_nifti_file(filename,
                                                         mask1, mask2)
        activity_data1.append(masked_data)
        if mask1 is not None:
            activity_data2.append(masked_data2)
        logger.info(
            'file %s is loaded and masked.' % f
        )
    time2 = time.time()
    logger.info(
        'data reading done, takes %.2f s' %
        (time2 - time1)
    )
    if mask1 is None:
        activity_data2 = None
    return activity_data1, activity_data2


def _separate_epochs(activity_data, epoch_list):
    """ create data epoch by epoch

    Separate data into epochs of interest specified in epoch_list
    and z-score them for computing correlation

    Parameters
    ----------
    activity\_data: list of 2D array in shape [nVoxels, nTRs]
        the masked activity data organized in voxel*TR formats of all subjects
    epoch\_list: list of 3D array in shape [condition, nEpochs, nTRs]
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


def prepare_fcma_data(data_dir, extension, epoch_file, mask_file1,
                      mask_file2=None):
    """ obtain the data for correlation-based computation and analysis

    read the data in and generate epochs of interests,
    then broadcast to all workers

    Parameters
    ----------
    data_dir: str
        the path to all subject files
    extension: str
        the file extension, usually nii.gz or nii
    epoch_file: str
        the absolute path of the epoch file
    mask_file1: str
        the absolute path of the mask file,
        we apply the mask right after reading a file for saving memory.
    mask_file2: Optional[str]
        the absolute path of another mask file,
        we apply the mask right after reading a file for saving memory.
        If it is not specified, the method will assign None to
        the returning variable raw_data2

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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    labels = []
    raw_data1 = []
    raw_data2 = []
    if rank == 0:
        activity_data1, activity_data2 = read_activity_data(data_dir,
                                                            extension,
                                                            mask_file1,
                                                            mask_file2)
        # a list of numpy array in shape [condition, nEpochs, nTRs]
        epoch_list = np.load(epoch_file)
        raw_data1, labels = _separate_epochs(activity_data1, epoch_list)
        if mask_file2 is not None:
            raw_data2, _ = _separate_epochs(activity_data2, epoch_list)
        time1 = time.time()
    raw_data_length = len(raw_data1)
    raw_data_length = comm.bcast(raw_data_length, root=0)
    # broadcast the data subject by subject to prevent size overflow
    for i in range(raw_data_length):
        if rank != 0:
            raw_data1.append(None)
            if mask_file2 is not None:
                raw_data2.append(None)
        raw_data1[i] = comm.bcast(raw_data1[i], root=0)
        if mask_file2 is not None:
            raw_data2[i] = comm.bcast(raw_data2[i], root=0)

    if comm.Get_size() > 1:
        labels = comm.bcast(labels, root=0)
    if comm.Get_size() > 1 and rank == 0:
        time2 = time.time()
        logger.info(
            'data broadcasting done, takes %.2f s' %
            (time2 - time1)
        )
    if mask_file2 is None:
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
    epoch\_info: list of tuple (label, sid, start, end).
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
                    epoch_info.append((cond, sid, start, start+r))
    time2 = time.time()
    logger.debug(
        'epoch separation done, takes %.2f s' %
        (time2 - time1)
    )
    return epoch_info


def prepare_mvpa_data(data_dir, extension, epoch_file, mask_file):
    """ obtain the data for activity-based model training and prediction

    Average the activity within epochs and z-scoring within subject.

    Parameters
    ----------
    data_dir: str
        the path to all subject files
    extension: str
        the file extension, usually nii.gz or nii
    epoch_file: str
        the absolute path of the epoch file
    mask_file: str
        the absolute path of the mask file,
        we apply the mask right after reading a file for saving memory

    Returns
    -------
    processed\_data: 2D array in shape [num_voxels, num_epochs]
        averaged epoch by epoch processed data

    labels: 1D array
        contains labels of the data
    """
    activity_data, _ = read_activity_data(data_dir, extension, mask_file)
    epoch_list = np.load(epoch_file)
    epoch_info = generate_epochs_info(epoch_list)
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


def prepare_searchlight_mvpa_data(data_dir, extension, epoch_file):
    """ obtain the data for activity-based voxel selection using Searchlight

    Average the activity within epochs and z-scoring within subject,
    while maintaining the 3D brain structure. In order to save memory,
    the data is processed subject by subject instead of reading all in before
    processing. Assuming all subjects live in the identical cube.

    Parameters
    ----------
    data_dir: str
        the path to all subject files
    extension: str
        the file extension, usually nii.gz or nii
    epoch_file: str
        the absolute path of the epoch file

    Returns
    -------
    processed\_data: 4D array in shape [brain 3D + epoch]
        averaged epoch by epoch processed data

    labels: 1D array
        contains labels of the data
    """
    time1 = time.time()
    epoch_list = np.load(epoch_file)
    epoch_info = generate_epochs_info(epoch_list)
    num_epochs = len(epoch_info)
    processed_data = None
    files = [f for f in sorted(os.listdir(data_dir))
             if os.path.isfile(os.path.join(data_dir, f))
             and f.endswith(extension)]
    logger.info(
        'there are %d subjects, and in total %d epochs' %
        (len(files), num_epochs)
    )
    labels = np.empty(num_epochs)
    # assign labels
    for idx, epoch in enumerate(epoch_info):
        labels[idx] = epoch[0]
    # counting the epochs per subject for z-scoring
    subject_count = np.zeros(len(files), dtype=np.int32)

    for sid, f in enumerate(files):
        filename = os.path.join(data_dir, f)
        data, _ = _read_one_nifti_file(filename)
        if processed_data is None:
            [d1, d2, d3, _] = data.shape
            processed_data = np.empty([d1, d2, d3, num_epochs],
                                      dtype=np.float32)
        # averaging
        for idx, epoch in enumerate(epoch_info):
            if sid == epoch[1]:
                subject_count[sid] += 1
                processed_data[:, :, :, idx] = \
                    np.mean(data[:, :, :, epoch[2]:epoch[3]], axis=3)

        logger.info(
            'file %s is loaded and processed, with data shape %s' %
            (f, data.shape)
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


def write_nifti_file(data, affine, filename):
    """ write a nifti file given data and affine

    Parameters
    ----------
    data: 3D/4D numpy array
        the brain data with/without time dimension
    affine: 2D numpy array
        affine of the image, usually inherited from an existing image
    filename: string
        the output filename
    """
    img = Nifti1Pair(data, affine)
    nib.nifti1.save(img, filename)
