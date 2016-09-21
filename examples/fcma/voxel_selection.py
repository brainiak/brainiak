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

from brainiak.fcma.voxelselector import VoxelSelector
from scipy.stats.mstats import zscore
from sklearn import svm
import sys
from mpi4py import MPI
import nibabel as nib
import os
import math
import time
import numpy as np
import logging

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

def readActivityData(dir, file_extension, mask_file):
    """ read data in NIfTI format and apply the spatial mask to them

    Parameters
    ----------
    dir: str
        the path to all subject files
    file_extension: str
        the file extension, usually nii.gz or nii
    mask_file: str
        the absolute path of the mask file, we apply the mask right after
        reading a file for saving memory

    Returns
    -------
    activity_data:  list of 2D array in shape [nTRs, nVoxels]
        the masked activity data organized in TR*voxel formats
        len(activity_data) equals the number of subjects
    """
    time1 = time.time()
    mask_img = nib.load(mask_file)
    mask = mask_img.get_data()
    count = 0
    for index in np.ndindex(mask.shape):
        if mask[index] != 0:
            count += 1
    files = [f for f in sorted(os.listdir(dir))
             if os.path.isfile(os.path.join(dir, f))
             and f.endswith(file_extension)]
    activity_data = []
    for f in files:
        img = nib.load(os.path.join(dir, f))
        data = img.get_data()
        (d1, d2, d3, d4) = data.shape
        masked_data = np.zeros([d4, count], np.float32, order='C')
        count1 = 0
        for index in np.ndindex(mask.shape):
            if mask[index] != 0:
                masked_data[:, count1] = np.copy(data[index])
                count1 += 1
        activity_data.append(masked_data)
        logger.info(
            'file %s is loaded and masked, with data shape %s' %
            (f, masked_data.shape)
        )
    time2 = time.time()
    logger.info(
        'data reading done, takes %.2f s' %
        (time2 - time1)
    )
    return activity_data


def separateEpochs(activity_data, epoch_list):
    """ separate data into epochs of interest specified in epoch_list
    and z-score them for computing correlation

    Parameters
    ----------
    activity_data: list of 2D array in shape [nTRs, nVoxels]
        the masked activity data organized in TR*voxel formats of all subjects
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
                    mat = activity_data[sid][sub_epoch[eid, :] == 1, :]
                    mat = zscore(mat, axis=0, ddof=0)
                    # if zscore fails (standard deviation is zero),
                    # set all values to be zero
                    mat = np.nan_to_num(mat)
                    mat = mat / math.sqrt(r)
                    raw_data.append(mat)
                    labels.append(cond)
    time2 = time.time()
    logger.info(
        'epoch separation done, takes %.2f s' %
        (time2 - time1)
    )
    return raw_data, labels


def prepareData(data_dir, extension, mask_file, epoch_file):
    """ read the data in and generate epochs of interests,
    then broadcast to all workers

    Parameters
    ----------
    data_dir: str
        the path to all subject files
    extension: str
        the file extension, usually nii.gz or nii
    mask_file: str
        the absolute path of the mask file,
        we apply the mask right after reading a file for saving memory
    epoch_file: str
        the absolute path of the epoch file

    Returns
    -------
    raw_data: list of 2D array in shape [epoch length, nVoxels]
        the data organized in epochs
        len(raw_data) equals the number of epochs
    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    labels = []
    raw_data = []
    if rank == 0:
        activity_data = readActivityData(data_dir, extension, mask_file)
        # a list of numpy array in shape [condition, nEpochs, nTRs]
        epoch_list = np.load(epoch_file)
        raw_data, labels = separateEpochs(activity_data, epoch_list)
        time1 = time.time()
    raw_data_length = len(raw_data)
    raw_data_length = comm.bcast(raw_data_length, root=0)
    # broadcast the data subject by subject to prevent size overflow
    for i in range(raw_data_length):
        if rank != 0:
            raw_data.append(None)
        raw_data[i] = comm.bcast(raw_data[i], root=0)
    labels = comm.bcast(labels, root=0)
    if rank == 0:
        time2 = time.time()
        logger.info(
            'data broadcasting done, takes %.2f s' %
            (time2 - time1)
        )
    return raw_data, labels


"""
example running command:
mpirun -np 2 python voxel_selection.py /Users/yidawang/data/face_scene/raw nii.gz /Users/yidawang/data/face_scene/mask.nii.gz
                        data/fs_epoch_labels.npy 12 18
"""
if __name__ == '__main__':
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'programming starts in %d process(es)' %
            MPI.COMM_WORLD.Get_size()
        )
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]
    raw_data, labels = prepareData(data_dir, extension, mask_file, epoch_file)
    epochs_per_subj = int(sys.argv[5])
    num_subjs = int(sys.argv[6])
    vs = VoxelSelector(raw_data, epochs_per_subj, labels, num_subjs)
    # for cross validation, use SVM with precomputed kernel
    # no shrinking, set C=10
    clf = svm.SVC(kernel='precomputed', shrinking=False, C=10)
    results = vs.run(clf)
    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        print(results[0:100])
