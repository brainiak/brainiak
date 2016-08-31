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

This implementation is based on the following publications:

.. [Wang2015-1] Full correlation matrix analysis (FCMA): An unbiased method for
   task-related functional connectivity",
   Yida Wang, Jonathan D Cohen, Kai Li, Nicholas B Turk-Browne.
   Journal of Neuroscience Methods, 2015.

.. [Wang2015-2] "Full correlation matrix analysis of fMRI data on Intel® Xeon
   Phi™ coprocessors",
   Yida Wang, Michael J. Anderson, Jonathan D. Cohen, Alexander Heinecke,
   Kai Li, Nadathur Satish, Narayanan Sundaram, Nicholas B. Turk-Browne,
   Theodore L. Willke.
   In Proceedings of the International Conference for
   High Performance Computing,
   Networking, Storage and Analysis. 2015.
"""

# Authors: Yida Wang
# (Intel Labs), 2016

import numpy as np
import nibabel as nib
import os
import math
import sys
import time
from mpi4py import MPI
from scipy.stats.mstats import zscore
from sklearn import cross_validation
import sklearn
from . import fcma_extension
from . import cython_blas as blas

WORKTAG = 0
TERMINATETAG = 1


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
        print(f, masked_data.shape)
        sys.stdout.flush()
    time2 = time.time()
    print('data reading done, takes', round(time2 - time1, 2), 's')
    sys.stdout.flush()
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
    print('epoch separation done, takes', round(time2 - time1, 2), 's')
    sys.stdout.flush()
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
        print('data broadcasting done, takes', round(time2 - time1, 2), 's')
    return raw_data, labels


class VoxelSelector:
    """Correlation-based voxel selection component of FCMA

    Parameters
    ----------

    raw_data: list of 2D array in shape [epoch length, nVoxels]
        Assumption: 1. all activity data contains the same number of voxels
                    2. the activity data has been z-scored,
                       ready to compute correlation as matrix multiplication
                    3. all subjects have the same number of epochs
                    4. voxel selection is always done in the auto-correlation,
                       i.e. raw_data correlate with themselves

    epochs_per_subj: int
        The number of epochs of each subject

    num_voxels: int
        The number of voxels participating in the voxel selection

    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs

    num_folds: int
        The number of folds to be conducted in the cross validation

    voxel_unit: int, default 100
        The number of voxel assigned to a worker each time
    """
    def __init__(self,
                 raw_data,
                 epochs_per_subj,
                 labels,
                 num_folds,
                 voxel_unit=100):
        self.raw_data = raw_data
        self.epochs_per_subj = epochs_per_subj
        self.num_voxels = raw_data[0].shape[1]
        self.labels = labels
        self.num_folds = num_folds
        self.voxel_unit = voxel_unit
        if self.num_voxels == 0:
            raise RuntimeError("Zero processed voxels")

    def run(self, clf):
        """ run correlation-based voxel selection in master-worker model

        Sort the voxels based on the cross-validation accuracy
        of their correlation vectors

        Parameters
        ----------
        clf: classification function
            the classifier to be used in cross validation

        Returns
        -------
        results: list of tuple (voxel_id, accuracy)
            the accuracy numbers of all voxels, in accuracy descending order
            the length of array equals the number of voxels
        """
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            results = self.master()
            # Sort the voxels
            results.sort(key=lambda tup: tup[1], reverse=True)
        else:
            self.worker(clf)
            results = []
        return results

    def master(self):
        """ master node's operation, assigning tasks and collecting results

        Parameters
        ----------
        None

        Returns
        -------
        results: list of tuple (voxel_id, accuracy)
            the accuracy numbers of all voxels, in accuracy descending order
            the length of array equals the number of voxels
        """
        results = []
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        sending_voxels = self.voxel_unit if self.voxel_unit < self.num_voxels \
            else self.num_voxels
        current_task = (0, sending_voxels)
        status = MPI.Status()
        # using_size is used when the number of tasks
        # is smaller than the number of workers
        using_size = size
        for i in range(1, size):
            if current_task[1] == 0:
                using_size = i
                break
            comm.send(current_task, dest=i, tag=WORKTAG)
            next_start = current_task[0] + current_task[1]
            sending_voxels = self.voxel_unit \
                if self.voxel_unit < self.num_voxels - next_start \
                else self.num_voxels - next_start
            current_task = (next_start, sending_voxels)

        while using_size == size:
            if current_task[1] == 0:
                break
            result = comm.recv(source=MPI.ANY_SOURCE,
                               tag=MPI.ANY_TAG,
                               status=status)
            results += result
            comm.send(current_task, dest=status.Get_source(), tag=WORKTAG)
            next_start = current_task[0] + current_task[1]
            sending_voxels = self.voxel_unit \
                if self.voxel_unit < self.num_voxels - next_start \
                else self.num_voxels - next_start
            current_task = (next_start, sending_voxels)

        for i in range(1, using_size):
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            results += result

        for i in range(1, size):
            comm.send(None, dest=i, tag=TERMINATETAG)

        return results

    def worker(self, clf):
        """ worker node's operation,
        receiving tasks from the master to process and sending the result back

        Parameters
        ----------
        clf: classification function
            the classifier to be used in cross validation

        Returns
        -------
        None
        """
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        while 1:
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag():
                break
            comm.send(self.voxelScoring(task, clf), dest=0)

    def correlationComputation(self, task):
        """ use BLAS API to do correlation computation (matrix multiplication)

        Parameters
        ----------
        task: tuple (start_voxel_id, num_assigned_voxels)
            depicting the voxels assigned to compute

        Returns
        -------
        corr: 3D array in shape [num_selected_voxels, num_epochs, num_voxels]
            the correlation values of all subjects in all epochs
            for the assigned values, in row-major
        """
        s = task[0]
        nEpochs = len(self.raw_data)
        corr = np.zeros((task[1], nEpochs, self.num_voxels),
                        np.float32, order='C')
        count = 0
        for mat in self.raw_data:
            no_trans = 'N'
            trans = 'T'
            blas.compute_correlation(no_trans, trans,
                                     self.num_voxels, task[1],
                                     mat.shape[0], 1.0,
                                     mat, self.num_voxels,
                                     s, self.num_voxels,
                                     0.0, corr,
                                     self.num_voxels * nEpochs, count)
            count += 1
        return corr

    def correlationNormalization(self, corr):
        """ within-subject normalization

        this method uses scipy.zscore to normalize the data,
        but is much slower than its C++ counterpart

        Parameters
        ----------
        corr: 3D array in shape [num_selected_voxels, num_epochs, num_voxels]
            the correlation values of all subjects in all epochs
            for the assigned values, in row-major

        Returns
        -------
        corr: 3D array in  shape [num_selected_voxels, num_epochs, num_voxels]
            the normalized correlation values of all subjects in all epochs
            for the assigned values, in row-major
        """
        (sv, e, av) = corr.shape
        for i in range(sv):
            start = 0
            while start < e:
                cur_val = corr[i, start: start + self.epochs_per_subj, :]
                cur_val = .5 * np.log(cur_val + 1) / (1 - cur_val)
                corr[i, start: start + self.epochs_per_subj, :] = \
                    zscore(cur_val, axis=0, ddof=0)
                start += self.epochs_per_subj
        # if zscore fails (standard deviation is zero),
        # set all values to be zero
        corr = np.nan_to_num(corr)
        return corr

    def crossValidation(self, task, corr, clf):
        """ voxelwise cross validation based on correlation vectors

        Parameters
        ----------
        task: tuple (start_voxel_id, num_assigned_voxels)
            depicting the voxels assigned to compute
        corr: 3D array in shape [num_selected_voxels, num_epochs, num_voxels]
            the normalized correlation values of all subjects in all epochs
            for the assigned values, in row-major

        Returns
        -------
        results: list of tuple (voxel_id, accuracy)
            the accuracy numbers of all voxels, in accuracy descending order
            the length of array equals the number of assigned voxels
        """
        (sv, e, av) = corr.shape
        kernel_matrix = np.zeros((e, e), np.float32, order='C')
        results = []
        for i in range(sv):
            if isinstance(clf, sklearn.svm.SVC) \
                    and clf.kernel == 'precomputed':
                blas.compute_kernel_matrix('L', 'T',
                                           e, self.num_voxels,
                                           1.0, corr,
                                           i, self.num_voxels,
                                           0.0, kernel_matrix, e)
                data = kernel_matrix
            else:
                data = corr[i, :, :]
            # no shuffling in cv
            skf = cross_validation.StratifiedKFold(self.labels,
                                                   n_folds=self.num_folds,
                                                   shuffle=False)
            scores = cross_validation.cross_val_score(clf, data, self.labels,
                                                      cv=skf, n_jobs=1)
            results.append((i + task[0], scores.mean()))
        return results

    def voxelScoring(self, task, clf):
        """ voxel selection processing done in the worker node

        Take the task in,
        do analysis on voxels specified by the task (voxel id, num_voxels)
        It is a three-stage pipeline consisting of:
        1. correlation computation
        2. within-subject normalization
        3. voxelwise cross validaion

        Parameters
        ----------
        task: tuple (start_voxel_id, num_assigned_voxels),
            depicting the voxels assigned to compute
        clf: classification function
            the classifier to be used in cross validation

        Returns
        -------
        results: list of tuple (voxel_id, accuracy)
            the accuracy numbers of all voxels, in accuracy descending order
            the length of array equals the number of assigned voxels
        """
        time1 = time.time()
        # correlation computation
        # corr is a 3D array in row major,
        # in (selected_voxels, epochs, all_voxels) shape
        # corr[i, e, s + j] = corr[j, e, s + i]
        corr = self.correlationComputation(task)
        # normalization
        # in-place z-score, the result is still in corr
        # corr = self.correlationNormalization(corr)
        fcma_extension.normalization(corr, self.epochs_per_subj)

        # cross validation
        results = self.crossValidation(task, corr, clf)
        time2 = time.time()
        print('task:', int(task[0] / self.voxel_unit),
              round(time2 - time1, 2), 's')
        sys.stdout.flush()
        return results
