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
import time
from mpi4py import MPI
from scipy.stats.mstats import zscore
from sklearn import model_selection
import sklearn
from . import fcma_extension
from . import cython_blas as blas
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "VoxelSelector",
]


class VoxelSelector:
    """Correlation-based voxel selection component of FCMA

    Parameters
    ----------

    raw_data: list of 2D array in shape [epoch length, nVoxels]
        Assumption: 1. all activity data contains the same number of voxels
                    2. the activity data has been z-scored,
                       ready to compute correlation as matrix multiplication
                    3. all subjects have the same number of epochs
                    4. epochs belonging to the same subject are adjacent
                       in the list
                    5. voxel selection is always done in the auto-correlation,
                       i.e. raw_data correlate with themselves
                    6. if MPI jobs are running on multiple nodes, the user
                       home directory is shared by all nodes

    epochs_per_subj: int
        The number of epochs of each subject

    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs

    num_folds: int
        The number of folds to be conducted in the cross validation

    voxel_unit: int, default 100
        The number of voxels assigned to a worker each time

    master_rank: int, default 0
        The process which serves as the master
    """
    def __init__(self,
                 raw_data,
                 epochs_per_subj,
                 labels,
                 num_folds,
                 voxel_unit=100,
                 master_rank=0):
        self.raw_data = raw_data
        self.epochs_per_subj = epochs_per_subj
        self.num_voxels = raw_data[0].shape[1]
        self.labels = labels
        self.num_folds = num_folds
        self.voxel_unit = voxel_unit
        self.master_rank = master_rank
        if self.num_voxels == 0:
            raise ValueError('Zero processed voxels')
        if MPI.COMM_WORLD.Get_size() == 1:
            raise RuntimeError('one process cannot run the '
                               'master-worker model')
        if self.master_rank >= MPI.COMM_WORLD.Get_size():
            logger.warn('Master rank exceeds the number of '
                        'launched processes, set to 0')
            self.master_rank = 0

    # tags for MPI messages
    _WORKTAG = 0
    _TERMINATETAG = 1

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
        if rank == self.master_rank:
            results = self._master()
            # Sort the voxels
            results.sort(key=lambda tup: tup[1], reverse=True)
        else:
            self._worker(clf)
            results = []
        return results

    def _master(self):
        """ master node's operation

        Assigning tasks to workers and collecting results from them

        Parameters
        ----------
        None

        Returns
        -------
        results: list of tuple (voxel_id, accuracy)
            the accuracy numbers of all voxels, in accuracy descending order
            the length of array equals the number of voxels
        """
        logger.info(
            'Master starts to allocate tasks'
        )
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
        for i in range(0, size):
            if i == self.master_rank:
                continue
            if current_task[1] == 0:
                using_size = i
                break
            comm.send(current_task,
                      dest=i,
                      tag=self._WORKTAG)
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
            comm.send(current_task,
                      dest=status.Get_source(),
                      tag=self._WORKTAG)
            next_start = current_task[0] + current_task[1]
            sending_voxels = self.voxel_unit \
                if self.voxel_unit < self.num_voxels - next_start \
                else self.num_voxels - next_start
            current_task = (next_start, sending_voxels)

        for i in range(0, using_size):
            if i == self.master_rank:
                continue
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            results += result

        for i in range(0, size):
            if i == self.master_rank:
                continue
            comm.send(None,
                      dest=i,
                      tag=self._TERMINATETAG)

        return results

    def _worker(self, clf):
        """ worker node's operation

        Receiving tasks from the master to process and sending the result back

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
            task = comm.recv(source=self.master_rank,
                             tag=MPI.ANY_TAG,
                             status=status)
            if status.Get_tag():
                break
            comm.send(self._voxelScoring(task, clf),
                      dest=self.master_rank)

    def _correlationComputation(self, task):
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
            corr[i, e, s + j] = corr[j, e, s + i]
        """
        time1 = time.time()
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
        time2 = time.time()
        logger.debug(
            'correlation computation for %d voxels, takes %.2f s' %
            (task[1], (time2 - time1))
        )
        return corr

    def _correlationNormalization(self, corr):
        """ within-subject normalization

        This method uses scipy.zscore to normalize the data,
        but is much slower than its C++ counterpart。
        It is doing in-place z-score.

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
        time1 = time.time()
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
        time2 = time.time()
        logger.debug(
            'within-subject normalization for %d voxels '
            'using numpy zscore function, takes %.2f s' %
            (sv, (time2 - time1))
        )
        return corr

    def _crossValidation(self, task, corr, clf):
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
        time1 = time.time()
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
            skf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                  shuffle=False)
            scores = model_selection.cross_val_score(clf, data,
                                                     y=self.labels,
                                                     cv=skf, n_jobs=1)
            results.append((i + task[0], scores.mean()))
        time2 = time.time()
        logger.debug(
            'cross validation for %d voxels, takes %.2f s' %
            (task[1], (time2 - time1))
        )
        return results

    def _voxelScoring(self, task, clf):
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
        corr = self._correlationComputation(task)
        # normalization
        # corr = self._correlationNormalization(corr)
        time3 = time.time()
        fcma_extension.normalization(corr, self.epochs_per_subj)
        time4 = time.time()
        logger.debug(
            'within-subject normalization for %d voxels '
            'using C++, takes %.2f s' %
            (task[1], (time4 - time3))
        )

        # cross validation
        results = self._crossValidation(task, corr, clf)
        time2 = time.time()
        logger.info(
            'in rank %d, task %d takes %.2f s' %
            (MPI.COMM_WORLD.Get_rank(),
             (int(task[0] / self.voxel_unit)), (time2 - time1))
        )
        return results
