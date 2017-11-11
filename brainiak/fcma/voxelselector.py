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

Correlation-based voxel selection
"""

# Authors: Yida Wang
# (Intel Labs), 2016

import numpy as np
import time
from mpi4py import MPI
from scipy.stats.mstats import zscore
from sklearn import model_selection
import sklearn
from . import fcma_extension  # type: ignore
from . import cython_blas as blas  # type: ignore
from ..utils.utils import usable_cpu_count
import logging
import multiprocessing

logger = logging.getLogger(__name__)

__all__ = [
    "VoxelSelector",
]


def _cross_validation_for_one_voxel(clf, vid, num_folds, subject_data, labels):
    """Score classifier on data using cross validation."""
    # no shuffling in cv
    skf = model_selection.StratifiedKFold(n_splits=num_folds,
                                          shuffle=False)
    scores = model_selection.cross_val_score(clf, subject_data,
                                             y=labels,
                                             cv=skf, n_jobs=1)
    logger.debug(
        'cross validation for voxel %d is done' %
        vid
    )
    return (vid, scores.mean())


class VoxelSelector:
    """Correlation-based voxel selection component of FCMA.

    Parameters
    ----------

    labels: list of 1D array
        the condition labels of the epochs
        len(labels) labels equals the number of epochs

    epochs_per_subj: int
        The number of epochs of each subject

    num_folds: int
        The number of folds to be conducted in the cross validation

    raw_data: list of 2D array in shape [epoch length, nVoxels]
        Assumption: 1. all activity data contains the same number of voxels
                    2. the activity data has been z-scored,
                       ready to compute correlation as matrix multiplication
                    3. all subjects have the same number of epochs
                    4. epochs belonging to the same subject are adjacent
                       in the list
                    5. if MPI jobs are running on multiple nodes, the path
                       used must be on a filesystem shared by all nodes

    raw_data2: Optional, list of 2D array in shape [epoch length, nVoxels]
        raw_data2 shares the data structure of the assumptions of raw_data
        If raw_data2 is None, the correlation will be computed as
        raw_data by raw_data.
        If raw_data2 is specified, len(raw_data) MUST equal len(raw_data2),
        the correlation will be computed as raw_data by raw_data2.

    voxel_unit: int, default 64
        The number of voxels assigned to a worker each time

    process_num: Optional[int]
        The maximum number of processes used in cross validation.
        If None, the number of processes will equal
        the number of available hardware threads, considering cpusets
        restrictions.
        If 0, cross validation will not use python multiprocessing.

    master_rank: int, default 0
        The process which serves as the master
    """
    def __init__(self,
                 labels,
                 epochs_per_subj,
                 num_folds,
                 raw_data,
                 raw_data2=None,
                 voxel_unit=64,
                 process_num=4,
                 master_rank=0):
        self.labels = labels
        self.epochs_per_subj = epochs_per_subj
        self.num_folds = num_folds
        self.raw_data = raw_data
        self.num_voxels = raw_data[0].shape[1]
        self.raw_data2 = raw_data2
        self.num_voxels2 = raw_data2[0].shape[1] \
            if raw_data2 is not None else self.num_voxels
        self.voxel_unit = voxel_unit
        usable_cpus = usable_cpu_count()
        if process_num is None:
            self.process_num = usable_cpus
        else:
            self.process_num = np.min((process_num, usable_cpus))
        if self.process_num == 0:
            self.use_multiprocessing = False
        else:
            self.use_multiprocessing = True
        self.master_rank = master_rank
        if self.raw_data2 is not None \
                and len(self.raw_data) != len(self.raw_data2):
            raise ValueError('The raw data lists must have the same number '
                             'of elements for computing the correlations '
                             'element by element')
        if self.num_voxels == 0 or self.num_voxels2 == 0:
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
        """Run correlation-based voxel selection in master-worker model.

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
        """Master node's operation.

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
            'Master at rank %d starts to allocate tasks',
            MPI.COMM_WORLD.Get_rank()
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
            logger.debug(
                'master starts to send a task to worker %d' %
                i
            )
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
        """Worker node's operation.

        Receiving tasks from the master to process and sending the result back

        Parameters
        ----------
        clf: classification function
            the classifier to be used in cross validation

        Returns
        -------
        None
        """
        logger.debug(
            'worker %d is running, waiting for tasks from master at rank %d' %
            (MPI.COMM_WORLD.Get_rank(), self.master_rank)
        )
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        while 1:
            task = comm.recv(source=self.master_rank,
                             tag=MPI.ANY_TAG,
                             status=status)
            if status.Get_tag():
                break
            comm.send(self._voxel_scoring(task, clf),
                      dest=self.master_rank)

    def _correlation_computation(self, task):
        """Use BLAS API to do correlation computation (matrix multiplication).

        Parameters
        ----------
        task: tuple (start_voxel_id, num_processed_voxels)
            depicting the voxels assigned to compute

        Returns
        -------
        corr: 3D array in shape [num_processed_voxels, num_epochs, num_voxels]
            the correlation values of all subjects in all epochs
            for the assigned values, in row-major
            corr[i, e, s + j] = corr[j, e, s + i]
        """
        time1 = time.time()
        s = task[0]
        nEpochs = len(self.raw_data)
        logger.debug(
            'start to compute the correlation: #epochs: %d, '
            '#processed voxels: %d, #total voxels to compute against: %d' %
            (nEpochs, task[1], self.num_voxels2)
        )
        corr = np.zeros((task[1], nEpochs, self.num_voxels2),
                        np.float32, order='C')
        count = 0
        for i in range(len(self.raw_data)):
            mat = self.raw_data[i]
            mat2 = self.raw_data2[i] if self.raw_data2 is not None else mat
            no_trans = 'N'
            trans = 'T'
            blas.compute_self_corr_for_voxel_sel(no_trans, trans,
                                                 self.num_voxels2, task[1],
                                                 mat.shape[0], 1.0,
                                                 mat2, self.num_voxels2,
                                                 s, mat, self.num_voxels,
                                                 0.0, corr,
                                                 self.num_voxels2 * nEpochs,
                                                 count)
            count += 1
        time2 = time.time()
        logger.debug(
            'correlation computation for %d voxels, takes %.2f s' %
            (task[1], (time2 - time1))
        )
        return corr

    def _correlation_normalization(self, corr):
        """Do within-subject normalization.

        This method uses scipy.zscore to normalize the data,
        but is much slower than its C++ counterpart.
        It is doing in-place z-score.

        Parameters
        ----------
        corr: 3D array in shape [num_processed_voxels, num_epochs, num_voxels]
            the correlation values of all subjects in all epochs
            for the assigned values, in row-major

        Returns
        -------
        corr: 3D array in shape [num_processed_voxels, num_epochs, num_voxels]
            the normalized correlation values of all subjects in all epochs
            for the assigned values, in row-major
        """
        time1 = time.time()
        (sv, e, av) = corr.shape
        for i in range(sv):
            start = 0
            while start < e:
                cur_val = corr[i, start: start + self.epochs_per_subj, :]
                cur_val = .5 * np.log((cur_val + 1) / (1 - cur_val))
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

    def _prepare_for_cross_validation(self, corr, clf):
        """Prepare data for voxelwise cross validation.

        If the classifier is sklearn.svm.SVC with precomputed kernel,
        the kernel matrix of each voxel is computed, otherwise do nothing.

        Parameters
        ----------
        corr: 3D array in shape [num_processed_voxels, num_epochs, num_voxels]
            the normalized correlation values of all subjects in all epochs
            for the assigned values, in row-major
        clf: classification function
            the classifier to be used in cross validation

        Returns
        -------
        data: 3D numpy array
            If using sklearn.svm.SVC with precomputed kernel,
            it is in shape [num_processed_voxels, num_epochs, num_epochs];
            otherwise it is the input argument corr,
            in shape [num_processed_voxels, num_epochs, num_voxels]
        """
        time1 = time.time()
        (num_processed_voxels, num_epochs, _) = corr.shape
        if isinstance(clf, sklearn.svm.SVC) and clf.kernel == 'precomputed':
            # kernel matrices should be computed
            kernel_matrices = np.zeros((num_processed_voxels, num_epochs,
                                        num_epochs),
                                       np.float32, order='C')
            for i in range(num_processed_voxels):
                blas.compute_kernel_matrix('L', 'T',
                                           num_epochs, self.num_voxels2,
                                           1.0, corr,
                                           i, self.num_voxels2,
                                           0.0, kernel_matrices[i, :, :],
                                           num_epochs)
                # shrink the values for getting more stable alpha values
                # in SVM training iteration
                num_digits = len(str(int(kernel_matrices[i, 0, 0])))
                if num_digits > 2:
                    proportion = 10**(2-num_digits)
                    kernel_matrices[i, :, :] *= proportion
            data = kernel_matrices
        else:
            data = corr
        time2 = time.time()
        logger.debug(
            'cross validation data preparation takes %.2f s' %
            (time2 - time1)
        )
        return data

    def _do_cross_validation(self, clf, data, task):
        """Run voxelwise cross validation based on correlation vectors.

        clf: classification function
            the classifier to be used in cross validation
        data: 3D numpy array
            If using sklearn.svm.SVC with precomputed kernel,
            it is in shape [num_processed_voxels, num_epochs, num_epochs];
            otherwise it is the input argument corr,
            in shape [num_processed_voxels, num_epochs, num_voxels]
        task: tuple (start_voxel_id, num_processed_voxels)
            depicting the voxels assigned to compute

        Returns
        -------
        results: list of tuple (voxel_id, accuracy)
            the accuracy numbers of all voxels, in accuracy descending order
            the length of array equals the number of assigned voxels
        """
        time1 = time.time()

        if isinstance(clf, sklearn.svm.SVC) and clf.kernel == 'precomputed'\
                and self.use_multiprocessing:
            inlist = [(clf, i + task[0], self.num_folds, data[i, :, :],
                       self.labels) for i in range(task[1])]

            with multiprocessing.Pool(self.process_num) as pool:
                results = list(pool.starmap(_cross_validation_for_one_voxel,
                                            inlist))
        else:
            results = []
            for i in range(task[1]):
                result = _cross_validation_for_one_voxel(clf, i + task[0],
                                                         self.num_folds,
                                                         data[i, :, :],
                                                         self.labels)
                results.append(result)
        time2 = time.time()
        logger.debug(
            'cross validation for %d voxels, takes %.2f s' %
            (task[1], (time2 - time1))
        )
        return results

    def _voxel_scoring(self, task, clf):
        """The voxel selection process done in the worker node.

        Take the task in,
        do analysis on voxels specified by the task (voxel id, num_voxels)
        It is a three-stage pipeline consisting of:
        1. correlation computation
        2. within-subject normalization
        3. voxelwise cross validation

        Parameters
        ----------
        task: tuple (start_voxel_id, num_processed_voxels),
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
        corr = self._correlation_computation(task)
        # normalization
        # corr = self._correlation_normalization(corr)
        time3 = time.time()
        fcma_extension.normalization(corr, self.epochs_per_subj)
        time4 = time.time()
        logger.debug(
            'within-subject normalization for %d voxels '
            'using C++, takes %.2f s' %
            (task[1], (time4 - time3))
        )

        # cross validation
        data = self._prepare_for_cross_validation(corr, clf)
        if isinstance(clf, sklearn.svm.SVC) and clf.kernel == 'precomputed':
            # to save memory so that the process can be forked
            del corr
        results = self._do_cross_validation(clf, data, task)
        time2 = time.time()
        logger.info(
            'in rank %d, task %d takes %.2f s' %
            (MPI.COMM_WORLD.Get_rank(),
             (int(task[0] / self.voxel_unit)), (time2 - time1))
        )
        return results
