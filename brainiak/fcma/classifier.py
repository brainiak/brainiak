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
from sklearn.base import BaseEstimator
import sklearn
from . import fcma_extension
from . import cython_blas as blas
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "Classifier",
]

class Classifier(BaseEstimator):
    """
    the data has been processed by top voxels and prapared for correlation computation
    """
    def __init__(self,
                 epochs_per_subj=0,
                 clf=None):
        self.epochs_per_subj = epochs_per_subj
        self.clf = clf
        return

    def fit(self, X, y):
        """
        Parameters:
        ----------
        X: a list of numpy array in shape [nun_TRs, num_voxels]
        Y: labels, len(X) equals len(Y)

        Returns:
        self: return the object itself
        """
        time1 = time.time()
        assert len(X) == len(y), \
            'the number of samples does not match the number labels'
        num_samples = len(X)
        num_TRs = X[0].shape[0]
        num_voxels = X[0].shape[1]
        corr_data = np.zeros((num_samples, num_voxels, num_voxels),
                             np.float32, order='C')
        # compute correlation
        count = 0
        for data in X:
            blas.compute_single_self_correlation('L', 'N',
                                                 num_voxels,
                                                 num_TRs,
                                                 1.0, data,
                                                 num_voxels, 0.0,
                                                 corr_data,
                                                 num_voxels, count)
            count += 1
        logger.info(
            'correlation computation done'
        )
        # normalization if necessary
        if self.epochs_per_subj > 0:
            corr_data = corr_data.reshape(1, num_samples, num_voxels*num_voxels)
            fcma_extension.normalization(corr_data, self.epochs_per_subj)
            corr_data = corr_data.reshape(num_samples, num_voxels, num_voxels)
            logger.info(
                'normalization done'
            )
        # training
        if isinstance(self.clf, sklearn.svm.SVC) \
                and self.clf.kernel == 'precomputed':
            kernel_matrix = np.zeros((num_samples, num_samples), np.float32, order='C')
            # for using kernel matrix computation from voxel selection
            corr_data = corr_data.reshape(1, num_samples, num_voxels * num_voxels)
            blas.compute_kernel_matrix('L', 'T',
                                       num_samples, num_voxels * num_voxels,
                                       1.0, corr_data,
                                       0, num_voxels * num_voxels,
                                       0.0, kernel_matrix, num_samples)
            data = kernel_matrix
            logger.info(
                'kernel computation done'
            )
        else:
            data = corr_data
        self.clf = self.clf.fit(data, y)
        time2 = time.time()
        logger.info(
            'training done, takes %.2f s' %
            (time2 - time1)
        )
