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

.. [Wang2015] Full correlation matrix analysis (FCMA): An unbiased method for
   task-related functional connectivity",
   Yida Wang, Jonathan D Cohen, Kai Li, Nicholas B Turk-Browne.
   Journal of Neuroscience Methods, 2015.
"""

# Authors: Yida Wang
# (Intel Labs), 2016

import numpy as np
import time
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
    the data has been processed by top voxels
    and prapared for correlation computation
    """
    def __init__(self,
                 epochs_per_subj=0,
                 clf=None):
        self.epochs_per_subj = epochs_per_subj
        self.clf = clf
        self.training_data = None
        self.num_voxels = -1
        self.num_samples = -1
        return

    def fit(self, X, y):
        """ use correlation data to train a model

        the input data X is activity data, which needs to be first
        converted to correlation, and then normalized within subject
        if more than one sample in one subject, and then fit to a model
        defined by self.clf

        Parameters
        ----------
        X: a list of numpy array in shape [nun_TRs, num_voxels]
           assuming all elements of X has the same num_voxels value
        y: labels, len(X) equals len(Y)

        Returns
        -------
        self: return the object itself
        """
        time1 = time.time()
        assert len(X) == len(y), \
            'the number of samples does not match the number labels'
        num_samples = len(X)
        num_voxels = X[0].shape[1]  # see assumption above
        corr_data = np.zeros((num_samples, num_voxels, num_voxels),
                             np.float32, order='C')
        # compute correlation
        count = 0
        for data in X:
            num_TRs = data.shape[0]
            # syrk performs slower in this case
            # blas.compute_single_self_correlation_syrk('L', 'N',
            #                                     num_voxels,
            #                                     num_TRs,
            #                                    1.0, data,
            #                                     num_voxels, 0.0,
            #                                     corr_data,
            #                                     num_voxels, count)
            blas.compute_single_self_correlation_gemm('N', 'T',
                                                      num_voxels,
                                                      num_voxels,
                                                      num_TRs,
                                                      1.0, data,
                                                      num_voxels, num_voxels,
                                                      0.0, corr_data,
                                                      num_voxels, count)
            count += 1
        logger.debug(
            'correlation computation done'
        )
        # normalize if necessary
        if self.epochs_per_subj > 0:
            corr_data = corr_data.reshape(1,
                                          num_samples,
                                          num_voxels*num_voxels)
            fcma_extension.normalization(corr_data, self.epochs_per_subj)
            corr_data = corr_data.reshape(num_samples, num_voxels, num_voxels)
            logger.debug(
                'normalization done'
            )
        # training
        if isinstance(self.clf, sklearn.svm.SVC) \
                and self.clf.kernel == 'precomputed':
            kernel_matrix = np.zeros((num_samples, num_samples),
                                     np.float32,
                                     order='C')
            # for using kernel matrix computation from voxel selection
            corr_data = corr_data.reshape(1,
                                          num_samples,
                                          num_voxels * num_voxels)
            blas.compute_kernel_matrix('L', 'T',
                                       num_samples, num_voxels * num_voxels,
                                       1.0, corr_data,
                                       0, num_voxels * num_voxels,
                                       0.0, kernel_matrix, num_samples)
            data = kernel_matrix
            # training data is in shape [num_samples, num_voxels * num_voxels]
            self.training_data = corr_data.reshape(num_samples,
                                                   num_voxels * num_voxels)
            logger.debug(
                'kernel computation done'
            )
        else:
            data = corr_data.reshape(num_samples, num_voxels * num_voxels)
        self.num_voxels = num_voxels
        self.num_samples = num_samples
        self.clf = self.clf.fit(data, y)
        time2 = time.time()
        logger.info(
            'training done, takes %.2f s' %
            (time2 - time1)
        )
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X: a list of numpy array in shape [nun_TRs, num_voxels]
            len(X) equals num_samples
            if num_samples > 0: normalization is done on all subjects
            num_voxels equals the one used in the model

        Returns
        -------
        y_pred: the predicted label of X, in shape [num_samples,]
        """
        time1 = time.time()
        num_samples = len(X)
        assert num_samples > 0, \
            'at least one sample is needed'
        corr_data = np.zeros((num_samples, self.num_voxels, self.num_voxels),
                             np.float32,
                             order='C')
        # compute correlation
        count = 0
        for data in X:
            num_TRs = data.shape[0]
            num_voxels = data.shape[1]
            assert self.num_voxels == num_voxels, \
                'the number of voxels provided by X does not match ' \
                'the number of voxels defined in the model'
            blas.compute_single_self_correlation_gemm('N', 'T',
                                                      num_voxels,
                                                      num_voxels,
                                                      num_TRs,
                                                      1.0, data,
                                                      num_voxels, num_voxels,
                                                      0.0, corr_data,
                                                      num_voxels, count)
            count += 1
        logger.debug(
            'correlation computation done'
        )
        # normalize if necessary
        if num_samples > 1:
            corr_data = corr_data.reshape(1,
                                          num_samples,
                                          num_voxels * num_voxels)
            fcma_extension.normalization(corr_data, num_samples)
            corr_data = corr_data.reshape(num_samples, num_voxels, num_voxels)
            logger.debug(
                'normalization done'
            )
        # predict
        if isinstance(self.clf, sklearn.svm.SVC) \
                and self.clf.kernel == 'precomputed':
            assert self.training_data is not None, \
                'when using precomputed kernel of SVM, ' \
                'all training data must be provided'
            num_training_samples = self.training_data.shape[0]
            data = np.zeros((num_samples, num_training_samples),
                            np.float32,
                            order='C')
            corr_data = corr_data.reshape(num_samples, num_voxels * num_voxels)
            # compute the similarity matrix using corr_data and training_data
            blas.compute_single_matrix_multiplication('T', 'N',
                                                      num_training_samples,
                                                      num_samples,
                                                      num_voxels * num_voxels,
                                                      1.0,
                                                      self.training_data,
                                                      num_voxels * num_voxels,
                                                      corr_data,
                                                      num_voxels * num_voxels,
                                                      0.0,
                                                      data,
                                                      num_training_samples)
            logger.debug(
                'similarity matrix computation done'
            )
        else:
            data = corr_data.reshape(num_samples, num_voxels*num_voxels)
        y_pred = self.clf.predict(data)
        time2 = time.time()
        logger.info(
            'prediction done, takes %.2f s' %
            (time2 - time1)
        )
        return y_pred
