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
    """Correlation-based classification component of FCMA

    Parameters
    ----------

    clf: class
        The classifier used, normally a classifier class of sklearn

    epochs_per_subj: int, default 0
        The number of epochs of each subject
        within-subject normalization will be performed during
        classifier training if epochs_per_subj is specified
        default 0 means no within-subject normalization


    Attributes
    ----------

    training_data_: 2D numpy array in shape [num_samples, num_features]
        default None
        training_data\_ is None except clf is SVM.SVC with precomputed kernel,
        in which case training data is needed to compute
        the similarity vector for each sample to be classified

    test_raw_data_: a list of 2D array in shape [num_TRs, num_voxels]
        default None
        test_raw_data\_ is set after a prediction is called,
        if the new input data equals test_raw_data\_,
        test_data\_ can be reused

    test_data_: 2D numpy array in shape [num_samples, num_features]
        default None
        test_data\_ is set after a prediction is called,
        so that the test data does not need to be regenerated in the
        subsequent operations, e.g. getting decision values of the prediction

    num_voxels_: int
        The number of voxels per brain used in this classifier
        this is defined by the applied mask, normally the top voxels
        selected by FCMA voxel selection
        num_voxels\_ must be consistent in both training and classification

    num_samples_: int
        The number of samples of the training set
    """
    def __init__(self,
                 clf,
                 epochs_per_subj=0):
        self.clf = clf
        self.epochs_per_subj = epochs_per_subj
        return

    def _prepare_corerelation_data(self, X):
        """ compute auto-correlation for the input data X

        Parameters
        ----------
        X: a list of numpy array in shape [num_TRs, num_voxels]
            len(X) is the number of samples
            assuming all elements of X has the same num_voxels value

        Returns
        -------
        corr_data: the correlation data
                    in shape [len(X), num_voxels, num_voxels]
        """
        num_samples = len(X)
        assert num_samples > 0, \
            'at least one sample is needed for correlation computation'
        num_voxels = X[0].shape[1]
        assert num_voxels == self.num_voxels_, \
            'the number of voxels provided by X does not match ' \
            'the number of voxels defined in the model'
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
        return corr_data

    def _normalize_correlation_data(self, corr_data, norm_unit):
        """ normalize the correlation data if necessary

        Fisher-transform and then z-score the data for every norm_unit samples
        if norm_unit > 1.

        Parameters
        ----------
        corr_data: the correlation data
                    in shape [num_samples, num_voxels, num_voxels]
        norm_unit: int
                    the number of samples on which the normalization
                    is performed

        Returns
        -------
        normalized_corr_data: the normalized correlation data
                    in shape [num_samples, num_voxels, num_voxels]
        """
        # normalize if necessary
        if norm_unit > 1:
            num_samples = len(corr_data)
            num_voxels = self.num_voxels_
            normalized_corr_data = corr_data.reshape(1,
                                                     num_samples,
                                                     num_voxels * num_voxels)
            fcma_extension.normalization(normalized_corr_data, norm_unit)
            normalized_corr_data = normalized_corr_data.reshape(num_samples,
                                                                num_voxels,
                                                                num_voxels)
            logger.debug(
                'normalization done'
            )
        else:
            normalized_corr_data = corr_data
        return normalized_corr_data

    def _prepare_test_data(self, corr_data):
        """ prepare the data to be applied to the predict function

        if the classifier is SVM, do kernel precomputation,
        otherwise the test data is the reshaped corr_data

        Parameters
        ----------
        corr_data: the (normalized) correlation data
                    in shape [num_samples, num_voxels, num_voxels]

        Returns
        -------
        data: the data to be predicted, in shape of [num_samples, num_dim]
        """
        num_test_samples = len(corr_data)
        assert num_test_samples > 0, \
            'at least one test sample is needed'
        num_voxels = self.num_voxels_
        if isinstance(self.clf, sklearn.svm.SVC) \
                and self.clf.kernel == 'precomputed':
            assert self.training_data_ is not None, \
                'when using precomputed kernel of SVM, ' \
                'all training data must be provided'
            num_training_samples = self.training_data_.shape[0]
            data = np.zeros((num_test_samples, num_training_samples),
                            np.float32,
                            order='C')
            corr_data = corr_data.reshape(num_test_samples,
                                          num_voxels * num_voxels)
            # compute the similarity matrix using corr_data and training_data
            blas.compute_single_matrix_multiplication('T', 'N',
                                                      num_training_samples,
                                                      num_test_samples,
                                                      num_voxels * num_voxels,
                                                      1.0,
                                                      self.training_data_,
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
            data = corr_data.reshape(num_test_samples,
                                     num_voxels * num_voxels)
        return data

    def fit(self, X, y):
        """ use correlation data to train a model

        first compute the correlation of the input data,
        and then normalize within subject
        if more than one sample in one subject,
        and then fit to a model defined by self.clf.

        Parameters
        ----------
        X: a list of numpy array in shape [num_TRs, num_voxels]
            X contains the activity data filtered by top voxels
            and prepared for correlation computation.
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
        self.num_voxels_ = num_voxels
        self.num_samples_ = num_samples
        # correlation computation
        corr_data = self._prepare_corerelation_data(X)
        # normalization
        normalized_corr_data = self._normalize_correlation_data(
            corr_data,
            self.epochs_per_subj)
        # training
        if isinstance(self.clf, sklearn.svm.SVC) \
                and self.clf.kernel == 'precomputed':
            kernel_matrix = np.zeros((num_samples, num_samples),
                                     np.float32,
                                     order='C')
            # for using kernel matrix computation from voxel selection
            normalized_corr_data = normalized_corr_data.reshape(
                1,
                num_samples,
                num_voxels * num_voxels)
            blas.compute_kernel_matrix('L', 'T',
                                       num_samples, num_voxels * num_voxels,
                                       1.0, corr_data,
                                       0, num_voxels * num_voxels,
                                       0.0, kernel_matrix, num_samples)
            data = kernel_matrix
            # training data is in shape [num_samples, num_voxels * num_voxels]
            self.training_data_ = normalized_corr_data.reshape(
                num_samples,
                num_voxels * num_voxels)
            logger.debug(
                'kernel computation done'
            )
        else:
            data = normalized_corr_data.reshape(num_samples,
                                                num_voxels * num_voxels)
            self.training_data_ = None

        self.clf = self.clf.fit(data, y)
        self.test_raw_data_ = None
        self.test_data_ = None
        time2 = time.time()
        logger.info(
            'training done, takes %.2f s' %
            (time2 - time1)
        )
        return self

    def predict(self, X):
        """ use a trained model to predict correlation data

        first compute the correlation of the input data,
        and then normalize across all samples in the list
        if len(X) > 1,
        and then predict via self.clf.

        Parameters
        ----------
        X: a list of numpy array in shape [num_TRs, self.num_voxels\_]
            X contains the activity data filtered by top voxels
            and prepared for correlation computation.
            len(X) is the number of test samples
            if len(X) > 0: normalization is done
            on all test samples

        Returns
        -------
        y_pred: the predicted label of X, in shape [len(X),]
        """
        time1 = time.time()
        self.test_raw_data_ = X
        # correlation computation
        corr_data = self._prepare_corerelation_data(X)
        # normalization
        normalized_corr_data = self._normalize_correlation_data(
                                                                corr_data,
                                                                len(X))
        # test data generation
        self.test_data_ = self._prepare_test_data(normalized_corr_data)
        # prediction
        y_pred = self.clf.predict(self.test_data_)
        time2 = time.time()
        logger.info(
            'prediction done, takes %.2f s' %
            (time2 - time1)
        )
        return y_pred

    def _is_equal_to_test_raw_data(self, X):
        """ check if the new input data X is equal to the old one

        compare X and self.test_raw_data_ if it exists

        Parameters
        ----------
        X: a list of numpy array in shape [num_TRs, self.num_voxels\_]
            the input data to be checked

        Returns
        -------
        a boolean value to indicate if X == self.test_raw_data_
        """
        if self.test_raw_data_ is None or len(X) != len(self.test_raw_data_):
            return False
        # this for loop is faster than
        # doing np.array_equal(X, self.test_raw_data_) directly
        for new, old in zip(X, self.test_raw_data_):
            if not np.array_equal(new, old):
                return False
        return True

    def decision_function(self, X):
        """ output the decision value of the prediction

        if X is not equal to self.test_raw_data\_, i.e. predict is not called,
        first generate the test_data
        after getting the test_data, get the decision value via self.clf.

        Parameters
        ----------
        X: a list of numpy array in shape [num_TRs, self.num_voxels\_]
            X contains the activity data filtered by top voxels
            and prepared for correlation computation.
            len(X) is the number of test samples
            if len(X) > 1: normalization is done
            on all test samples

        Returns
        -------
        confidence: the predictions confidence values of X, in shape [len(X),]
        """
        if not self._is_equal_to_test_raw_data(X):
            self.test_raw_data_ = X
            # generate the test_data first
            # correlation computation
            corr_data = self._prepare_corerelation_data(X)
            # normalization
            normalized_corr_data = self._normalize_correlation_data(
                                                                    corr_data,
                                                                    len(X))
            # test data generation
            self.test_data_ = self._prepare_test_data(normalized_corr_data)
        confidence = self.clf.decision_function(self.test_data_)
        return confidence
