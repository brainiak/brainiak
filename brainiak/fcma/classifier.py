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

Correlation-based training and prediction
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

    The classifier first computes correlation of the input data,
    and normalizes them if needed, then uses the given classifier
    to train and/or predict the correlation data.
    NOTE: if the classifier is sklearn.svm.SVC with precomputed kernel,
    the test data may be provided in the fit method to compute
    the kernel matrix together with the training data to save the memory usage,
    but the test data will NEVER be seen in the model training.

    Parameters
    ----------

    clf: class
        The classifier used, normally a classifier class of sklearn

    num_processed_voxels: int, default 2000
        Used for SVM with precomputed kernel,
        every time it only computes correlation between num_process_voxels and
        the whole mask to aggregate the kernel matrices.
        This is to save the memory
        so as to handle correlations at a larger scale.

    epochs_per_subj: int, default 0
        The number of epochs of each subject
        within-subject normalization will be performed during
        classifier training if epochs_per_subj is specified
        default 0 means no within-subject normalization


    Attributes
    ----------

    training_data_: 2D numpy array in shape [num_samples, num_features]
        training_data\_ is None except clf is SVM.SVC with precomputed kernel,
        in which case training data is needed to compute
        the similarity vector for each sample to be classified.
        However, if the test samples are also provided during the fit,
        the similarity vectors can be precomputed too
        and then training_data\ is None

    test_raw_data_: a list of 2D array in shape [num_TRs, num_voxels]
        default None
        test_raw_data\_ is set after a prediction is called,
        if the new input data equals test_raw_data\_,
        test_data\_ can be reused

    test_data_: 2D numpy array in shape [num_samples, num_features]
        default None
        test_data\_ is set after a prediction is called,
        so that the test data does not need to be regenerated in the
        subsequent operations, e.g. getting decision values of the prediction.
        test_data\_ may also be set in the fit method
        if sklearn.svm.SVC with precomputed kernel
        and the test samples are known.
        NOTE: the test samples will never be used to fit the model.

    num_voxels_: int
        The number of voxels per brain used in this classifier
        this is defined by the applied mask, normally the top voxels
        selected by FCMA voxel selection
        num_voxels\_ must be consistent in both training and classification

    num_samples_: int
        The number of samples of the training set

    num_digits_: int
        The number of digits of the first value of the kernel matrix,
        for normalizing the similarity values accordingly
    """
    def __init__(self,
                 clf,
                 num_processed_voxels=2000,
                 epochs_per_subj=0):
        self.clf = clf
        self.num_processed_voxels = num_processed_voxels
        self.epochs_per_subj = epochs_per_subj
        self.num_digits_ = 0
        return

    def _prepare_auto_corerelation_data(self, X,
                                        start_voxel=0,
                                        num_processed_voxels=None):
        """ compute auto-correlation for the input data X

        it will generate the correlation between some voxels and all voxels

        Parameters
        ----------
        X: a list of numpy array in shape [num_TRs, num_voxels]
            len(X) is the number of samples
            assuming all elements of X has the same num_voxels value
        start_voxel: int, default 0
            the starting voxel id for correlation computation
        num_processed_voxels: int, default None
            the number of voxels it computes for correlation computation
            if it is None, it is set to self.num_voxels

        Returns
        -------
        corr_data: the correlation data
                    in shape [len(X), num_processed_voxels, num_voxels]
        """
        num_samples = len(X)
        assert num_samples > 0, \
            'at least one sample is needed for correlation computation'
        num_voxels = X[0].shape[1]
        assert num_voxels == self.num_voxels_, \
            'the number of voxels provided by X does not match ' \
            'the number of voxels defined in the model'
        if num_processed_voxels is None:
            num_processed_voxels = num_voxels
        corr_data = np.zeros((num_samples, num_processed_voxels, num_voxels),
                             np.float32, order='C')
        # compute correlation
        for idx, data in enumerate(X):
            num_TRs = data.shape[0]
            blas.compute_corr_vectors('N', 'T',
                                      num_voxels, num_processed_voxels,
                                      num_TRs,
                                      1.0, data, num_voxels,
                                      data, num_voxels,
                                      0.0, corr_data, num_voxels,
                                      start_voxel, idx)
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
                    in shape [num_samples, num_processed_voxels, num_voxels]
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
            [_, d2, d3] = corr_data.shape
            second_dimension = d2 * d3
            # this is a shallow copy
            normalized_corr_data = corr_data.reshape(1,
                                                     num_samples,
                                                     second_dimension)
            fcma_extension.normalization(normalized_corr_data, norm_unit)
            normalized_corr_data = normalized_corr_data.reshape(num_samples,
                                                                d2, d3)
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
            # shrink the values for getting more stable alpha values
            # in SVM training iteration
            num_digits = self.num_digits_
            if num_digits > 2:
                proportion = 10**(2-num_digits)
                data *= proportion
            logger.debug(
                'similarity matrix computation done'
            )
        else:
            data = corr_data.reshape(num_test_samples,
                                     num_voxels * num_voxels)
        return data

    def _compute_similarity_matrix_in_portion(self, X):
        kernel_matrix = np.zeros((self.num_samples_, self.num_samples_),
                                 np.float32,
                                 order='C')
        sr = 0
        row_length = self.num_processed_voxels
        normalized_corr_data = None
        while sr < self.num_voxels_:
            if row_length >= self.num_voxels_ - sr:
                row_length = self.num_voxels_ - sr
            # compute sub-correlation
            corr_data = self._prepare_auto_corerelation_data(X, sr,
                                                             row_length)
            # normalization
            normalized_corr_data = self._normalize_correlation_data(
                corr_data,
                self.epochs_per_subj)
            # compute partial similarity matrices
            # for using kernel matrix computation from voxel selection
            normalized_corr_data = normalized_corr_data.reshape(
                1,
                self.num_samples_,
                row_length * self.num_voxels_)
            blas.compute_kernel_matrix('L', 'T',
                                       self.num_samples_,
                                       row_length * self.num_voxels_,
                                       1.0, normalized_corr_data,
                                       0, row_length * self.num_voxels_,
                                       1.0, kernel_matrix, self.num_samples_)
            sr += row_length
        # shrink the values for getting more stable alpha values
        # in SVM training iteration
        num_digits = len(str(int(kernel_matrix[0, 0])))
        self.num_digits_ = num_digits
        if num_digits > 2:
            proportion = 10**(2-num_digits)
            kernel_matrix *= proportion
        return kernel_matrix, normalized_corr_data

    def fit(self, X, y, num_training_samples=None):
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
        num_training_samples: int, default None
            The number of samples used in the training,
            which is set when the similarity matrix is constructed
            portion by portion so the similarity vectors of the
            test data have to be computed here.
            This is ONLY set when sklearn.svm.SVC with
            precomputed kernel is used.
            If it is set, only those samples will be used to fit the model.

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
        if not (isinstance(self.clf, sklearn.svm.SVC)
                and self.clf.kernel == 'precomputed'):
            if num_training_samples is not None:
                num_training_samples = None
                logger.warn(
                    'num_training_samples should not be set for classifiers '
                    'other than SVM with precomputed kernels'
                )
            # correlation computation
            corr_data = self._prepare_auto_corerelation_data(X)
            # normalization
            normalized_corr_data = self._normalize_correlation_data(
                corr_data,
                self.epochs_per_subj)
            # training data prepare
            data = normalized_corr_data.reshape(num_samples,
                                                num_voxels * num_voxels)
            self.training_data_ = None
        else:  # SVM with precomputed kernel
            if self.num_processed_voxels < num_voxels:
                if num_training_samples is None:
                    raise RuntimeError('the similarity matrix will be '
                                       'computed portion by portion, '
                                       'the test samples must be predefined '
                                       'by specifying '
                                       'num_training_samples')
                if num_training_samples >= num_samples:
                    raise ValueError('the number of training samples '
                                     'must be smaller than '
                                     'the number of total samples')
            data, normalized_corr_data = \
                self._compute_similarity_matrix_in_portion(X)
            if self.num_processed_voxels >= num_voxels:
                # training data is in shape
                # [num_samples, num_voxels * num_voxels]
                self.training_data_ = normalized_corr_data.reshape(
                    num_samples,
                    num_voxels * num_voxels)
            else:
                # do not store training data because it was partially computed
                self.training_data_ = None
            logger.debug(
                'kernel computation done'
            )

        if num_training_samples is not None:
            self.test_raw_data_ = data[num_training_samples:,
                                       0:num_training_samples]
            self.test_data_ = data[num_training_samples:,
                                   0:num_training_samples]
            # limit training to the data specified by num_training_samples
            data = data[0:num_training_samples, 0:num_training_samples]
        # training
        self.clf = self.clf.fit(data, y[0:num_training_samples])
        # set the test data
        if num_training_samples is None:
            self.test_raw_data_ = None
            self.test_data_ = None
        time2 = time.time()
        logger.info(
            'training done, takes %.2f s' %
            (time2 - time1)
        )
        return self

    def predict(self, X=None):
        """ use a trained model to predict correlation data

        first compute the correlation of the input data,
        and then normalize across all samples in the list
        if len(X) > 1,
        and then predict via self.clf.
        If X is None, use the similarity vectors produced in fit
        to predict

        Parameters
        ----------
        X: a list of numpy array in shape [num_TRs, self.num_voxels\_]
            default None, meaning that the data to be predicted
            have been processed in the fit method.
            Otherwise, X contains the activity data filtered by top voxels
            and prepared for correlation computation.
            len(X) is the number of test samples
            if len(X) > 0: normalization is done
            on all test samples

        Returns
        -------
        y_pred: the predicted label of X, in shape [len(X),]
        """
        time1 = time.time()
        if X is not None:
            self.test_raw_data_ = X
            # correlation computation
            corr_data = self._prepare_auto_corerelation_data(X)
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

    def decision_function(self, X=None):
        """ output the decision value of the prediction

        if X is not equal to self.test_raw_data\_, i.e. predict is not called,
        first generate the test_data
        after getting the test_data, get the decision value via self.clf.
        if X is None, test_data\_ is ready to be used

        Parameters
        ----------
        X: a list of numpy array in shape [num_TRs, self.num_voxels\_]
            default None, meaning that the data to be predicted
            have been processed in the fit method.
            Otherwise, X contains the activity data filtered by top voxels
            and prepared for correlation computation.
            len(X) is the number of test samples
            if len(X) > 1: normalization is done
            on all test samples

        Returns
        -------
        confidence: the predictions confidence values of X, in shape [len(X),]
        """
        if X is not None and not self._is_equal_to_test_raw_data(X):
            self.test_raw_data_ = X
            # generate the test_data first
            # correlation computation
            corr_data = self._prepare_auto_corerelation_data(X)
            # normalization
            normalized_corr_data = self._normalize_correlation_data(
                                                                    corr_data,
                                                                    len(X))
            # test data generation
            self.test_data_ = self._prepare_test_data(normalized_corr_data)
        confidence = self.clf.decision_function(self.test_data_)
        return confidence
