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
"""Semi-Supervised Shared Response Model (SS-SRM)

The implementations are based on the following publications:

.. [Turek2016] "A Semi-Supervised Method for Multi-Subject fMRI Functional
   Alignment",
   J. Turek, T. Willke, P.-H. Chen, P. Ramadge
   under review, 2016.
"""

# Authors: Javier Turek (Intel Labs), 2016

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import NotFittedError
from sklearn.utils.multiclass import unique_labels
import theano
import theano.tensor as T
import theano.compile.sharedvalue as S
from pymanopt.manifolds import Euclidean
from pymanopt.manifolds import Product
from pymanopt.solvers import ConjugateGradient
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
import gc

from brainiak.utils import utils
from brainiak.funcalign import srm

__all__ = [
    "SSSRM"
]

logger = logging.getLogger(__name__)


class SSSRM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Semi-Supervised Shared Response Model (SS-SRM)

    Given multi-subject data, factorize it as a shared response S among all
    subjects and an orthogonal transform W per subject, using also labeled
    data to train a Multinomial Logistic Regression (MLR) classifier (with
    l2 regularization) in a semi-supervised manner:

    .. math::
        (1-\\alpha) Loss_{SRM}(W_i,S;X_i)
        + \\alpha/\\gamma Loss_{MLR}(\\theta, bias; {(W_i^T \\times Z_i, y_i})
        + R(\\theta)
        :label: sssrm-eq

    (see Equations (1) and (4) in [Turek2016]_).

    Parameters
    ----------

    n_iter : int, default: 10
        Number of iterations to run the algorithm.

    features : int, default: 50
        Number of features to compute.

    gamma : float, default: 1.0
        Regularization parameter for the classifier.

    alpha : float, default: 0.5
        Balance parameter between the SRM term and the MLR term.

    rand_seed : int, default: 0
        Seed for initializing the random number generator.


    Attributes
    ----------

    w_ : list of array, element i has shape=[voxels_i, features]
        The orthogonal transforms (mappings) for each subject.

    s_ : array, shape=[features, samples]
        The shared response.

    theta_ : array, shape=[classes, features]
        The MLR class plane parameters.

    bias_ : array, shape=[classes]
        The MLR class biases.

    classes_ : array of int, shape=[classes]
        Mapping table for each classes to original class label.

    random_state_: `RandomState`
        Random number generator initialized using rand_seed

    Note
    ----

        The number of voxels may be different between subjects. However, the
        number of samples for the alignment data must be the same across
        subjects. The number of labeled samples per subject can be different.

        The Semi-Supervised Shared Response Model is approximated using the
        Block-Coordinate Descent (BCD) algorithm proposed in [Turek2016]_.

        This is a single node version.
    """

    def __init__(self, n_iter=10, features=50, gamma=1.0, alpha=0.5,
                 rand_seed=0):
        self.n_iter = n_iter
        self.features = features
        self.gamma = gamma
        self.alpha = alpha
        self.rand_seed = rand_seed
        return

    def fit(self, X, y, Z):
        """Compute the Semi-Supervised Shared Response Model

        Parameters
        ----------

        X : list of 2D arrays, element i has shape=[voxels_i, n_align]
            Each element in the list contains the fMRI data for alignment of
            one subject. There are n_align samples for each subject.

        y : list of arrays of int, element i has shape=[samples_i]
            Each element in the list contains the labels for the data samples
            in Z.

        Z : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject
            for training the MLR classifier.

        """
        logger.info('Starting SS-SRM')

        # Check that the alpha value is in range (0.0,1.0)
        if 0.0 >= self.alpha or self.alpha >= 1.0:
            raise ValueError("Alpha parameter should be in range (0.0, 1.0)")

        # Check that the regularizer value is positive
        if 0.0 >= self.gamma:
            raise ValueError("Gamma parameter should be positive.")

        # Check the number of subjects
        if len(X) <= 1 or len(y) <= 1 or len(Z) <= 1:
            raise ValueError("There are not enough subjects in the input "
                             "data to train the model.")

        if not (len(X) == len(y)) or not (len(X) == len(Z)):
            raise ValueError("Different number of subjects in data.")

        # Check for input data sizes
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough samples to train the model with "
                "{0:d} features.".format(self.features))

        # Check if all subjects have same number of TRs for alignment
        # and if alignment and classification data have the same number of
        # voxels per subject. Also check that there labels for all the classif.
        # sample
        number_trs = X[0].shape[1]
        number_subjects = len(X)
        for subject in range(number_subjects):
            assert_all_finite(X[subject])
            assert_all_finite(Z[subject])
            if X[subject].shape[1] != number_trs:
                raise ValueError("Different number of alignment samples "
                                 "between subjects.")
            if X[subject].shape[0] != Z[subject].shape[0]:
                raise ValueError("Different number of voxels between alignment"
                                 " and classification data (subject {0:d})"
                                 ".".format(subject))
            if Z[subject].shape[1] != y[subject].size:
                raise ValueError("Different number of samples and labels in "
                                 "subject {0:d}.".format(subject))

        # Map the classes to [0..C-1]
        new_y = self._init_classes(y)

        # Run SS-SRM
        self.w_, self.s_, self.theta_, self.bias_ = self._sssrm(X, Z, new_y)

        return self

    def _init_classes(self, y):
        """Map all possible classes to the range [0,..,C-1]

        Parameters
        ----------

        y : list of arrays of int, each element has shape=[samples_i,]
            Labels of the samples for each subject


        Returns
        -------
        new_y : list of arrays of int, each element has shape=[samples_i,]
            Mapped labels of the samples for each subject

        Note
        ----
            The mapping of the classes is saved in the attribute classes_.
        """
        self.classes_ = unique_labels(utils.concatenate_not_none(y))
        new_y = [None] * len(y)
        for s in range(len(y)):
            new_y[s] = np.digitize(y[s], self.classes_) - 1
        return new_y

    def transform(self, X, y=None):
        """Use the model to transform matrix to Shared Response space

        Parameters
        ----------

        X : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject
            note that number of voxels and samples can vary across subjects.

        y : not used as it only applies the mappings


        Returns
        -------

        s : list of 2D arrays, element i has shape=[features_i, samples_i]
            Shared responses from input data (X)
        """

        # Check if the model exist
        if hasattr(self, 'w_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        s = [None] * len(X)
        for subject in range(len(X)):
            s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    def predict(self, X):
        """Classify the output for given data

        Parameters
        ----------

        X : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject
            The number of voxels should be according to each subject at
            the moment of training the model.

        Returns
        -------
        p: list of arrays, element i has shape=[samples_i]
            Predictions for each data sample.
        """
        # Check if the model exist
        if hasattr(self, 'w_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        X_shared = self.transform(X)
        p = [None] * len(X_shared)
        for subject in range(len(X_shared)):
            sumexp, _, exponents = utils.sumexp_stable(
                self.theta_.T.dot(X_shared[subject]) + self.bias_)
            p[subject] = self.classes_[
                (exponents / sumexp[np.newaxis, :]).argmax(axis=0)]

        return p

    def _sssrm(self, data_align, data_sup, labels):
        """Block-Coordinate Descent algorithm for fitting SS-SRM.

        Parameters
        ----------

        data_align : list of 2D arrays, element i has shape=[voxels_i, n_align]
            Each element in the list contains the fMRI data for alignment of
            one subject. There are n_align samples for each subject.

        data_sup : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject for
            the classification task.

        labels : list of arrays of int, element i has shape=[samples_i]
            Each element in the list contains the labels for the data samples
            in data_sup.


        Returns
        -------

        w : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        s : array, shape=[features, samples]
            The shared response.
        """
        classes = self.classes_.size

        # Initialization:
        self.random_state_ = np.random.RandomState(self.rand_seed)
        random_states = [
            np.random.RandomState(self.random_state_.randint(2**32))
            for i in range(len(data_align))]
        # Set Wi's to a random orthogonal voxels by TRs
        w, _ = srm._init_w_transforms(data_align, self.features, random_states)

        # Initialize the shared response S
        s = SSSRM._compute_shared_response(data_align, w)

        # Initialize theta and bias
        theta, bias = self._update_classifier(data_sup, labels, w, classes)

        # calculate and print the objective function
        if logger.isEnabledFor(logging.INFO):
            objective = self._objective_function(data_align, data_sup, labels,
                                                 w, s, theta, bias)
            logger.info('Objective function %f' % objective)

        # Main loop:
        for iteration in range(self.n_iter):
            logger.info('Iteration %d' % (iteration + 1))

            # Update the mappings Wi
            w = self._update_w(data_align, data_sup, labels, w, s, theta, bias)

            # Output the objective function
            if logger.isEnabledFor(logging.INFO):
                objective = self._objective_function(data_align, data_sup,
                                                     labels, w, s, theta, bias)
                logger.info('Objective function after updating Wi  %f'
                            % objective)

            # Update the shared response S
            s = SSSRM._compute_shared_response(data_align, w)

            # Output the objective function
            if logger.isEnabledFor(logging.INFO):
                objective = self._objective_function(data_align, data_sup,
                                                     labels, w, s, theta, bias)
                logger.info('Objective function after updating S   %f'
                            % objective)

            # Update the MLR classifier, theta and bias
            theta, bias = self._update_classifier(data_sup, labels, w, classes)

            # Output the objective function
            if logger.isEnabledFor(logging.INFO):
                objective = self._objective_function(data_align, data_sup,
                                                     labels, w, s, theta, bias)
                logger.info('Objective function after updating MLR %f'
                            % objective)

        return w, s, theta, bias

    def _update_classifier(self, data, labels, w, classes):
        """Update the classifier parameters theta and bias

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject for
            the classification task.

        labels : list of arrays of int, element i has shape=[samples_i]
            Each element in the list contains the labels for the data samples
            in data_sup.

        w : list of 2D array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        classes : int
            The number of classes in the classifier.


        Returns
        -------

        theta : array, shape=[features, classes]
            The MLR parameter for the class planes.

        bias : array shape=[classes,]
            The MLR parameter for class biases.
        """

        # Stack the data and labels for training the classifier
        data_stacked, labels_stacked, weights = \
            SSSRM._stack_list(data, labels, w)

        features = w[0].shape[1]
        total_samples = weights.size

        data_th = S.shared(data_stacked.astype(theano.config.floatX))
        val_ = S.shared(labels_stacked)
        total_samples_S = S.shared(total_samples)
        theta_th = T.matrix(name='theta', dtype=theano.config.floatX)
        bias_th = T.col(name='bias', dtype=theano.config.floatX)
        constf2 = S.shared(self.alpha / self.gamma, allow_downcast=True)
        weights_th = S.shared(weights)

        log_p_y_given_x = \
            T.log(T.nnet.softmax((theta_th.T.dot(data_th.T)).T + bias_th.T))
        f = -constf2 * T.sum((log_p_y_given_x[T.arange(total_samples_S), val_])
                             / weights_th) + 0.5 * T.sum(theta_th ** 2)

        manifold = Product((Euclidean(features, classes),
                            Euclidean(classes, 1)))
        problem = Problem(manifold=manifold, cost=f, arg=[theta_th, bias_th],
                          verbosity=0)
        solver = ConjugateGradient(mingradnorm=1e-6)
        solution = solver.solve(problem)
        theta = solution[0]
        bias = solution[1]

        del constf2
        del theta_th
        del bias_th
        del data_th
        del val_
        del solver
        del solution

        return theta, bias

    def _update_w(self, data_align, data_sup, labels, w, s, theta, bias):
        """

        Parameters
        ----------
        data_align : list of 2D arrays, element i has shape=[voxels_i, n_align]
            Each element in the list contains the fMRI data for alignment of
            one subject. There are n_align samples for each subject.

        data_sup : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject for
            the classification task.

        labels : list of arrays of int, element i has shape=[samples_i]
            Each element in the list contains the labels for the data samples
            in data_sup.

        w : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        s : array, shape=[features, samples]
            The shared response.

        theta : array, shape=[classes, features]
            The MLR class plane parameters.

        bias : array, shape=[classes]
            The MLR class biases.

        Returns
        -------

        w : list of 2D array, element i has shape=[voxels_i, features]
            The updated orthogonal transforms (mappings).
        """
        subjects = len(data_align)

        s_th = S.shared(s.astype(theano.config.floatX))
        theta_th = S.shared(theta.T.astype(theano.config.floatX))
        bias_th = S.shared(bias.T.astype(theano.config.floatX),
                           broadcastable=(True, False))

        for subject in range(subjects):
            logger.info('Subject Wi %d' % subject)
            # Solve for subject i
            # Create the theano function
            w_th = T.matrix(name='W', dtype=theano.config.floatX)
            data_srm_subject = \
                S.shared(data_align[subject].astype(theano.config.floatX))
            constf1 = \
                S.shared((1 - self.alpha) * 0.5 / data_align[subject].shape[1],
                         allow_downcast=True)
            f1 = constf1 * T.sum((data_srm_subject - w_th.dot(s_th))**2)

            if data_sup[subject] is not None:
                lr_samples_S = S.shared(data_sup[subject].shape[1])
                data_sup_subject = \
                    S.shared(data_sup[subject].astype(theano.config.floatX))
                labels_S = S.shared(labels[subject])
                constf2 = S.shared(-self.alpha / self.gamma
                                   / data_sup[subject].shape[1],
                                   allow_downcast=True)

                log_p_y_given_x = T.log(T.nnet.softmax((theta_th.dot(
                    w_th.T.dot(data_sup_subject))).T + bias_th))
                f2 = constf2 * T.sum(
                    log_p_y_given_x[T.arange(lr_samples_S), labels_S])
                f = f1 + f2
            else:
                f = f1

            # Define the problem and solve
            f_subject = self._objective_function_subject(data_align[subject],
                                                         data_sup[subject],
                                                         labels[subject],
                                                         w[subject],
                                                         s, theta, bias)
            minstep = np.amin(((10**-np.floor(np.log10(f_subject))), 1e-1))
            manifold = Stiefel(w[subject].shape[0], w[subject].shape[1])
            problem = Problem(manifold=manifold, cost=f, arg=w_th, verbosity=0)
            solver = ConjugateGradient(mingradnorm=1e-2, minstepsize=minstep)
            w[subject] = np.array(solver.solve(
                problem, x=w[subject].astype(theano.config.floatX)))
            if data_sup[subject] is not None:
                del f2
                del log_p_y_given_x
                del data_sup_subject
                del labels_S
            del solver
            del problem
            del manifold
            del f
            del f1
            del data_srm_subject
            del w_th
        del theta_th
        del bias_th
        del s_th

        # Run garbage collector to avoid filling up the memory
        gc.collect()
        return w

    @staticmethod
    def _compute_shared_response(data, w):
        """ Compute the shared response S

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        w : list of 2D arrays, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.


        Returns
        -------

        s : array, shape=[features, samples]
            The shared response for the subjects data with the mappings in w.
        """
        s = np.zeros((w[0].shape[1], data[0].shape[1]))
        for m in range(len(w)):
            s = s + w[m].T.dot(data[m])
        s /= len(w)
        return s

    def _objective_function(self, data_align, data_sup, labels, w, s, theta,
                            bias):
        """Compute the objective function of the Semi-Supervised SRM

        See :eq:`sssrm-eq`.

        Parameters
        ----------

        data_align : list of 2D arrays, element i has shape=[voxels_i, n_align]
            Each element in the list contains the fMRI data for alignment of
            one subject. There are n_align samples for each subject.

        data_sup : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject for
            the classification task.

        labels : list of arrays of int, element i has shape=[samples_i]
            Each element in the list contains the labels for the data samples
            in data_sup.

        w : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        s : array, shape=[features, samples]
            The shared response.

        theta : array, shape=[classes, features]
            The MLR class plane parameters.

        bias : array, shape=[classes]
            The MLR class biases.


        Returns
        -------

        f_val : float
            The SS-SRM objective function evaluated based on the parameters to
            this function.
        """
        subjects = len(data_align)

        # Compute the SRM  loss
        f_val = 0.0
        for subject in range(subjects):
            samples = data_align[subject].shape[1]
            f_val += (1 - self.alpha) * (0.5 / samples) \
                * np.linalg.norm(data_align[subject] - w[subject].dot(s),
                                 'fro')**2

        # Compute the MLR loss
        f_val += self._loss_lr(data_sup, labels, w, theta, bias)

        return f_val

    def _objective_function_subject(self, data_align, data_sup, labels, w, s,
                                    theta, bias):
        """Compute the objective function for one subject.

        .. math:: (1-C)*Loss_{SRM}_i(W_i,S;X_i)
        .. math:: + C/\gamma * Loss_{MLR_i}(\theta, bias; {(W_i^T*Z_i, y_i})
        .. math:: + R(\theta)

        Parameters
        ----------

        data_align : 2D array, shape=[voxels_i, samples_align]
            Contains the fMRI data for alignment of subject i.

        data_sup : 2D array, shape=[voxels_i, samples_i]
            Contains the fMRI data of one subject for the classification task.

        labels : array of int, shape=[samples_i]
            The labels for the data samples in data_sup.

        w : array, shape=[voxels_i, features]
            The orthogonal transform (mapping) :math:`W_i` for subject i.

        s : array, shape=[features, samples]
            The shared response.

        theta : array, shape=[classes, features]
            The MLR class plane parameters.

        bias : array, shape=[classes]
            The MLR class biases.


        Returns
        -------

        f_val : float
            The SS-SRM objective function for subject i evaluated on the
            parameters to this function.
        """
        # Compute the SRM  loss
        f_val = 0.0
        samples = data_align.shape[1]
        f_val += (1 - self.alpha) * (0.5 / samples) \
            * np.linalg.norm(data_align - w.dot(s), 'fro')**2

        # Compute the MLR loss
        f_val += self._loss_lr_subject(data_sup, labels, w, theta, bias)

        return f_val

    def _loss_lr_subject(self, data, labels, w, theta, bias):
        """Compute the Loss MLR for a single subject (without regularization)

        Parameters
        ----------

        data : array, shape=[voxels, samples]
            The fMRI data of subject i for the classification task.

        labels : array of int, shape=[samples]
            The labels for the data samples in data.

        w : array, shape=[voxels, features]
            The orthogonal transform (mapping) :math:`W_i` for subject i.

        theta : array, shape=[classes, features]
            The MLR class plane parameters.

        bias : array, shape=[classes]
            The MLR class biases.

        Returns
        -------

        loss : float
            The loss MLR for the subject
        """
        if data is None:
            return 0.0

        samples = data.shape[1]

        thetaT_wi_zi_plus_bias = theta.T.dot(w.T.dot(data)) + bias
        sum_exp, max_value, _ = utils.sumexp_stable(thetaT_wi_zi_plus_bias)
        sum_exp_values = np.log(sum_exp) + max_value

        aux = 0.0
        for sample in range(samples):
            label = labels[sample]
            aux += thetaT_wi_zi_plus_bias[label, sample]
        return self.alpha / samples / self.gamma * (sum_exp_values.sum() - aux)

    def _loss_lr(self, data, labels, w, theta, bias):
        """Compute the Loss MLR (with the regularization)

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject for
            the classification task.

        labels : list of arrays of int, element i has shape=[samples_i]
            Each element in the list contains the labels for the samples in
            data.

        w : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        theta : array, shape=[classes, features]
            The MLR class plane parameters.

        bias : array, shape=[classes]
            The MLR class biases.


        Returns
        -------

        loss : float
            The loss MLR for the SS-SRM model
        """
        subjects = len(data)
        loss = 0.0
        for subject in range(subjects):
            if labels[subject] is not None:
                loss += self._loss_lr_subject(data[subject], labels[subject],
                                              w[subject], theta, bias)

        return loss + 0.5 * np.linalg.norm(theta, 'fro')**2

    @staticmethod
    def _stack_list(data, data_labels, w):
        """Construct a numpy array by stacking arrays in a list

        Parameter
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject for
            the classification task.

        data_labels : list of arrays of int, element i has shape=[samples_i]
            Each element in the list contains the labels for the samples in
            data.

        w : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.


        Returns
        -------

        data_stacked : 2D array, shape=[samples, features]
            The data samples from all subjects are stacked into a single
            2D array, where "samples" is the sum of samples_i.

        labels_stacked : array, shape=[samples,]
            The labels from all subjects are stacked into a single
            array, where "samples" is the sum of samples_i.

        weights : array, shape=[samples,]
            The number of samples of the subject that are related to that
            sample. They become a weight per sample in the MLR loss.
        """
        labels_stacked = utils.concatenate_not_none(data_labels)

        weights = np.empty((labels_stacked.size,))
        data_shared = [None] * len(data)
        curr_samples = 0
        for s in range(len(data)):
            if data[s] is not None:
                subject_samples = data[s].shape[1]
                curr_samples_end = curr_samples + subject_samples
                weights[curr_samples:curr_samples_end] = subject_samples
                data_shared[s] = w[s].T.dot(data[s])
                curr_samples += data[s].shape[1]

        data_stacked = utils.concatenate_not_none(data_shared, axis=1).T
        return data_stacked, labels_stacked, weights
