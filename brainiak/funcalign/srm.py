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
"""Probabilistic Shared Response Model (SRM)

This implementation is based on the following publications:

.. [Chen2015] "A Reduced-Dimension fMRI Shared Response Model",
   P.-H. Chen, J. Chen, Y. Yeshurun-Dishon, U. Hasson, J. Haxby, P. Ramadge
   Advances in Neural Information Processing Systems (NIPS), 2015.
   http://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model

.. [Anderson2016] "Scaling Up Machine Learning Algorithms For Multi-Subject
   Neuroimaging Analysis",
   Michael J. Anderson, Mihai Capotă, Javier S. Turek, Xia Zhu, Theodore L.
   Willke, Yida Wang, Po-Hsuan Chen, Jeremy R. Manning, Peter J. Ramadge,
   Kenneth A. Norman, under review, 2016.
"""

# Authors: Po-Hsuan Chen (Princeton Neuroscience Institute) and Javier Turek
# (Intel Labs), 2015

import numpy as np
import scipy
from sklearn.base import BaseEstimator
from sklearn.utils import assert_all_finite

__all__ = [
    "SRM",
]


def _init_w_transforms(data, features):
    """Initialize the mappings (Wi) for the SRM with random orthogonal matrices.

    Parameters
    ----------

    data : list of 2D arrays, element i has shape=[voxels_i, samples]
        Each element in the list contains the fMRI data of one subject.

    features : int
        The number of features in the model.


    Returns
    -------

    w : list of array, element i has shape=[voxels_i, features]
        The initialized orthogonal transforms (mappings) :math:`W_i` for each
        subject.

    voxels : list of int
        A list with the number of voxels per subject.


    .. note:: This function assumes that the numpy random number generator was
       initialized.

       Not thread safe.
    """
    w = []
    subjects = len(data)
    voxels = np.empty(subjects, dtype=int)

    # Set Wi to a random orthogonal voxels by features matrix
    for subject in range(subjects):
        voxels[subject] = data[subject].shape[0]
        rnd_matrix = np.mat(np.random.random((voxels[subject], features)))
        q, r = np.linalg.qr(rnd_matrix)
        w.append(q)

    return w, voxels


class SRM(BaseEstimator):
    """Probabilistic Shared Response Model (SRM)

    Given multi-subject data, factorize it as a shared response S among all
    subjects and an orthogonal transform W per subject:

    .. math:: X_i \\approx W_i S ,~for~all~i=1\dots N

    Parameters
    ----------

    n_iter : int, default: 10
        Number of iterations to run the algorithm.

    features : int, default: 50
        Number of features to compute.

    rand_seed : int, default: 0
        Seed for initializing the random number generator.

    verbose : boolean, default: False
        Verbose mode flag.


    Attributes
    ----------

    w_ : list of array, element i has shape=[voxels_i, features]
        The orthogonal transforms (mappings) for each subject.

    s_ : array, shape=[features, samples]
        The shared response.

    sigma_s_ : array, shape=[features, features]
        The covariance of the shared response Normal distribution.

    mu_ : list of array, element i has shape=[voxels_i]
        The voxel means over the samples for each subject.

    rho2_ : array, shape=[subjects]
        The estimated noise variance :math:`\\rho_i^2` for each subject


    .. note::
       The number of voxels may be different between subjects. However, the
       number of samples must be the same across subjects.

       The probabilistic Shared Response Model is approximated using the
       Expectation Maximization (EM) algorithm proposed in [Chen2015]_. The
       implementation follows the optimizations published in [Anderson2016]_.

       This is a single node version.

       The run-time complexity is :math:`O(I (V T K + K^3))` and the memory
       complexity is :math:`O(V T)` with I - the number of iterations, V - the
       sum of voxels from all subjects, T - the number of samples, and K - the
       number of features (typically, :math:`V \\gg T \\gg K`).
    """

    def __init__(self, n_iter=10, features=50, rand_seed=0, verbose=False):
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        self.verbose = verbose
        return

    def fit(self, X, y=None):
        """Compute the probabilistic Shared Response Model

        Parameters
        ----------
        X :  list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used
        """
        if self.verbose:
            print('Running Probabilistic SRM')  # noqa FIXME

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError("There are not enough subjects "
                             "({0:d}) to train the model.".format(len(X)))

        # Check for input data sizes
        if X[0].shape[1] < self.features:
            raise ValueError(
                "There are not enough samples to train the model with "
                "{0:d} features.".format(self.features))

        # Check if all subjects have same number of TRs
        number_trs = X[0].shape[1]
        number_subjects = len(X)
        for subject in range(number_subjects):
            assert_all_finite(X[subject])
            if X[subject].shape[1] != number_trs:
                raise ValueError(
                    "Different number of samples between subjects.")

        # Run SRM
        self.sigma_s_, self.w_, self.mu_, self.rho2_, self.s_ = self._srm(X)

        return self

    def _init_structures(self, data, subjects):
        """Initializes data structures for SRM and preprocess the data.


        Parameters
        ----------
        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        subjects : int
            The total number of subjects in `data`.


        Returns
        -------
        x : list of array, element i has shape=[voxels_i, samples]
            Demeaned data for each subject.

        mu : list of array, element i has shape=[voxels_i]
            Voxel means over samples, per subject.

        rho2 : array, shape=[subjects]
            Noise variance :math:`\\rho^2` per subject.

        trace_xtx : array, shape=[subjects]
            The squared Frobenius norm of the demeaned data in `x`.
        """
        x = []
        mu = []
        rho2 = np.zeros(subjects)

        trace_xtx = np.zeros(subjects)
        for subject in range(subjects):
            mu.append(np.mean(data[subject], 1))
            rho2[subject] = 1
            trace_xtx[subject] = np.sum(data[subject] ** 2)
            x.append(data[subject] - mu[subject][:, np.newaxis])

        return x, mu, rho2, trace_xtx

    def _likelihood(self, chol_sigma_s_rhos, log_det_psi, chol_sigma_s,
                    trace_xt_invsigma2_x, inv_sigma_s_rhos, wt_invpsi_x,
                    samples):
        """Calculate the log-likelihood function


        Parameters
        ----------

        chol_sigma_s_rhos : array, shape=[features, features]
            Cholesky factorization of the matrix (Sigma_S + sum_i(1/rho_i^2)
            * I)

        log_det_psi : float
            Determinant of diagonal matrix Psi (containing the rho_i^2 value
            voxels_i times).

        chol_sigma_s : array, shape=[features, features]
            Cholesky factorization of the matrix Sigma_S

        trace_xt_invsigma2_x : float
            Trace of :math:`\\sum_i (||X_i||_F^2/\\rho_i^2)`

        inv_sigma_s_rhos : array, shape=[features, features]
            Inverse of :math:`(\\Sigma_S + \\sum_i(1/\\rho_i^2) * I)`

        wt_invpsi_x : array, shape=[features, samples]

        samples : int
            The total number of samples in the data.


        Returns
        -------

        loglikehood : float
            The log-likelihood value.
        """
        log_det = (np.log(np.diag(chol_sigma_s_rhos) ** 2).sum() + log_det_psi
                   + np.log(np.diag(chol_sigma_s) ** 2).sum())
        loglikehood = -0.5 * samples * log_det - 0.5 * trace_xt_invsigma2_x
        loglikehood += 0.5 * np.trace(
            wt_invpsi_x.T.dot(inv_sigma_s_rhos).dot(wt_invpsi_x))
        # + const --> -0.5*nTR*nvoxel*subjects*math.log(2*math.pi)

        return loglikehood

    def _srm(self, data):
        """Expectation-Maximization algorithm for fitting the probabilistic SRM.

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.


        Returns
        -------

        sigma_s : array, shape=[features, features]
            The covariance :math:`\\Sigma_s` of the shared response Normal
            distribution.

        w : list of array, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        mu : list of array, element i has shape=[voxels_i]
            The voxel means :math:`\\mu_i` over the samples for each subject.

        rho2 : array, shape=[subjects]
            The estimated noise variance :math:`\\rho_i^2` for each subject

        s : array, shape=[features, samples]
            The shared response.
        """

        samples = data[0].shape[1]
        subjects = len(data)

        np.random.seed(self.rand_seed)

        # Initialization step: initialize the outputs with initial values,
        # voxels with the number of voxels in each subject, and trace_xtx with
        # the ||X_i||_F^2 of each subject.
        w, voxels = _init_w_transforms(data, self.features)
        x, mu, rho2, trace_xtx = self._init_structures(data, subjects)
        shared_response = np.zeros((self.features, samples))
        sigma_s = np.identity(self.features)

        # Main loop of the algorithm (run
        for iteration in range(self.n_iter):
            if self.verbose:
                print('Iteration %d' % (iteration + 1))  # noqa FIXME

            # E-step:

            # Sum the inverted the rho2 elements for computing W^T * Psi^-1 * W
            rho0 = (1 / rho2).sum()

            # Invert Sigma_s using Cholesky factorization
            (chol_sigma_s, lower_sigma_s) = scipy.linalg.cho_factor(
                sigma_s, check_finite=False)
            inv_sigma_s = scipy.linalg.cho_solve(
                (chol_sigma_s, lower_sigma_s), np.identity(self.features),
                check_finite=False)

            # Invert (Sigma_s + rho_0 * I) using Cholesky factorization
            sigma_s_rhos = inv_sigma_s + np.identity(self.features) * rho0
            (chol_sigma_s_rhos, lower_sigma_s_rhos) = scipy.linalg.cho_factor(
                sigma_s_rhos, check_finite=False)
            inv_sigma_s_rhos = scipy.linalg.cho_solve(
                (chol_sigma_s_rhos, lower_sigma_s_rhos),
                np.identity(self.features), check_finite=False)

            # Compute the sum of W_i^T * rho_i^-2 * X_i, and the sum of traces
            # of X_i^T * rho_i^-2 * X_i
            wt_invpsi_x = np.zeros((self.features, samples))
            trace_xt_invsigma2_x = 0.0
            for subject in range(subjects):
                wt_invpsi_x += (w[subject].T.dot(x[subject])) / rho2[subject]
                trace_xt_invsigma2_x += trace_xtx[subject] / rho2[subject]

            log_det_psi = np.sum(np.log(rho2) * voxels)

            # Update the shared response
            shared_response = sigma_s.dot(
                np.identity(self.features) - rho0 * inv_sigma_s_rhos).dot(
                    wt_invpsi_x)

            # M-step

            # Update Sigma_s and compute its trace
            sigma_s = (inv_sigma_s_rhos
                       + shared_response.dot(shared_response.T) / samples)
            trace_sigma_s = samples * np.trace(sigma_s)

            # Update each subject's mapping transform W_i and error variance
            # rho_i^2
            for subject in range(subjects):
                if self.verbose:
                    print('.'),  # noqa FIXME
                a_subject = x[subject].dot(shared_response.T)
                perturbation = np.zeros(a_subject.shape)
                np.fill_diagonal(perturbation, 0.001)
                u_subject, s_subject, v_subject = np.linalg.svd(
                    a_subject + perturbation, full_matrices=False)
                w[subject] = u_subject.dot(v_subject)
                rho2[subject] = trace_xtx[subject]
                rho2[subject] += -2 * np.sum(w[subject] * a_subject).sum()
                rho2[subject] += trace_sigma_s
                rho2[subject] /= samples * voxels[subject]

            if self.verbose:
                # Calculate and print the current log-likelihood for checking
                # convergence
                loglike = self._likelihood(
                    chol_sigma_s_rhos, log_det_psi, chol_sigma_s,
                    trace_xt_invsigma2_x, inv_sigma_s_rhos, wt_invpsi_x,
                    samples)
                print('Objective function %f' % loglike)  # noqa FIXME

        return sigma_s, w, mu, rho2, shared_response
