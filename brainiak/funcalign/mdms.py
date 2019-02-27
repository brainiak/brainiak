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
"""multi-dataset multi-subject (MDMS) SRM analysis

The implementations are based on the following publications:

.. [Zhang2018] "Transfer learning on fMRI datasets",
   H. Zhang, P.-H. Chen, P. Ramadge
   The 21st International Conference on Artificial Intelligence and
   Statistics (AISTATS), 2018.
   http://proceedings.mlr.press/v84/zhang18b/zhang18b.pdf
"""

# Authors: Hejia Zhang (Princeton Neuroscience Institute), 2018

import logging
import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.exceptions import NotFittedError
from mpi4py import MPI
import sys
import json
import os
import glob
from scipy import sparse as sp
import pickle as pkl

__all__ = [
    "DetMDMS",
    "MDMS"
]

logging.basicConfig(filename='mdms.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %('
                    'message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _init_w_transforms(voxels, features, random_states, datasets):
    """Initialize the mappings (W_s) for the MDMS with random orthogonal
    matrices.

    Parameters
    ----------

    voxels : dict of int, voxels[s] is number of voxels where s is the name
        of the subject.
        A dict with the number of voxels for each subject.

    features : int
        The number of features in the model.

    random_states : dict of `RandomState`s
        One `RandomState` instance per subject.

    datasets : a Dataset object
        The Dataset object containing datasets structures.

    comm : mpi4py.MPI.Intracomm
        The MPI communicator containing the data

    Returns
    -------

    w : dict of array, w[s] has shape=[voxels[s], features] where s is the
    name
        of the subject.
        The initialized orthogonal transforms (mappings) :math:`W_s` for each
        subject.


    Note
    ----

        This function assumes that the numpy random number generator was
        initialized.

        Not thread safe.
    """
    w = {}
    subjects = datasets.get_subjects_list()

    # Set Wi to a random orthogonal voxels by features matrix
    for subject in subjects:
        rnd_matrix = random_states[subject].random_sample((
            voxels[subject], features))
        q, r = np.linalg.qr(rnd_matrix)
        w[subject] = q
    return w


def _sanity_check(X, datasets, comm):
    """Check if the input data and datasets information have valid shape/
    configuration.

    Parameters
    ----------

    X : dict of list of 2D arrays or dict of dict of 2D arrays
        1) When it is a dict of list of 2D arrays:
            X[d] is a list of data of dataset d, where d is the name of the
            dataset.
            Element i in the list has shape=[voxels_i, samples_d]
            which is the fMRI data of the i'th subject in d.
        2) When it is a dict of dict of 2D arrays:
            X[d][s] has shape=[voxels_s, samples_d], which is the fMRI data
            of subject s in dataset d, where s is the name of the subject and
            d is the name of the dataset.

    datasets : a Dataset object
        The Dataset object containing datasets structures.

    comm : mpi4py.MPI.Intracomm
        The MPI communicator containing the data.


    Returns
    -------

    voxels_ : dict of int, voxels_[s] is number of voxels where s is the name
        of the subject.
        A dict with the number of voxels for each subject.

    samples_ : dict of int, samples_[d] is number of samples where d is the
    name of the dataset.
    A dict with the number of samples for each dataset.
    """
    # Check the number of subjects and all ranks have all datasets in the
    # Dataset object
    ds_list = datasets.get_datasets_list()
    for (ds, ns) in datasets.num_subj_dataset.items():
        if ns < 1:
            raise ValueError("Dataset {} should have positive "
                             "num_subj_dataset".format(ds))
        if ds not in X:
            raise ValueError("Dataset {} not in all ranks".format(ds))
        if X[ds] is not None and len(X[ds]) < ns:
            raise ValueError("Dataset {} does not have enough subjects: Need"
                             " equal to or more than {0:d} subjects but "
                             "got {0:d} to train the model."
                             .format(ds, ns, len(X[ds])))

    # Collect size information
    shape0, shape1, data_exist = _collect_size_information(X, datasets, comm)

    # Check if all required data appears once and only once
    # Also remove size information of data that is not in 'datasets'
    shape0, shape1 = _check_missing_data(datasets, shape0,
                                         shape1, data_exist)

    # Check if each subject has same number of voxels across different
    # datasets
    voxels_ = {}
    for subj in range(datasets.num_subj):
        all_vxs_tmp = [v[subj] for v in shape0.values() if v[subj] != 0]
        subj_name = datasets.idx_to_subject[subj]
        voxels_[subj_name] = np.min(all_vxs_tmp)
        if any([v != voxels_[subj_name] for v in all_vxs_tmp]):
            raise ValueError("Subject {} has different number of voxels "
                             "across datasets.".format(subj_name))

    # Check if all subjects have same number of TRs within the same dataset
    samples_ = {}
    for ds in ds_list:
        all_trs_tmp = [t for t in shape1[ds] if t != 0]
        samples_[ds] = np.min(all_trs_tmp)
        if any([t != samples_[ds] for t in all_trs_tmp]):
            raise ValueError("Different number of samples between subjects"
                             "in dataset {}.".format(ds))

    return voxels_, samples_


def _collect_size_information(X, datasets, comm):
    """Collect the shape of datasets and check if all data required are in X.

    Parameters
    ----------

    X : dict of list of 2D arrays or dict of dict of 2D arrays
        1) When it is a dict of list of 2D arrays:
            X[d] is a list of data of dataset d, where d is the name of the
            dataset.
            Element i in the list has shape=[voxels_i, samples_d]
            which is the fMRI data of the i'th subject in d.
        2) When it is a dict of dict of 2D arrays:
            X[d][s] has shape=[voxels_s, samples_d], which is the fMRI data
            of subject s in dataset d, where s is the name of the subject and
            d is the name of the dataset.

    datasets : a Dataset object
        The Dataset object containing datasets structures.

    comm : mpi4py.MPI.Intracomm
        The MPI communicator containing the data.


    Returns
    -------

    shape0 : dict of list, shape0[d] has shape [num_subj]
        Size of the 1st dimension of each 2D data array.

    shape1 : dict of list, shape1[d] has shape [num_subj]
        Size of the 2nd dimension of each 2D data array.

    data_exist : dict of list, data_exist[d] has shape [num_subj]
        How many times the same 2D data array appears in the dataset.
    """
    shape0, shape1, data_exist = {}, {}, {}
    ds_list = datasets.get_datasets_list()
    for ds in ds_list:
        # initialization
        shape0[ds] = np.zeros((datasets.num_subj,), dtype=np.int)
        shape1[ds] = np.zeros((datasets.num_subj,), dtype=np.int)
        data_exist[ds] = np.zeros((datasets.num_subj,), dtype=np.int)
        ds_idx = datasets.dataset_to_idx[ds]
        # collect size information of each dataset
        if X[ds] is not None:
            for subj in range(datasets.num_subj):
                if datasets.dok_matrix[subj, ds_idx] != 0:
                    if datasets.built_from_data:
                        idx = datasets.idx_to_subject[subj]
                        if idx not in X[ds]:
                            raise Exception('Subject {} in dataset {} is '
                                            'missing.'.format(idx, ds))
                    else:
                        idx = datasets.dok_matrix[subj, ds_idx] - 1
                        if len(X[ds]) <= idx:
                            raise ValueError("Dataset {} does not have "
                                             "enough subjects: Need more "
                                             "than {0:d} subjects but got "
                                             "{0:d} to train the model.".
                                             format(ds, idx, len(X[ds])))
                    if X[ds][idx] is not None:
                        assert_all_finite(X[ds][idx])
                        shape0[ds][subj] = X[ds][idx].shape[0]
                        shape1[ds][subj] = X[ds][idx].shape[1]
                        data_exist[ds][subj] = 1
        # reduce from all ranks
        shape0[ds] = comm.allreduce(shape0[ds], op=MPI.SUM)
        shape1[ds] = comm.allreduce(shape1[ds], op=MPI.SUM)
        data_exist[ds] = comm.allreduce(data_exist[ds], op=MPI.SUM)

    return shape0, shape1, data_exist


def _check_missing_data(datasets, shape0, shape1, data_exist):
    """Check if all required data appears once and only once.
    Also remove size information of data that is not in 'datasets'

    Parameters
    ----------

    datasets : a Dataset object
        The Dataset object containing datasets structures.

    shape0 : dict of list, shape0[d] has shape [num_subj]
        Size of the 1st dimension of each 2D data array.

    shape1 : dict of list, shape1[d] has shape [num_subj]
        Size of the 2nd dimension of each 2D data array.

    data_exist : dict of list, data_exist[d] has shape [num_subj]
        How many times the same 2D data array appears in the dataset.


    Returns
    -------

    shape0 : dict of list, shape0[d] has shape [num_subj]
        Size of the 1st dimension of each 2D data array.

    shape1 : dict of list, shape1[d] has shape [num_subj]
        Size of the 2nd dimension of each 2D data array.
    """
    ds_list = datasets.get_datasets_list()
    for ds in ds_list:
        ds_idx = datasets.dataset_to_idx[ds]
        for subj in range(datasets.num_subj):
            if datasets.dok_matrix[subj, ds_idx] != 0:
                if data_exist[ds][subj] == 0:
                    raise ValueError("Data of subject {} in dataset {} is "
                                     "missing."
                                     .format(datasets.idx_to_subject[subj],
                                             ds))
                elif data_exist[ds][subj] > 1:
                    raise ValueError("Data of subject {} in dataset {} "
                                     "appears more than once."
                                     .format(datasets.idx_to_subject[subj],
                                             ds))
            else:
                shape0[ds][subj] = 0
                shape1[ds][subj] = 0
    return shape0, shape1


class MDMS(BaseEstimator, TransformerMixin):
    """multi-dataset multi-subject (MDMS) SRM analysis

    Given multi-dataset multi-subject data, factorize it as a shared
    response S among all subjects per dataset and an orthogonal transform W
    across all datasets per subject:

    .. math::
       X_{ds} \\approx W_s S_d, \\forall s=1 \\dots N, \\forall d=1 \\dots M\\

    Parameters
    ----------

    n_iter : int, default: 10
        Number of iterations to run the algorithm.

    features : int, default: 50
        Number of features to compute.

    rand_seed : int, default: 0
        Seed for initializing the random number generator.

    comm : mpi4py.MPI.Intracomm
        The MPI communicator containing the data

    Attributes
    ----------

    w_ : dict of array, w_[s] has shape=[voxels_[s], features], where
        s is the name of the subject.
        The orthogonal transforms (mappings) for each subject.

    s_ : dict of array, s_[d] has shape=[features, samples_[d]], where
        d is the name of the dataset.
        The shared response for each dataset.

    voxels_ : dict of int, voxels_[s] is number of voxels where s is the name
        of the subject.
        A dict with the number of voxels for each subject.

    samples_ : dict of int, samples_[d] is number of samples where d is the
        name of the dataset.
        A dict with the number of samples for each dataset.

    sigma_s_ : dict of array, sigma_s_[d] has shape=[features, features]
        The covariance of the shared response Normal distribution for each
        dataset.

    mu_ : dict of array, mu_[s] has shape=[voxels_[s]] where s is the name
        of the subject.
        The voxel means over the samples in all datasets for each subject.

    rho2_ : dict of dict of float, rho2_[d][s] is a float, where d is the
        name of the dataset and s is the name of the subject.
        The estimated noise variance :math:`\\rho_{di}^2` for each subject
        in each dataset.

    comm : mpi4py.MPI.Intracomm
        The MPI communicator containing the data

    random_state_: `RandomState`
        Random number generator initialized using rand_seed

    Note
    ----

       The number of voxels may be different between subjects within a
       dataset and number of samples may be different between datasets.
       However, the number of samples must be the same across subjects
       within a dataset and number of voxels must be the same across
       datasets for the same subject.

       The probabilistic multi-dataset multi-subject model is approximated
       using the Expectation Maximization (EM) algorithm proposed in
       [Zhang2018]_. The implementation follows the optimizations published
       in [Anderson2016]_.

       The run-time complexity is :math:`O(I (V T K + V K^2 + K^3))` and the
       memory complexity is :math:`O(V T)` with I - the number of iterations,
       V - the sum of voxels from all subjects, T - the sum of samples from
       all datasets, and K - the number of features (typically, :math:`V \\
       gg T \\gg K`).
    """

    def __init__(self, n_iter=10, features=50, rand_seed=0,
                 comm=MPI.COMM_SELF):
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        self.comm = comm
        self.logger = logger
        return

    def fit(self, X, datasets, y=None):
        """Compute the probabilistic multi-dataset multi-subject (MDMS) SRM
        analysis

        Parameters
        ----------
        X : dict of list of 2D arrays or dict of dict of 2D arrays
            1) When it is a dict of list of 2D arrays:
                'datasets' must be defined in this case.
                X[d] is a list of data of dataset d, where d is the name of
                the dataset.
                Element i in the list has shape=[voxels_i, samples_d]
                which is the fMRI data of the i'th subject in d.
            2) When it is a dict of dict of 2D arrays:
                'datasets' can be omitted in this case.
                X[d][s] has shape=[voxels_s, samples_d], which is the fMRI
                data of subject s in dataset d, where s is the name of the
                subject and d is the name of the dataset.

        datasets : (optional) a Dataset object
            The Dataset object containing datasets structure.
            If you only have X, call datasets.build_from_data(X) with full
            data to infer datasets.

        y : not used
        """
        if self.comm.Get_rank() == 0:
            self.logger.info('Starting Probabilistic MDMS')

        # Check if datasets is initialized
        if datasets is None or datasets.matrix is None:
            raise NotFittedError('Dataset object is not initialized.')

        # Check X format
        if type(X) != dict:
            raise Exception('X should be a dict.')
        format_X = type(next(iter(X.values())))
        if format_X != dict and format_X != list:
            raise Exception('X should be a dict of dict of arrays or dict of'
                            ' list of arrays.')
        if format_X == list and (datasets.built_from_data is None or
                                 datasets.built_from_data):
            raise Exception("Argument 'datasets' must be defined and built "
                            "from JSON files when X is a dict of list of 2D "
                            "arrays. ")
        if format_X == dict:
            datasets.built_from_data = True
        for v in X.values():
            if type(v) != format_X:
                raise Exception('X should be a dict of dict of arrays or '
                                'dict of list of arrays.')

        self.voxels_, self.samples_ = _sanity_check(X, datasets, self.comm)

        # Run MDMS
        self.sigma_s_, self.w_, self.mu_, \
            self.rho2_, self.s_ = self._mdms(X, datasets)

        return self

    def transform(self, X, subjects, centered=True, y=None):
        """Use the model to transform new data to Shared Response space

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the new fMRI data of one
            subject

        subjects : list of string, element i is the name of subject of X[i]

        centered : bool, if the data in X is already centered.
            If centered = False, the voxel means computed during mode fitting
            will be subtracted before transformation.

        y : not used (as it is unsupervised learning)


        Returns
        -------
        s : list of 2D arrays, element i has shape=[features_i, samples_i]
            Shared responses from input data (X)
        """

        # Check if X and subjects have the same length
        if len(X) != len(subjects):
            raise ValueError("X and subjects must have the same length.")

        # Check if the model exist
        if not hasattr(self, 'w_'):
            raise NotFittedError("The model fit has not been run yet.")

        # Check if the subject exist in the fitted model and has the right
        # number of voxels
        for idx in range(len(X)):
            if subjects[idx] not in self.w_:
                raise NotFittedError("The model has not been fitted to "
                                     "subject {}.".format(subjects[idx]))
            if X[idx] is not None and (self.w_[subjects[idx]].
                                       shape[0] != X[idx].shape[0]):
                raise ValueError("{}-th element of data has inconsistent "
                                 "number of voxels with fitted model. Model"
                                 " has {} voxels while data has {}.".
                                 format(idx, self.w_[subjects[idx]].shape[0],
                                        X[idx].shape[0]))

        s = [None] * len(X)
        for idx in range(len(X)):
            if X[idx] is not None:
                if centered:
                    s[idx] = self.w_[subjects[idx]].T.dot(X[idx])
                else:
                    s[idx] = self.w_[subjects[idx]].T.\
                             dot(X[idx] - self.mu_[subjects[idx]][:, None])

        return s

    def _compute_mean(self, x, datasets):
        """Compute the mean of data.

        Parameters
        ----------
        x : dict of dict of array, x[d][s] has shape=[voxels[s], samples[d]]
            where d is the name of the dataset and s is the name of the
            subject.
            Demeaned data for each subject.

        datasets : a Dataset object
            The Dataset object containing datasets structures.

        Returns
        -------

        mu : dict of array, mu_[s] has shape=[voxels_[s]] where s is the
            name of the subject.
            The voxel means over the samples in all datasets for each
            subject.
        """
        # collect mean from each MPI worker
        weights = {}
        mu_tmp = {}
        for subj in datasets.subject_to_idx.keys():
            weights[subj], mu_tmp[subj] = {}, {}
            for ds in x.keys():
                if subj in x[ds]:
                    if x[ds][subj] is not None:
                        mu_tmp[subj][ds] = np.mean(x[ds][subj], 1)
                        weights[subj][ds] = x[ds][subj].shape[1]
                    else:
                        mu_tmp[subj][ds] = np.zeros((self.voxels_[subj],))
                        weights[subj][ds] = 0
        # collect mean from all MPI workers
        for subj in datasets.subject_to_idx.keys():
            for ds in mu_tmp[subj].keys():
                mu_tmp[subj][ds] = self.comm.allreduce(mu_tmp[subj][ds],
                                                       op=MPI.SUM)
                weights[subj][ds] = self.comm.allreduce(weights[subj][ds],
                                                        op=MPI.SUM)
        # compute final mean
        mu = {}
        for subj in datasets.subject_to_idx.keys():
            mu[subj] = np.zeros((self.voxels_[subj],))
            nsample = np.sum(list(weights[subj].values()))
            for ds in mu_tmp[subj].keys():
                mu[subj] += weights[subj][ds] * mu_tmp[subj][ds] / nsample
        return mu

    def _init_structures(self, data, datasets, ds_subj_list):
        """Initializes data structures for MDMS and preprocess the data.

        Parameters
        ----------
        data : dict of list of 2D arrays or dict of dict of 2D arrays
            1) When it is a dict of list of 2D arrays:
                'datasets' must be defined in this case.
                X[d] is a list of data of dataset d, where d is the name of
                the dataset.
                Element i in the list has shape=[voxels_i, samples_d]
                which is the fMRI data of the i'th subject in d.
            2) When it is a dict of dict of 2D arrays:
                'datasets' can be omitted in this case.
                X[d][s] has shape=[voxels_s, samples_d], which is the fMRI
                data of subject s in dataset d, where s is the name of the
                subject and d is the name of the dataset.

        datasets : a Dataset object
            The Dataset object containing datasets structures.

        ds_subj_list : dict of list of string, ds_subj_list[s] is a list
            of names of datasets with subject s, where s is the name
            of the subject.

        Returns
        -------
        x : dict of dict of array, x[d][s] has shape=[voxels[s], samples[d]]
            where d is the name of the dataset and s is the name of the
            subject.
            Demeaned data for each subject.

        mu : dict of array, mu_[s] has shape=[voxels_[s]] where s is the name
            of the subject.
            The voxel means over the samples in all datasets for each
            subject.

        rho2 : dict of dict of float, rho2_[d][s] is a float, where d is the
            name of the dataset and s is the name of the subject.
            The estimated noise variance :math:`\\rho_{di}^2` for each
            subject in each dataset.

        trace_xtx : dict of dict of float, trace_xtx[d][s] is a float, where
            d is the name of the dataset and s is the name of the subject.
            The squared Frobenius norm of the demeaned data in `x`.
        """
        x = {}
        rho2 = {}
        trace_xtx = {}

        # re-arrange data to x
        for ds_idx, ds in datasets.idx_to_dataset.items():
            x[ds] = {}
            for subj in range(datasets.num_subj):
                if datasets.dok_matrix[subj, ds_idx] != 0:
                    if datasets.built_from_data:
                        x[ds][datasets.
                              idx_to_subject[subj]] =\
                              data[ds][datasets.idx_to_subject[subj]]
                    else:
                        x[ds][datasets.
                              idx_to_subject[subj]] =\
                              data[ds][datasets.dok_matrix[subj, ds_idx]-1]
        del data

        # compute mean
        mu = self._compute_mean(x, datasets)

        # subtract mean from x and compute trace_xtx, initialize rho2
        for ds in x.keys():
            rho2[ds], trace_xtx[ds] = {}, {}
            for subj in x[ds].keys():
                rho2[ds][subj] = 1
                if x[ds][subj] is not None:
                    x[ds][subj] -= mu[subj][:, None]
                    trace_xtx[ds][subj] = np.sum(x[ds][subj] ** 2)
                else:
                    trace_xtx[ds][subj] = 0

        # broadcast values in trace_xtx to all ranks
        for subj in ds_subj_list.keys():
            for ds in ds_subj_list[subj]:
                trace_xtx[ds][subj] = self.comm.allreduce(
                    trace_xtx[ds][subj], op=MPI.SUM)

        return x, mu, rho2, trace_xtx

    def _likelihood(self, chol_sigma_s_rhos, log_det_psi, chol_sigma_s,
                    trace_xt_invsigma2_x, inv_sigma_s_rhos, wt_invpsi_x,
                    samples):
        """Calculate the log-likelihood function of one dataset


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
        log_det = (np.log(np.diag(chol_sigma_s_rhos) ** 2).sum() +
                   log_det_psi + np.log(np.diag(chol_sigma_s) ** 2).sum())
        loglikehood = -0.5 * samples * log_det - 0.5 * trace_xt_invsigma2_x
        loglikehood += 0.5 * np.trace(
            wt_invpsi_x.T.dot(inv_sigma_s_rhos).dot(wt_invpsi_x))

        # + const --> -0.5*nTR*sum(voxel[subjects])*math.log(2*math.pi)

        return loglikehood

    @staticmethod
    def _update_transform_subject(Xi, S):
        """Updates the mappings `W_i` for one subject.

        Parameters
        ----------

        Xi : array, shape=[voxels, timepoints]
            The fMRI data :math:`X_i` for aligning the subject.

        S : array, shape=[features, timepoints]
            The shared response.

        Returns
        -------

        Wi : array, shape=[voxels, features]
            The orthogonal transform (mapping) :math:`W_i` for the subject.
        """
        A = Xi.dot(S.T)
        # Solve the Procrustes problem
        U, _, V = np.linalg.svd(A, full_matrices=False)
        return U.dot(V)

    def transform_subject(self, X, dataset):
        """Transform a new subject using the existing model.
        The subject is assumed to have received equivalent stimulation
        of some dataset in the fitted model.

        Parameters
        ----------

        X : 2D array, shape=[voxels, timepoints]
            The fMRI data of the new subject.

        dataset : string, name of the dataset in the fitted model that
            has the same stimulation as the new subject

        Returns
        -------

        w : 2D array, shape=[voxels, features]
            Orthogonal mapping `W_{new}` for new subject

        """
        # Check if the model exist
        if not hasattr(self, 'w_'):
            raise NotFittedError("The model fit has not been run yet.")

        # Check if the dataset is in the model
        if dataset not in self.s_:
            raise NotFittedError("Dataset {} is not in the model yet."
                                 .format(dataset))

        # Check the number of TRs in the subject
        if X.shape[1] != self.s_[dataset].shape[1]:
            raise ValueError("The number of timepoints(TRs) does not match"
                             " the one in the model.")

        w = self._update_transform_subject(X, self.s_[dataset])

        return w

    def _compute_shared_response(self, x, w, shared_response, sigma_s,
                                 rho2, trace_xtx, ds_list, subj_ds_list,
                                 ds_rank, rank):
        """Part of E step in MDMS. Update shared response and sigma_s for
        each dataset.

        Parameters
        ----------

        x : dict of dict of array, x[d][s] has shape=[voxels[s], samples[d]]
            where d is the name of the dataset and s is the name of the
            subject.
            Demeaned data for each subject.

        w : dict of array, w[s] has shape=[voxels_[s], features] where s is
            the name of the subject.
            The orthogonal transforms (mappings) :math:`W_s` for each
            subject.

        shared_response : dict of array, shared_response[d] has
            shape=[features, samples_[d]] where d is the name of the dataset.
            The shared response for each dataset.

        sigma_s : dict of array, sigma_s[d] has shape=[features, features]
            where d is the name of dataset.
            The covariance :math:`\\Sigma_s` of the shared response Normal
            distribution for each dataset.

        rho2 : dict of dict of float, rho2[d][s] is a float, where d is the
            name of the dataset and s is the name of the subject.
            The estimated noise variance :math:`\\rho2{di}^2` for each
            subject in each dataset.

        trace_xtx : dict of dict of float, trace_xtx[d][s] is a float, where
            d is the name of the dataset and s is the name of the subject.
            The squared Frobenius norm of the demeaned data in `x`.

        ds_list : list of string, names of all datasets

        subj_ds_list : dict of list of string, subj_ds_list[d] is a list
            of names of subjects in dataset d, where d is the name
            of the dataset.

        ds_rank : set of string, name of datasets assigned to be processed
            by this rank.

        rank : int, the current MPI rank

        Returns
        -------

        shared_response : dict of array, shared_response[d] has
            shape=[features, samples_[d]] where d is the name of the dataset.
            The shared response for each dataset.

        trace_sigma_s : dict of float, trace of sigma_s for each dataset.

        sigma_s : dict of array, sigma_s[d] has shape=[features, features]
            where d is the name of dataset.
            The covariance :math:`\\Sigma_s` of the shared response Normal
            distribution for each dataset.
        """
        loglike = 0.
        other_ds = set(ds_list) - ds_rank

        # for multi-thread computation
        chol_sigma_s = {ds: np.zeros((self.features, self.features)) for ds
                        in other_ds}
        chol_sigma_s_rhos = {ds: np.zeros((self.features, self.features))
                             for ds in other_ds}
        inv_sigma_s_rhos = {ds: np.zeros((self.features, self.features))
                            for ds in other_ds}
        rho0 = {ds: 0.0 for ds in other_ds}
        wt_invpsi_x = {ds: np.zeros((self.features, self.samples_[ds]))
                       for ds in ds_list}
        trace_xt_invsigma2_x = {ds: 0.0 for ds in ds_list}
        trace_sigma_s = {ds: 0 for ds in ds_list}

        # iterate through all ds in this rank
        for ds in ds_rank:
            # Sum the inverted the rho2 elements for computing W^T *
            # Psi^-1 * W
            rho0[ds] = np.sum([1/v for v in rho2[ds].values()])

            # Invert Sigma_s[ds] using Cholesky factorization
            (chol_sigma_s[ds], lower_sigma_s) = scipy.linalg.cho_factor(
                sigma_s[ds], check_finite=False)
            inv_sigma_s = scipy.linalg.cho_solve(
                        (chol_sigma_s[ds], lower_sigma_s),
                        np.identity(self.features),
                        check_finite=False)

            # Invert (Sigma_s[ds] + rho_0 * I) using Cholesky
            # factorization
            sigma_s_rhos = inv_sigma_s + np.identity(self.features) *\
                rho0[ds]
            (chol_sigma_s_rhos[ds], lower_sigma_s_rhos) = \
                scipy.linalg.cho_factor(sigma_s_rhos, check_finite=False)
            inv_sigma_s_rhos[ds] = scipy.linalg.cho_solve(
                (chol_sigma_s_rhos[ds], lower_sigma_s_rhos),
                np.identity(self.features), check_finite=False)

        # collect info from all ranks
        chol_sigma_s = {ds: self.comm.
                        allreduce(chol_sigma_s[ds], op=MPI.SUM)
                        for ds in ds_list}
        chol_sigma_s_rhos = {ds: self.comm.
                             allreduce(chol_sigma_s_rhos[ds], op=MPI.SUM)
                             for ds in ds_list}
        inv_sigma_s_rhos = {ds: self.comm.
                            allreduce(inv_sigma_s_rhos[ds], op=MPI.SUM)
                            for ds in ds_list}

        # Compute the sum of W_i^T * rho_i^-2 * X_i, and the sum of
        # traces of X_i^T * rho_i^-2 * X_i
        for ds in ds_list:
            for subj in subj_ds_list[ds]:
                if x[ds][subj] is not None:
                    wt_invpsi_x[ds] += (w[subj].T.dot(x[ds][subj])) /\
                                        rho2[ds][subj]
                    trace_xt_invsigma2_x[ds] += trace_xtx[ds][subj] /\
                        rho2[ds][subj]

        # collect data from all ranks
        for ds in ds_list:
            wt_invpsi_x[ds] = self.comm.allreduce(wt_invpsi_x[ds],
                                                  op=MPI.SUM)
            trace_xt_invsigma2_x[ds] = self.comm.allreduce(
                                    trace_xt_invsigma2_x[ds], op=MPI.SUM)

        # compute shared response and Sigma_s of ds in this rank
        for ds in ds_rank:
            log_det_psi = np.sum([np.log(rho2[ds][subj]) * self.
                                  voxels_[subj] for subj
                                  in rho2[ds]])

            # Update the shared response
            shared_response[ds] = sigma_s[ds].dot(
                            np.identity(self.features) - rho0[ds] *
                            inv_sigma_s_rhos[ds]).dot(
                            wt_invpsi_x[ds])

            # Update Sigma_s and compute its trace
            sigma_s[ds] = (inv_sigma_s_rhos[ds]
                           + shared_response[ds].dot(
                           shared_response[ds].T) /
                           self.samples_[ds])
            trace_sigma_s[ds] = self.samples_[ds] *\
                np.trace(sigma_s[ds])

            # calculate log likelihood to check convergence
            loglike += self._likelihood(
                chol_sigma_s_rhos[ds], log_det_psi, chol_sigma_s[ds],
                trace_xt_invsigma2_x[ds], inv_sigma_s_rhos[ds],
                wt_invpsi_x[ds], self.samples_[ds])

        for ds in other_ds:
            shared_response[ds] = np.zeros((self.features,
                                            self.samples_[ds]))
            sigma_s[ds] = np.zeros((self.features, self.features))
            trace_sigma_s[ds] = 0

        # collect parameters from all ranks
        for ds in ds_list:
            shared_response[ds] = self.comm.allreduce(
                                    shared_response[ds], op=MPI.SUM)
            trace_sigma_s[ds] = self.comm.allreduce(
                                trace_sigma_s[ds], op=MPI.SUM)
            sigma_s[ds] = self.comm.allreduce(sigma_s[ds], op=MPI.SUM)

        # collect loglikelihood
        loglike = self.comm.allreduce(loglike, op=MPI.SUM)
        if rank == 0 and self.logger.isEnabledFor(logging.INFO):
            self.logger.info('Objective function %f' % loglike)

        return shared_response, trace_sigma_s, sigma_s

    def _compute_w(self, x, shared_response, ds_subj_list, rank):
        """Compute transformation matrix W for each subject.

        Parameters
        ----------

        x : dict of dict of array, x[d][s] has shape=[voxels[s], samples[d]]
            where d is the name of the dataset and s is the name of the
            subject.
            Demeaned data for each subject.

        shared_response : dict of array, shared_response[d] has
            shape=[features, samples_[d]] where d is the name of the dataset.
            The shared response for each dataset.

        ds_subj_list : dict of list of string, ds_subj_list[s] is a list
            of names of datasets with subject s, where s is the name
            of the subject.

        rank : int, the current MPI rank

        Returns
        -------

        w : dict of array, w[s] has shape=[voxels_[s], features] where s is
            the name of the subject.
            The orthogonal transforms (mappings) :math:`W_s` for each
            subject.
        """
        w = {}
        for subj in ds_subj_list.keys():
            # update w
            a_subject = np.zeros((self.voxels_[subj], self.features))
            # use x data from all ranks
            for ds in ds_subj_list[subj]:
                if x[ds][subj] is not None:
                    a_subject += x[ds][subj].dot(shared_response[ds].T)
            # collect a_subject from all ranks
            a_subject = self.comm.allreduce(a_subject, op=MPI.SUM)
            # compute w in one rank and broadcast
            if rank == 0:
                perturbation = np.zeros(a_subject.shape)
                np.fill_diagonal(perturbation, 0.0001)
                u_subject, _, v_subject = np.linalg.svd(
                    a_subject + perturbation, full_matrices=False)
                w[subj] = u_subject.dot(v_subject)
            else:
                w[subj] = None
            w[subj] = self.comm.bcast(w[subj], root=0)
        return w

    def _compute_rho2(self, x, shared_response, w, ds_subj_list, ds_list,
                      trace_xtx, trace_sigma_s, rank):
        """Compute the estimated noise variance for each subject in each
        dataset.

        Parameters
        ----------

        x : dict of dict of array, x[d][s] has shape=[voxels[s], samples[d]]
            where d is the name of the dataset and s is the name of the
            subject.
            Demeaned data for each subject.

        shared_response : dict of array, shared_response[d] has
            shape=[features, samples_[d]] where d is the name of the dataset.
            The shared response for each dataset.

        w : dict of array, w[s] has shape=[voxels_[s], features] where s is
            the name of the subject.
            The orthogonal transforms (mappings) :math:`W_s` for each
            subject.

        ds_subj_list : dict of list of string, ds_subj_list[s] is a list
            of names of datasets with subject s, where s is the name
            of the subject.

        ds_list : list of string, names of all datasets

        trace_xtx : dict of dict of float, trace_xtx[d][s] is a float, where
            d is the name of the dataset and s is the name of the subject.
            The squared Frobenius norm of the demeaned data in `x`.

        trace_sigma_s : dict of int, trace of sigma_s for each dataset

        rank : int, the current MPI rank

        Returns
        -------

        rho2 : dict of dict of float, rho2_[d][s] is a float, where d is the
            name of the dataset and s is the name of the subject.
            The estimated noise variance :math:`\\rho_{di}^2` for each
            subject in each dataset.
        """
        # update rho2
        rho2 = {d: {} for d in ds_list}
        for subj in ds_subj_list.keys():
            # compute trace_xtws_tmp of data in this rank
            trace_xtws_tmp = {}
            for ds in ds_subj_list[subj]:
                if x[ds][subj] is not None:
                    trace_xtws_tmp[ds] = np.trace(x[ds][subj].T.dot(
                                    w[subj]).dot(shared_response[ds]))
                else:
                    trace_xtws_tmp[ds] = 0.0
            # collect trace_xtws_tmp in all ranks
            for ds in ds_subj_list[subj]:
                trace_xtws_tmp[ds] = self.comm.allreduce(
                                    trace_xtws_tmp[ds], op=MPI.SUM)
            # compute rho2
            for ds in ds_subj_list[subj]:
                if rank == 0:
                    rho2[ds][subj] = trace_xtx[ds][subj]
                    rho2[ds][subj] += -2 * trace_xtws_tmp[ds]
                    rho2[ds][subj] += trace_sigma_s[ds]
                    rho2[ds][subj] /= self.samples_[ds] *\
                        self.voxels_[subj]
                else:
                    rho2[ds][subj] = None
            # broadcast to all ranks
            for ds in ds_subj_list[subj]:
                rho2[ds][subj] = self.comm.bcast(rho2[ds][subj], root=0)

        return rho2

    def _mdms(self, data, datasets):
        """Expectation-Maximization algorithm for fitting the probabilistic
        MDMS.

        Parameters
        ----------

        data : dict of list of 2D arrays or dict of dict of 2D arrays
            1) When it is a dict of list of 2D arrays:
                'datasets' must be defined in this case.
                X[d] is a list of data of dataset d, where d is the name of
                the dataset.
                Element i in the list has shape=[voxels_i, samples_d]
                which is the fMRI data of the i'th subject in d.
            2) When it is a dict of dict of 2D arrays:
                'datasets' can be omitted in this case.
                X[d][s] has shape=[voxels_s, samples_d], which is the fMRI
                data of subject s in dataset d, where s is the name of the
                subject and d is the name of the dataset.

        datasets : a Dataset object
            The Dataset object containing datasets structures.

        Returns
        -------

        sigma_s : dict of array, sigma_s[d] has shape=[features, features]
            where d is the name of dataset.
            The covariance :math:`\\Sigma_s` of the shared response Normal
            distribution for each dataset.

        w : dict of array, w[s] has shape=[voxels_[s], features] where s is
            the name of the subject.
            The orthogonal transforms (mappings) :math:`W_s` for each
            subject.

        mu : dict of array, mu[s] has shape=[voxels_[s]] where s is the name
            of the subject.
            The voxel means :math:`\\mu_i` over the samples in all datasets
            for each subject.

        rho2 : dict of dict of float, rho2[d][s] is a float, where d is the
            name of the dataset and s is the name of the subject.
            The estimated noise variance :math:`\\rho2{di}^2` for each
            subject in each dataset.

        s : dict of array, s[d] has shape=[features, samples_[d]] where d is
            the name of the dataset.
            The shared response for each dataset.
        """
        # get information from datasets structures
        ds_list, subj_list = datasets.get_datasets_list(),\
            datasets.get_subjects_list()
        subj_ds_list = datasets.subjects_in_dataset_all()
        ds_subj_list = datasets.datasets_with_subject_all()

        # initialize random states
        self.random_state_ = np.random.RandomState(self.rand_seed)
        random_states = {subj_list[i]: np.random.RandomState(
                        self.random_state_.randint(2 ** 32))
                        for i in range(datasets.num_subj)}

        # assign ds to different ranks for parallel computing
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        ds_rank = set()
        if datasets.num_dataset <= size:
            if rank < datasets.num_dataset:
                ds_rank.add(ds_list[rank])
        else:
            ds_rank_len = datasets.num_dataset // size
            if rank != size - 1:
                ds_rank.update(set(ds_list[ds_rank_len*rank:
                                           ds_rank_len*(rank+1)]))
            else:
                ds_rank.update(set(ds_list[ds_rank_len*rank:]))

        # Initialization step: initialize the outputs with initial values
        # and trace_xtx with the ||X_i||_F^2 of each subject in each dataset.
        w = _init_w_transforms(self.voxels_, self.features, random_states,
                               datasets)
        x, mu, rho2, trace_xtx = self._init_structures(data, datasets,
                                                       ds_subj_list)
        del data

        shared_response, sigma_s = {}, {}
        for ds in ds_list:
            shared_response[ds] = np.zeros((self.features,
                                            self.samples_[ds]))
            if ds in ds_rank:
                sigma_s[ds] = np.identity(self.features)
            else:
                sigma_s[ds] = np.zeros((self.features, self.features))

        # Main loop of the algorithm (run)
        for iteration in range(self.n_iter):
            if rank == 0:
                self.logger.info('Iteration %d' % (iteration + 1))

            # E-step and some M-step: update shared_response and sigma_s of
            # each dataset
            shared_response, trace_sigma_s, sigma_s = self.\
                _compute_shared_response(x, w, shared_response, sigma_s,
                                         rho2, trace_xtx, ds_list,
                                         subj_ds_list, ds_rank, rank)

            # The rest of M-step: update w and rho2
            # Update each subject's mapping transform W_i and error variance
            # rho_di^2
            w = self._compute_w(x, shared_response, ds_subj_list, rank)
            rho2 = self._compute_rho2(x, shared_response, w, ds_subj_list,
                                      ds_list, trace_xtx, trace_sigma_s,
                                      rank)

        return sigma_s, w, mu, rho2, shared_response

    def save(self, file):
        """Save the MDMS object to a file (as pickle)

        Parameters
        ----------

        file : The name (including full path) of the file that the object
            will be saved to.

        Returns
        -------

        None

        Note
        ----

        The MPI communicator cannot be saved, so it will not be saved. When
            restored, self.comm will be initialized to MPI.COMM_SELF

        """
        # get attributes from object
        variables = self.__dict__.keys()
        data = {k: getattr(self, k) for k in variables}
        # remove attributes that cannot be pickled
        del data['comm']
        del data['logger']
        if 'random_state_' in data:
            del data['random_state_']
        # save attributes to file
        with open(file, 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        self.logger.info('MDMS object saved to {}.'.format(file))
        return

    def restore(self, file):
        """Restore the MDMS object from a (pickle) file

        Parameters
        ----------

        file : The name (including full path) of the file that the object
            will be restored from.

        Returns
        -------

        None

        Note
        ----

        The MPI communicator cannot be saved, so self.comm is initialized to
            MPI.COMM_SELF

        """
        # get attributes from file
        with open(file, 'rb') as f:
            data = pkl.load(f)
        # set attributes to object
        for (k, v) in data.items():
            setattr(self, k, v)
        # set attributes that were not pickled
        self.comm = MPI.COMM_SELF
        self.random_state_ = np.random.RandomState(self.rand_seed)
        self.logger = logger
        self.logger.info('MDMS object restored from {}.'.format(file))
        return


class DetMDMS(BaseEstimator, TransformerMixin):
    """Deterministic multi-dataset multi-subject (MDMS)

    Given multi-dataset multi-subject data, factorize it as a shared
    response S among all subjects per dataset and an orthogonal transform W
    across all datasets per subject.

    Parameters
    ----------

    n_iter : int, default: 10
        Number of iterations to run the algorithm.

    features : int, default: 50
        Number of features to compute.

    rand_seed : int, default: 0
        Seed for initializing the random number generator.

    comm : mpi4py.MPI.Intracomm
        The MPI communicator containing the data

    Attributes
    ----------

    w_ : dict of array, w_[s] has shape=[voxels_[s], features], where
        s is the name of the subject.
        The orthogonal transforms (mappings) for each subject.

    s_ : dict of array, s_[d] has shape=[features, samples_[d]], where
        d is the name of the dataset.
        The shared response for each dataset.

    voxels_ : dict of int, voxels_[s] is number of voxels where s is the name
        of the subject.
        A dict with the number of voxels for each subject.

    samples_ : dict of int, samples_[d] is number of samples where d is the
        name of the dataset.
        A dict with the number of samples for each dataset.

    mu_ : dict of array, mu_[s] has shape=[voxels_[s]] where s is the name
        of the subject.
        The voxel means over the samples in all datasets for each subject.

    random_state_: `RandomState`
        Random number generator initialized using rand_seed

    comm : mpi4py.MPI.Intracomm
        The MPI communicator containing the data

    Note
    ----

        The number of voxels may be different between subjects within a
        dataset and number of samples may be different between datasets.
        However, the number of samples must be the same across subjects
        within a dataset and number of voxels must be the same across
        datasets for the same subject.

        The probabilistic multi-dataset multi-subject model is approximated
        using the Block Coordinate Descent (BCD) algorithm proposed in
        [Zhang2018]_.

        The run-time complexity is :math:`O(I (V T K + V K^2))` and the
        memory complexity is :math:`O(V T)` with I - the number of
        iterations, V - the sum of number of voxels from all subjects, T -
        the sum of number of samples from all datasets, K - the number of
        features (typically, :math:`V \\gg T \\gg K`), and
        N - the number of subjects.
    """

    def __init__(self, n_iter=10, features=50, rand_seed=0,
                 comm=MPI.COMM_SELF):
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        self.comm = comm
        self.logger = logger
        return

    def fit(self, X, datasets, demean=True, y=None):
        """Compute the Deterministic Shared Response Model

        Parameters
        ----------

        X : dict of list of 2D arrays or dict of dict of 2D arrays
            1) When it is a dict of list of 2D arrays:
                'datasets' must be defined in this case.
                X[d] is a list of data of dataset d, where d is the name of
                the dataset.
                Element i in the list has shape=[voxels_i, samples_d]
                which is the fMRI data of the i'th subject in d.
            2) When it is a dict of dict of 2D arrays:
                'datasets' can be omitted in this case.
                X[d][s] has shape=[voxels_s, samples_d], which is the fMRI
                data of subject s in dataset d, where s is the name of the
                subject and d is the name of the dataset.

        datasets : (optional) a Dataset object
            The Dataset object containing datasets structure.
            If you only have X, call datasets.build_from_data(X) with full
            data to infer datasets.

        demean : (optional) If True, compute voxel means for each subject
            and subtract from data. If False, voxel means are set to zero
            and data values are not changed.

        y : not used
        """
        if self.comm.Get_rank() == 0:
            self.logger.info('Starting Deterministic SRM')

        # Check if datasets is initialized
        if datasets is None or datasets.matrix is None:
            raise NotFittedError('Dataset object is not initialized.')

        # Check X format
        if type(X) != dict:
            raise Exception('X should be a dict.')
        format_X = type(next(iter(X.values())))
        if format_X != dict and format_X != list:
            raise Exception('X should be a dict of dict of arrays or dict of'
                            ' list of arrays.')
        if format_X == list and (datasets.built_from_data is None or
                                 datasets.built_from_data):
            raise Exception("Argument 'datasets' must be defined and built "
                            "from json files when X is a dict of list of 2D "
                            "arrays. ")
        if format_X == dict:
            datasets.built_from_data = True
        for v in X.values():
            if type(v) != format_X:
                raise Exception('X should be a dict of dict of arrays or '
                                'dict of list of arrays.')

        self.voxels_, self.samples_ = _sanity_check(X, datasets, self.comm)

        # Run MDMS
        self.w_, self.s_, self.mu_ = self._mdms(X, datasets, demean)

        return self

    def transform(self, X, subjects, centered=True, y=None):
        """Use the model to transform new data to Shared Response space

        Parameters
        ----------

        X : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the new fMRI data of one
            subject.

        subjects : list of string, element i is the name of subject of X[i]

        centered : (optional) bool, if the data in X is already centered.
            If centered = False, the voxel means computed during mode
            fitting will be subtracted before transformation.

        y : not used (as it is unsupervised learning)

        Returns
        -------

        s : list of 2D arrays, element i has shape=[features_i, samples_i]
            Shared responses from input data (X)
        """

        # Check if X and subjects have the same length
        if len(X) != len(subjects):
            raise ValueError("X and subjects must have the same length.")

        # Check if the model exist
        if not hasattr(self, 'w_'):
            raise NotFittedError("The model fit has not been run yet.")

        # Check if the subject exist in the fitted model and has the right
        # number of voxels
        for idx in range(len(X)):
            if subjects[idx] not in self.w_:
                raise NotFittedError("The model has not been fitted to "
                                     "subject {}.".format(subjects[idx]))
            if X[idx] is not None and (self.w_[subjects[idx]].shape[0] !=
                                       X[idx].shape[0]):
                raise ValueError("{}-th element of data has inconsistent "
                                 "number of voxels with fitted model. Model "
                                 "has {} voxels while data has {}."
                                 .format(idx, self.w_[subjects[idx]].
                                         shape[0], X[idx].shape[0]))

        if not centered and self.mu_ is None:
            raise Exception('Mean values are not computed during model '
                            'fitting. Please center the data to be '
                            'transformed beforehand.')

        s = [None] * len(X)
        for idx in range(len(X)):
            if X[idx] is not None:
                if centered:
                    s[idx] = self.w_[subjects[idx]].T.dot(X[idx])
                else:
                    s[idx] = self.w_[subjects[idx]].T.\
                             dot(X[idx] - self.mu_
                                 [subjects[idx]][:, None])

        return s

    def _compute_mean(self, x, datasets):
        """Compute the mean of data.

        Parameters
        ----------
        x : dict of dict of array, x[d][s] has shape=[voxels[s], samples[d]]
            where d is the name of the dataset and s is the name of the
            subject.
            Demeaned data for each subject.

        datasets : a Dataset object
            The Dataset object containing datasets structures.

        Returns
        -------

        mu : dict of array, mu_[s] has shape=[voxels_[s]] where s is the
            name of the subject.
            The voxel means over the samples in all datasets for each
            subject.
        """
        # collect mean from each MPI worker
        weights = {}
        mu_tmp = {}
        for subj in datasets.subject_to_idx.keys():
            weights[subj], mu_tmp[subj] = {}, {}
            for ds in x.keys():
                if subj in x[ds]:
                    if x[ds][subj] is not None:
                        mu_tmp[subj][ds] = np.mean(x[ds][subj], 1)
                        weights[subj][ds] = x[ds][subj].shape[1]
                    else:
                        mu_tmp[subj][ds] = np.zeros((
                                        self.voxels_[subj],))
                        weights[subj][ds] = 0

        # collect mean from all MPI workers
        for subj in datasets.subject_to_idx.keys():
            for ds in mu_tmp[subj].keys():
                mu_tmp[subj][ds] = self.comm.allreduce(
                                    mu_tmp[subj][ds], op=MPI.SUM)
                weights[subj][ds] = self.comm.allreduce(
                                    weights[subj][ds], op=MPI.SUM)

        # compute final mean
        mu = {}
        for subj in datasets.subject_to_idx.keys():
            mu[subj] = np.zeros((self.voxels_[subj],))
            nsample = np.sum(list(weights[subj].values()))
            for ds in mu_tmp[subj].keys():
                mu[subj] += weights[subj][ds] * mu_tmp[subj][ds] /\
                            nsample
        return mu

    def _preprocess_data(self, data, datasets, demean):
        """Preprocess and demean the data.

        Parameters
        ----------
        data : dict of list of 2D arrays or dict of dict of 2D arrays
            1) When it is a dict of list of 2D arrays:
                'datasets' must be defined in this case.
                X[d] is a list of data of dataset d, where d is the name of
                the dataset.
                Element i in the list has shape=[voxels_i, samples_d]
                which is the fMRI data of the i'th subject in d.
            2) When it is a dict of dict of 2D arrays:
                'datasets' can be omitted in this case.
                X[d][s] has shape=[voxels_s, samples_d], which is the fMRI
                data of subject s in dataset d, where s is the name of the
                subject and d is the name of the dataset.

        datasets : a Dataset object
            The Dataset object containing datasets structures.

        demean : If True, compute voxel means for each subject
            and subtract from data. If False, voxel means are set to zero
            and data values are not changed.

        Returns
        -------
        x : dict of dict of array, x[d][s] has shape=[voxels[s], samples[d]]
            where d is the name of the dataset and s is the name of the
            subject.
            Demeaned data for each subject.

        mu : dict of array, mu_[s] has shape=[voxels_[s]] where s is the
            name of the subject.
            The voxel means over the samples in all datasets for each
            subject.
        """
        x = {}

        # re-arrange data to x
        for ds_idx, ds in datasets.idx_to_dataset.items():
            x[ds] = {}
            for subj in range(datasets.num_subj):
                if datasets.dok_matrix[subj, ds_idx] != 0:
                    if datasets.built_from_data:
                        x[ds][datasets.idx_to_subject[subj]] = \
                            data[ds][datasets.idx_to_subject[subj]]
                    else:
                        x[ds][datasets.idx_to_subject[subj]] = \
                            data[ds][datasets.dok_matrix[subj, ds_idx]-1]
        del data

        # compute mean
        if demean:
            mu = self._compute_mean(x, datasets)
            # subtract mean from x
            for ds in x.keys():
                for subj in x[ds].keys():
                    if x[ds][subj] is not None:
                        x[ds][subj] -= mu[subj][:, None]
        else:
            mu = None

        return x, mu

    def _objective_function(self, data, subj_ds_list, w, s, num_sample):
        """Calculate the objective function

        Parameters
        ----------

        data : dict of dict of array, x[d][s] has shape=[voxels[s],
            samples[d]] where d is the name of the dataset and s is the name
            of the subject.
            Demeaned data for each subject.

        subj_ds_list : dict of list of string, subj_ds_list[d] is a list
            of names of subjects in dataset d, where d is the name
            of the subject.

        w : dict of array, w[s] has shape=[voxels_[s], features], where
            s is the name of the subject.
            The orthogonal transforms (mappings) for each subject.

        s : dict of array, s[d] has shape=[features, samples_[d]], where
            d is the name of the dataset.
            The shared response for each dataset.

        num_sample : int, total number of samples across all datasets and
        datasets

        Returns
        -------

        objective : float
            The objective function value.

        Note
        ----

        In the multi nodes mode where data is scattered in different nodes,
        objective needs to be reduced (summed) afterwards.
        """
        objective = 0.0
        for ds in subj_ds_list.keys():
            for subj in subj_ds_list[ds]:
                if data[ds][subj] is not None:
                    objective += np.linalg.norm(data[ds][subj] -
                                                w[subj].dot(s[ds]),
                                                'fro') ** 2

        return 0.5 * objective / num_sample

    def _compute_shared_response(self, data, subj_ds_list, w):
        """ Compute the shared response S of all datasets

        Parameters
        ----------

        data : dict of dict of array, data[d][s] has shape=[voxels[s],
            samples[d]] where d is the name of the dataset and s is the name
            of the subject
            Demeaned data for each subject.

        subj_ds_list : dict of list of string, subj_ds_list[d] is a list
            of names of subjects in dataset d, where d is the name
            of the dataset.

        w : dict of array, w[s] has shape=[voxels_[s], features] where
            s is the name of the subject.
            The orthogonal transforms (mappings) for each subject.

        Returns
        -------

        s : dict of array, s[d] has shape=[features, samples_[d]] where
            d is the name of the dataset.
            The shared response for each dataset.

        Note
        ----

        In the multi nodes mode where data is scattered in different nodes,
        s needs to be gathered afterwards.

        To get the final s, the returned s[d] needs to be devided by number
        of subjects in dataset d.
        """
        s = {}
        for ds in subj_ds_list.keys():
            s[ds] = np.zeros((self.features, self.samples_[ds]))
            for subj in subj_ds_list[ds]:
                if data[ds][subj] is not None:
                    s[ds] += w[subj].T.dot(data[ds][subj])
        return s

    @staticmethod
    def _update_transform_subject(Xi, S):
        """Updates the mappings `W_i` for one subject.

        Parameters
        ----------

        Xi : array, shape=[voxels, timepoints]
            The fMRI data :math:`X_i` for aligning the subject.

        S : array, shape=[features, timepoints]
            The shared response.

        Returns
        -------

        Wi : array, shape=[voxels, features]
            The orthogonal transform (mapping) :math:`W_i` for the subject.
        """
        A = Xi.dot(S.T)
        # Solve the Procrustes problem
        U, _, V = np.linalg.svd(A, full_matrices=False)
        return U.dot(V)

    def transform_subject(self, X, dataset):
        """Transform a new subject using the existing model.
        The subject is assumed to have received equivalent stimulation
        of some dataset in the fitted model.

        Parameters
        ----------

        X : 2D array, shape=[voxels, timepoints]
            The fMRI data of the new subject.

        dataset : string, name of the dataset in the fitted model that
            has the same stimulation as the new subject

        Returns
        -------

        w : 2D array, shape=[voxels, features]
            Orthogonal mapping `W_{new}` for new subject

        """
        # Check if the model exist
        if not hasattr(self, 'w_'):
            raise NotFittedError("The model fit has not been run yet.")

        # Check if the dataset is in the model
        if dataset not in self.s_:
            raise NotFittedError("Dataset {} is not in the model yet."
                                 .format(dataset))

        # Check the number of TRs in the subject
        if X.shape[1] != self.s_[dataset].shape[1]:
            raise ValueError("The number of timepoints(TRs) does not match "
                             "the one in the model.")

        w = self._update_transform_subject(X, self.s_[dataset])

        return w

    def _compute_w_subj(self, x, ds_subj_list, shared_response, rank):
        """ Compute the transformation matrix W of all subjects

        Parameters
        ----------

        x : dict of dict of array, x[d][s] has shape=[voxels[s],
            samples[d]] where d is the name of the dataset and s is the name
            of the subject
            Demeaned data for each subject.

        ds_subj_list : dict of list of string, ds_subj_list[s] is a list
            of names of datasets with subject s, where s is the name
            of the subject.

        shared_response : dict of array, shared_response[d] has
            shape=[features, samples_[d]] where d is the name of the dataset.
            The shared response for each dataset.

        rank: int, current MPI rank

        Returns
        -------

        w : dict of array, w[d] has shape=[voxels_[s], features] where
            s is the name of the subject.
            The transformation matrix for each subject.

        """
        w = {}
        for subj in ds_subj_list.keys():
            a_subject = np.zeros((self.voxels_[subj], self.features))
            # use x data from all ranks
            for ds in ds_subj_list[subj]:
                if x[ds][subj] is not None:
                    a_subject += x[ds][subj].dot(shared_response[ds].T)
            # collect a_subject from all ranks
            a_subject = self.comm.allreduce(a_subject, op=MPI.SUM)
            # compute w in one rank and broadcast
            if rank == 0:
                perturbation = np.zeros(a_subject.shape)
                np.fill_diagonal(perturbation, 0.0001)
                u_subject, _, v_subject = np.linalg.svd(
                    a_subject + perturbation, full_matrices=False)
                w[subj] = u_subject.dot(v_subject)
            else:
                w[subj] = None
            w[subj] = self.comm.bcast(w[subj], root=0)
        return w

    def _mdms(self, data, datasets, demean):
        """Block Coordinate Descent algorithm for fitting the deterministic
        MDMS.

        Parameters
        ----------

        data : dict of list of 2D arrays or dict of dict of 2D arrays
            1) When it is a dict of list of 2D arrays:
                data[d] is a list of data of dataset d, where d is the name
                of the dataset.
                Element i in the list has shape=[voxels_i, samples_[d]]
                which is the fMRI data of the i'th subject in dataset d.
            2) When it is a dict of dict of 2D arrays:
                data[d][s] has shape=[voxels_[s], samples_[d]], which is the
                fMRI data of subject s in dataset d, where s is the name of
                the subject and d is the name of the dataset.

        datasets : a Dataset object
            The Dataset object containing datasets structure.

        demean : If True, compute voxel means for each subject
            and subtract from data. If False, voxel means are set to zero
            and data values are not changed.

        Returns
        -------

        w : dict of array, w[s] has shape=[voxels_[s], features], where
            s is the name of the subject.
            The orthogonal transforms (mappings) for each subject.

        s : dict of array, s[d] has shape=[features, samples_[d]], where
            d is the name of the dataset.
            The shared response for each dataset.
        """

        # get information from datasets structure
        ds_list, subj_list = datasets.get_datasets_list(),\
            datasets.get_subjects_list()
        subj_ds_list = datasets.subjects_in_dataset_all()
        ds_subj_list = datasets.datasets_with_subject_all()
        num_sample = np.sum([datasets.num_subj_dataset[ds] *
                             self.samples_[ds] for ds in ds_list])

        # initialize random states
        self.random_state_ = np.random.RandomState(self.rand_seed)
        random_states = {subj_list[i]: np.random.RandomState(
                        self.random_state_.randint(2 ** 32))
                        for i in range(datasets.num_subj)}

        rank = self.comm.Get_rank()

        # Initialization step:
        # 1) preprocess data
        # 2) initialize the outputs with initial values

        w = _init_w_transforms(self.voxels_, self.features, random_states,
                               datasets)
        x, mu = self._preprocess_data(data, datasets, demean)
        del data
        # compute shared_response from data in this rank
        shared_response = self._compute_shared_response(x, subj_ds_list, w)
        # collect shared_response data from all ranks
        for ds in ds_list:
            shared_response[ds] = self.comm.allreduce(shared_response[ds],
                                                      op=MPI.SUM)
            shared_response[ds] /= datasets.num_subj_dataset[ds]

        if self.logger.isEnabledFor(logging.INFO):
            # Calculate the current objective function value
            objective = self._objective_function(x, subj_ds_list, w,
                                                 shared_response, num_sample)
            objective = self.comm.allreduce(objective, op=MPI.SUM)
            if rank == 0:
                self.logger.info('Objective function %f' % objective)

        # Main loop of the algorithm
        for iteration in range(self.n_iter):
            if rank == 0:
                self.logger.info('Iteration %d' % (iteration + 1))

            # Update each subject's mapping transform W_s:
            w = self._compute_w_subj(x, ds_subj_list, shared_response, rank)

            # Update the each dataset's shared response S_d:
            # compute shared_response from data in this rank
            shared_response = self._compute_shared_response(
                            x, subj_ds_list, w)
            # collect shared_response data from all ranks
            for ds in ds_list:
                shared_response[ds] = self.comm.allreduce(
                                        shared_response[ds], op=MPI.SUM)
                shared_response[ds] /= datasets.num_subj_dataset[ds]

            if self.logger.isEnabledFor(logging.INFO):
                # Calculate the current objective function value
                objective = self._objective_function(x, subj_ds_list, w,
                                                     shared_response,
                                                     num_sample)
                objective = self.comm.allreduce(objective, op=MPI.SUM)
                if rank == 0:
                    self.logger.info('Objective function %f' % objective)

        return w, shared_response, mu

    def save(self, file):
        """Save the DetMDMS object to a file (as pickle)

        Parameters
        ----------

        file : The name (including full path) of the file that the object
            will be saved to.

        Returns
        -------

        None

        Note
        ----

        The MPI communicator cannot be saved, so it will not be saved. When
            restored, self.comm will be initialized to MPI.COMM_SELF

        """
        # get attributes from object
        variables = self.__dict__.keys()
        data = {k: getattr(self, k) for k in variables}
        # remove attributes that cannot be pickled
        del data['comm']
        del data['logger']
        if 'random_state_' in data:
            del data['random_state_']
        # save attributes to file
        with open(file, 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        self.logger.info('DetMDMS object saved to {}.'.format(file))
        return

    def restore(self, file):
        """Restore the DetMDMS object from a (pickle) file

        Parameters
        ----------

        file : The name (including full path) of the file that the object
            will be restored from.

        Returns
        -------

        None

        Note
        ----

        The MPI communicator cannot be saved, so self.comm is initialized to
            MPI.COMM_SELF

        """
        # get attributes from file
        with open(file, 'rb') as f:
            data = pkl.load(f)
        # set attributes to object
        for (k, v) in data.items():
            setattr(self, k, v)
        # set attributes that were not pickled
        self.comm = MPI.COMM_SELF
        self.random_state_ = np.random.RandomState(self.rand_seed)
        self.logger = logger
        self.logger.info('DetMDMS object restored from {}.'.format(file))
        return


class Dataset(object):
    """Datasets structure organizer

    Given multi-dataset multi-subject data or JSON files with subject names
    in each dataset, infer datasets structure in different formats, such as
    a graph where each dataset is a node and each edge is number of shared
    subjects between the two datasets.

    .. math::
       X_{ds} \\approx W_s S_d, \\forall s=1 \\dots N, \\forall d=1 \\dots M\\

    This organizer is used in the MDMS or DetMDMS [Zhang2018]_ and can also
    be used as a standalone datasets organizer.


    Parameters
    ----------

    file : (optional) string, default: None
        JSON file name (including full path) or folder name with JSON files.

        Each JSON file should contain a dict or a list of dict where each
        dict has information of one dataset. Each dict must have 'dataset',
        'num_of_subj', and 'subjects' where 'dataset' is the name of the
        dataset, 'num_of_subj' is the number of subjects in the dataset, and
        'subjects' is a list of strings with names of subjects in the
        dataset in the same order as in the dataset. All datasets in all
        JSON files will be added to the organizer.

        Example of a JSON file:
        [{'dataset':'MyData','num_of_subj':3,'subjects':
        ['Adam','Bob','Carol']},
        {'dataset':'MyData2','num_of_subj':2,'subjects':['Tom','Bob']}]

    data : (optional) dict of dict of 2D array, default: None
        Multi-dataset multi-subject data used to build the organizer.

        data[d][s] has shape=[voxels[s], samples[d]], where d is the name of
        the dataset and s is the name of the subject.


    Attributes
    ----------

    num_subj : int,
        Total number of subjects

    num_dataset : int,
        Total number of datasets

    dataset_to_idx : dict of int, dataset_to_idx[d] is the column index of
        dataset d in self.matrix, where d is the name of the dataset.
        Dataset name to column index of matrix, 0-indexed

    idx_to_dataset : dict of string, idx_to_dataset[i] is name of the
        dataset mapped to the i'th column in self.matrix.
        Column index of metrix to dataset name, 0-indexed

    subject_to_idx : dict of int, subject_to_idx[s] is the row index of
        subject s in self.matrix, where s is the name of the subject.
        Subject name to row index of matrix, 0-indexed

    idx_to_subject : dict of string, idx_to_subject[i] is name of the
        subject mapped to the i'th row in self.matrix.
        Row index to subject name, 0-indexed

    connected : list of list of string, each element is a list of name of
        connected datasets (datasets can be connected through shared
        subjects).

    num_graph : int,
        Number of connected dataset graphs
        If 1, then all datasets are connected.

    adj_matrix : 2D csc sparse matrix of shape [num_dataset, num_dataset],
        Weighted adjacency matrix of all datasets, where each node is a
        dataset and weights on edges are number of shared subjects between
        the two datasets.
        Mapping between dataset name and dataset index is in
        self.dataset_to_idx.

    num_subj_dataset : dict of int, num_subj_dataset[d] is an int where d is
        the name of a dataset.
        Number of subjects of each dataset

    subj_in_dataset : dict of list of string, subj_in_dataset[d] is a list
        of name of subjects in dataset d in the same order as in d, where d
        is the name of a dataset. If any subject is removed from the
        organizer, the name will be replaced with None as a placeholder.
        Name of subjects in each dataset

    matrix : 2D coo sparse matrix of shape [num_subj, num_dataset],
        Dataset-subject membership matrix.
        If built from JSON files, subject self.idx_to_subject[i] is the
        self.matrix[i,j]'th subject in self.idx_to_dataset[j], 1-indexed
        If built from multi-dataset multi-subject data, self.matrix[i,j] = 1
        if subject self.idx_to_subject[i] is in dataset self.idx_to_dataset
        [j].

    dok_matrix : 2D dok sparse matrix of shape [num_subj, num_dataset],
        Dataset-subject membership matrix.
        It has the same content as self.matrix, but in Dictionary Of Keys
        format for fast access of individual access.

    built_from_data : bool,
        If the object is built from multi-dataset multi-subject data
        If True, the object is built from data; if False, it is built from
        JSON files.

    Note
    ----

    Example usage can be found in BrainIAK MDMS example jupyter notebook.

    """

    def __init__(self, file=None, data=None):
        self.num_subj = 0
        self.num_dataset = 0
        self.dataset_to_idx = {}
        self.idx_to_dataset = {}
        self.subject_to_idx = {}
        self.idx_to_subject = {}
        self.connected = []
        self.num_graph = 0
        self.adj_matrix = None
        self.num_subj_dataset = {}
        self.subj_in_dataset = {}
        self.matrix = None
        self.dok_matrix = None
        self.built_from_data = None

        if file is not None and data is not None:
            raise Exception('Dataset object can only be built from data OR '
                            'JSON files.')

        if file is not None:
            self.add(file)

        if data is not None:
            self.build_from_data(data)
        return

    def add(self, file):
        """Add JSON file(s) to the organizer

        Parameters
        ----------

        file : string, default: None
        JSON file name (including full path) or folder name with JSON files.

        Each JSON file should contain a dict or a list of dict where each
        dict has information of one dataset. Each dict must have 'dataset',
        'num_of_subj', and 'subjects' where 'dataset' is the name of the
        dataset, 'num_of_subj' is the number of subjects in the dataset, and
        'subjects' is a list of strings with names of subjects in the
        dataset in the same order as in the dataset. All datasets in all
        JSON files will be added to the organizer. If some datasets are
        already in the organizer, the information of those datasets will be
        replaced with this new version.

        Example of a JSON file:
        [{'dataset':'MyData','num_of_subj':3,'subjects':
        ['Adam','Bob','Carol']},
        {'dataset':'MyData2','num_of_subj':2,'subjects':['Tom','Bob']}]

        Returns
        -------

        None
        """
        # sanity check
        if self.built_from_data is not None and self.built_from_data:
            raise Exception('This Dataset object was already initialized '
                            'with fMRI datasets.')

        # file can be json file name or folder name
        # parse json filenames
        if os.path.isfile(file):
            # file
            files = [file]
        elif os.path.isdir(file):
            # path
            files = glob.glob(os.path.join(file, '*.json'))
            if not files:
                raise Exception('The path must contain JSON files.')
        else:
            raise Exception('Argument must be a filename or a path.')

        mem = []  # collect info of all datasets
        for f in files:
            tmp = json.load(open(f, 'r'))
            if type(tmp) == list:
                # multiple datasets
                mem.extend(tmp)
            elif type(tmp) == dict:
                # one dataset
                mem.append(tmp)
            else:
                raise Exception('JSON file must be in list or dict format.')

        self._add_mem(mem)  # add the information read from JSON files
        return

    def build_from_data(self, data):
        """Use multi-dataset multi-subject data to initialize the organizer

        Parameters
        ----------

        data : dict of dict of 2D array
        Multi-dataset multi-subject data used to build the organizer.
        data[d][s] has shape=[voxels[s], samples[d]], where d is the name of
        the dataset and s is the name of the subject.

        Returns
        -------

        None
        """
        # sanity check
        if self.built_from_data is not None and not self.built_from_data:
            raise Exception('This Dataset object was already initialized '
                            'with JSON files.')

        # find out which datasets and subjects are in the data
        if not type(data) == dict:
            raise Exception('To build Dataset object from data, data must be'
                            ' a dict of dict where data[d][s] is the fMRI '
                            'data of dataset d and subject s.')
        datasets = set(data.keys())
        subjects = set()
        for ds in data:
            if not type(data[ds]) == dict:
                raise Exception('To build Dataset object from data, data '
                                'must be a dict of dict where data[d][s] is '
                                'the fMRI data of dataset d and subject s.')
            subjects.update(set(data[ds].keys()))

        # set attributes
        self.num_dataset = len(datasets)
        self.num_subj = len(subjects)

        for idx, subj in enumerate(subjects):
            self.subject_to_idx[subj] = idx
            self.idx_to_subject[idx] = subj
        for idx, ds in enumerate(datasets):
            self.dataset_to_idx[ds] = idx
            self.idx_to_dataset[idx] = ds

        for ds in datasets:
            self.num_subj_dataset[ds] = len(data[ds])
            self.subj_in_dataset[ds] = list(data[ds].keys())

        # fill in sparse matrix
        coo_data, row, col = [], [], []
        for ds in datasets:
            col_idx = self.dataset_to_idx[ds]
            for subj in data[ds].keys():
                coo_data.append(1)
                col.append(col_idx)
                row.append(self.subject_to_idx[subj])
        self.matrix = sp.coo_matrix((coo_data, (row, col)),
                                    shape=(self.num_subj, self.num_dataset))
        self.dok_matrix = self.matrix.todok(copy=True)
        # compute connectivity
        self._compute_connected()

        self.built_from_data = True
        return

    def remove_dataset(self, datasets):
        """Remove some datasets from the organizer

        Parameters
        ----------

        datasets : set or list of string, each element is name of a dataset
        Name of datasets to be removed

        Returns
        -------

        removed_subjects : list of string, each element is name of a subject
        Name of subjects removed because of the removal of datasets.
        """
        # sanity check
        for ds in datasets:
            if ds not in self.dataset_to_idx:
                raise Exception('Dataset ' + ds + ' does not exist.')

        # extract data from the sparse matrix
        data, row, col = self.matrix.data.tolist(), self.matrix.row.\
            tolist(), self.matrix.col.tolist()

        # remove datasets from data
        data, row, col, subj_to_check = self._remove_datasets_from_data(
                                        datasets, data, row, col)

        # if all datasets are removed
        if not data:
            removed_subjects = list(self.subject_to_idx.keys())
            self.reset()
            return removed_subjects

        # find subjects not in any dataset
        removed_subjects = []
        for subj in subj_to_check:
            if not self.subject_to_idx[subj] in row:
                removed_subjects.append(subj)

        # re-arrange subject indices
        row = self._remove_subjects_by_re_indexing(removed_subjects, row)

        # re-arrange dataset indices
        col = self._remove_datasets_by_re_indexing(datasets, col)

        # re-construct the matrix
        self.matrix = sp.coo_matrix((data, (row, col)),
                                    shape=(self.num_subj, self.num_dataset))
        self.dok_matrix = self.matrix.todok(copy=True)

        # compute connectivity
        self._compute_connected()

        return removed_subjects

    def remove_subject(self, subjects):
        """Remove some subjects from the organizer

        Parameters
        ----------

        subjects : set or list of string, each element is name of a subject
        Name of subjects to be removed

        Returns
        -------

        removed_datasets : list of string, each element is name of a dataset
        Name of datasets removed because of the removal of subjects.
        """
        # sanity check
        for subj in subjects:
            if subj not in self.subject_to_idx:
                raise Exception('Subject ' + subj + ' does not exist.')

        # extract data from the sparse matrix
        data, row, col = self.matrix.data.tolist(), self.matrix.row.\
            tolist(), self.matrix.col.tolist()

        # remove subjects from data
        data, row, col = self._remove_subjects_from_data(
                        subjects, data, row, col)

        # if all subjects are removed
        if not data:
            removed_datasets = list(self.dataset_to_idx.keys())
            self.reset()
            return removed_datasets

        # find datasets without any subject
        removed_datasets = []
        for (k, v) in self.num_subj_dataset.items():
            if not v:
                removed_datasets.append(k)
        for k in removed_datasets:
            del self.num_subj_dataset[k]  # remove from num_subj_dataset
            del self.subj_in_dataset[k]

        # re-arrange subject indices
        row = self._remove_subjects_by_re_indexing(subjects, row)

        # re-arrange dataset indices
        col = self._remove_datasets_by_re_indexing(removed_datasets, col)

        # re-construct the matrix
        self.matrix = sp.coo_matrix((data, (row, col)),
                                    shape=(self.num_subj, self.num_dataset))
        self.dok_matrix = self.matrix.todok(copy=True)

        # compute connectivity
        self._compute_connected()

        return removed_datasets

    def num_shared_subjects_between_datasets(self, ds1, ds2):
        """Get number of shared subjects (subjects in both ds1 and ds2)
            between two datasets (ds1 and ds2)

        Parameters
        ----------

        ds1, ds2 : string,
        Name of two datasets

        Returns
        -------

        num_shared : int,
        Number of shared subjects between ds1 and ds2
        """
        # sanity check
        for ds in [ds1, ds2]:
            if ds not in self.dataset_to_idx:
                raise Exception('Dataset ' + ds + 'does not exist.')
        # find number of shared subjects
        idx1, idx2 = self.dataset_to_idx[ds1], self.dataset_to_idx[ds2]
        return self.adj_matrix[idx1, idx2]

    def shared_subjects_between_datasets(self, ds1, ds2):
        """Get name of shared subjects (subjects in both ds1 and ds2)
            between two datasets (ds1 and ds2)

        Parameters
        ----------

        ds1, ds2 : string,
        Name of two datasets

        Returns
        -------

        shared : list of string,
        Name of subjects shared between ds1 and ds2
        """
        # sanity check
        for ds in [ds1, ds2]:
            if ds not in self.dataset_to_idx:
                raise Exception('Dataset ' + ds + 'does not exist.')
        if self.matrix is None:
            raise Exception('Dataset object not initialized.')
        # find shared subjects
        matrix_csc = self.matrix.tocsc(copy=True)
        # indices of subjects in ds1
        subj1 = set(matrix_csc[:, self.dataset_to_idx[ds1]].indices)
        # indices of subjects in ds2
        subj2 = set(matrix_csc[:, self.dataset_to_idx[ds2]].indices)
        return [self.idx_to_subject[subj] for subj in
                subj1.intersection(subj2)]

    def datasets_with_subject(self, subj):
        """Get all datasets with some subject ('subj')

        Parameters
        ----------

        subj : string,
        Name of the subject

        Returns
        -------

        datasets : list of string,
        Name of datasets with subject 'subj'
        """
        # sanity check
        if subj not in self.subject_to_idx:
            raise Exception('Subject ' + subj + 'does not exist.')
        if self.matrix is None:
            raise Exception('Dataset object not initialized.')
        # find datasets with subject
        matrix_csr = self.matrix.tocsr(copy=True)
        indices = matrix_csr[self.subject_to_idx[subj], :].indices
        return [self.idx_to_dataset[ds] for ds in indices]

    def datasets_with_subject_all(self):
        """For each subject, get a list of datasets with that subject

        Parameters
        ----------

        None

        Returns
        -------

        ds_subj_list : dict of list of string, ds_subj_list[s] is a list
        where s is the name of a subject.
        List of datasets with subject s for each subject s
        """
        if self.matrix is None:
            raise Exception('Dataset object not initialized.')
        ds_subj_list = {}
        matrix_csr = self.matrix.tocsr(copy=True)
        for subj in range(self.num_subj):
            subj_name = self.idx_to_subject[subj]
            indices = matrix_csr[subj, :].indices
            ds_subj_list[subj_name] = [self.idx_to_dataset[ds]
                                       for ds in indices]
        return ds_subj_list

    def subjects_in_dataset(self, dataset):
        """Get all subjects in some dataset ('dataset')

        Parameters
        ----------

        dataset : string,
        Name of the dataset

        Returns
        -------

        subjects : list of string,
        Name of subjects in dataset 'dataset'
        """
        # sanity check
        if dataset not in self.dataset_to_idx:
            raise Exception('Dataset ' + dataset + 'does not exist.')
        if self.matrix is None:
            raise Exception('Dataset object not initialized.')
        # find subjects in dataset
        matrix_csc = self.matrix.tocsc(copy=True)
        indices = matrix_csc[:, self.dataset_to_idx[dataset]].indices
        return [self.idx_to_subject[subj] for subj in indices]

    def subjects_in_dataset_all(self):
        """For each dataset, get a list of subjects in that dataset

        Parameters
        ----------

        None

        Returns
        -------

        subj_ds_list : dict of list of string, subj_ds_list[d] is a list
        where d is the name of a dataset.
        List of subjects in dataset d for each dataset d
        """
        if self.matrix is None:
            raise Exception('Dataset object not initialized.')
        subj_ds_list = {}
        matrix_csc = self.matrix.tocsc(copy=True)
        for ds in range(self.num_dataset):
            ds_name = self.idx_to_dataset[ds]
            indices = matrix_csc[:, ds].indices
            subj_ds_list[ds_name] = [self.idx_to_subject[subj]
                                     for subj in indices]
        return subj_ds_list

    def get_subjects_list(self):
        """Get a list of all subjects in the organizer

        Parameters
        ----------

        None

        Returns
        -------

        subj_list : list of string,
        Name of all subjects in the organizer
        """
        return list(self.subject_to_idx.keys())

    def get_datasets_list(self):
        """Get a list of all datasets in the organizer

        Parameters
        ----------

        None

        Returns
        -------

        ds_list : list of string,
        Name of all datasets in the organizer
        """
        return list(self.dataset_to_idx.keys())

    def reset(self):
        """Reset all attributes in the organizer

        Parameters
        ----------

        None

        Returns
        -------

        None
        """
        self.num_subj = 0
        self.num_dataset = 0
        self.dataset_to_idx = {}
        self.idx_to_dataset = {}
        self.subject_to_idx = {}
        self.idx_to_subject = {}
        self.connected = []
        self.num_graph = 0
        self.adj_matrix = None
        self.num_subj_dataset = {}
        self.subj_in_dataset = {}
        self.matrix = None
        self.adj_matrix = None
        self.built_from_data = None
        return

    def save(self, file):
        """Save the Dataset object to a file (as pickle)

        Parameters
        ----------

        file : The name (including full path) of the file that the object
            will be saved to.

        Returns
        -------

        None
        """
        # get attributes from object
        variables = self.__dict__.keys()
        data = {k: getattr(self, k) for k in variables}
        # save attributes to file
        with open(file, 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
        self.logger.info('Dataset object saved to {}.'.format(file))
        return

    def restore(self, file):
        """Restore the Dataset object from a (pickle) file

        Parameters
        ----------

        file : The name (including full path) of the file that the object
            will be restored from.

        Returns
        -------

        None
        """
        # get attributes from file
        with open(file, 'rb') as f:
            data = pkl.load(f)
        # set attributes to object
        for (k, v) in data.items():
            setattr(self, k, v)
        self.logger.info('Dataset object restored from {}.'.format(file))
        return

    def _compute_connected(self):
        """Compute the weighted adjacency matrix and connectivity

        Parameters
        ----------

        None

        Returns
        -------

        None
        """
        # build the weighted adjacency matrix (how many shared subjects
        # between each pair of datasets)
        matrix_csc = self.matrix.tocsc(copy=True)
        row, col, data = [], [], []
        for i in range(self.num_dataset):
            for j in range(i+1, self.num_dataset):
                tmp = matrix_csc[:, i].multiply(matrix_csc[:, j]).nnz
                if tmp != 0:
                    row.extend([i, j])
                    col.extend([j, i])
                    data.extend([tmp, tmp])
        self.adj_matrix = sp.csc_matrix((data, (row, col)),
                                        shape=(self.num_dataset,
                                               self.num_dataset))

        self._compute_num_connect_graph()
        return

    def _compute_num_connect_graph(self):
        """Compute which datasets are connected

        Parameters
        ----------

        None

        Returns
        -------

        None
        """
        # find out which datasets are connected
        not_connected = set(range(self.num_dataset))
        connected = []
        dq = set()
        for idx in range(self.num_dataset):
            if idx in not_connected:
                tmp = []
                dq.add(idx)
                while dq:
                    n = dq.pop()
                    not_connected.remove(n)
                    tmp.append(n)
                    for neighbor in self.adj_matrix[:, n].indices:
                        if neighbor in not_connected:
                            dq.add(neighbor)
                    if not dq:
                        connected.append(tmp)

        # convert connected datasets from idx to dataset names
        self.connected = []
        for idx, graph in enumerate(connected):
            self.connected.append([])
            for node in graph:
                self.connected[idx].append(self.idx_to_dataset[node])

        # count number of connected graphs
        self.num_graph = len(self.connected)
        return

    def _add_mem(self, mem):
        """Add information from JSON files to the organizer

        Parameters
        ----------

        mem : list of dict, information from JSON files

        Returns
        -------

        None
        """
        # separate datasets into new datasets and datasets to update
        new_ds, new_sub, replace_ds, ds_dict = set(), set(), set(), {}
        for m in mem:
            # sanity check
            err_case = [m['num_of_subj'] <= 0,
                        m['num_of_subj'] != len(m['subjects']),
                        m['dataset'] in new_ds or m['dataset'] in replace_ds,
                        len(m['subjects']) != len(set(m['subjects']))]
            err_msg = ['Number of subjects in dataset {} must be positive.'.
                       format(m['dataset']),
                       'Number of subjects in dataset {} does not agree.'.
                       format(m['dataset']),
                       'Dataset {} appears more than once.'.
                       format(m['dataset']),
                       'Dataset {} has duplicate subjects.'.
                       format(m['dataset'])]

            for err, msg in zip(err_case, err_msg):
                if err:
                    raise Exception(msg)

            # if the dataset is already in the matrix
            if m['dataset'] in self.dataset_to_idx:
                replace_ds.add(m['dataset'])
            else:
                new_ds.add(m['dataset'])

            # save subjects info into a dict
            ds_dict[m['dataset']] = m['subjects']

            # add new subjects in this dataset
            for subj in m['subjects']:
                if subj not in self.subject_to_idx:
                    new_sub.add(subj)

        # add number of subjects info if mem passes all the sanity check
        for m in mem:
            self.num_subj_dataset[m['dataset']] = m['num_of_subj']

        del mem

        # construct or update the matrix
        if self.matrix is None:
            # construct a new matrix
            self._construct_matrix(new_ds, new_sub, ds_dict)
        else:
            # add new datasets
            self._add_new_dataset(new_ds, new_sub, ds_dict)
            if replace_ds:
                # replace some old datasets
                self._replace_dataset(replace_ds, ds_dict)
            self._compute_connected()

        self.built_from_data = False

        return

    def _construct_matrix(self, new_ds, new_sub, ds_dict):
        """Initialize the organizer with some datasets and subjects

        Parameters
        ----------

        new_ds : set or list of string,
        Name of all new datasets to add

        new_sub : set or list of string,
        Name of all new subjects to add

        ds_dict : dict of list of string, ds_dict[d] is a list of subject
        names in dataset d in the same order as in the dataset, where d is
        the name of the dataset.

        Returns
        -------

        None
        """
        # fill in datasets and subjects info
        self.num_subj = len(new_sub)
        self.num_dataset = len(new_ds)
        for idx, subj in enumerate(new_sub):
            self.subject_to_idx[subj] = idx
            self.idx_to_subject[idx] = subj
        for idx, ds in enumerate(new_ds):
            self.dataset_to_idx[ds] = idx
            self.idx_to_dataset[idx] = ds

        # fill in sparse matrix
        data, row, col = [], [], []
        for ds in new_ds:
            self.subj_in_dataset[ds] = ds_dict[ds]
            col_idx = self.dataset_to_idx[ds]
            for idx, subj in enumerate(ds_dict[ds]):
                data.append(idx+1)
                col.append(col_idx)
                row.append(self.subject_to_idx[subj])
        self.matrix = sp.coo_matrix((data, (row, col)),
                                    shape=(self.num_subj, self.num_dataset))
        self.dok_matrix = self.matrix.todok(copy=True)

        # compute connectivity
        self._compute_connected()
        return

    def _add_new_dataset(self, new_ds, new_sub, ds_dict):
        """Add some new datasets into the organizer when the organizer was
        already initialized and the new datasets are not in it yet.

        Parameters
        ----------

        new_ds : set or list of string,
        Name of all new datasets to add

        new_sub : set or list of string,
        Name of all new subjects to add

        ds_dict : dict of list of string, ds_dict[d] is a list of subject
        names in dataset d in the same order as in the dataset, where d is
        the name of the dataset.

        Returns
        -------

        None
        """
        # fill in new datasets and subjects info
        for idx, subj in enumerate(new_sub):
            self.subject_to_idx[subj] = self.num_subj + idx
            self.idx_to_subject[self.num_subj + idx] = subj
        for idx, ds in enumerate(new_ds):
            self.dataset_to_idx[ds] = self.num_dataset + idx
            self.idx_to_dataset[self.num_dataset + idx] = ds
        self.num_subj += len(new_sub)
        self.num_dataset += len(new_ds)

        # fill in sparse matrix
        data, row, col = self.matrix.data.tolist(), self.matrix.row.\
            tolist(), self.matrix.col.tolist()
        for ds in new_ds:
            self.subj_in_dataset[ds] = ds_dict[ds]
            col_idx = self.dataset_to_idx[ds]
            for idx, subj in enumerate(ds_dict[ds]):
                data.append(idx+1)
                col.append(col_idx)
                row.append(self.subject_to_idx[subj])
        self.matrix = sp.coo_matrix((data, (row, col)),
                                    shape=(self.num_subj, self.num_dataset))
        self.dok_matrix = self.matrix.todok(copy=True)
        return

    def _replace_dataset(self, replace_ds, ds_dict):
        """Replace information of some datasets with information in ds_dict
        assuming those datasets are already in the organizer

        Parameters
        ----------

        replace_ds : set or list of string,
        Name of all datasets to replace

        ds_dict : dict of list of string, ds_dict[d] is a list of subject
        names in dataset d in the same order as in the dataset, where d is
        the name of the dataset.

        Returns
        -------

        None
        """
        # extract data from the sparse matrix
        data, row, col = self.matrix.data.tolist(), self.matrix.row.\
            tolist(), self.matrix.col.tolist()

        # remove data of datasets to be replaced from the coo sparse matrix
        data, row, col, subj_to_check = self._remove_datasets_from_data(
                                    replace_ds, data, row, col)

        # add data of datasets to replace
        for ds in replace_ds:
            self.subj_in_dataset[ds] = ds_dict[ds]
            col_idx = self.dataset_to_idx[ds]
            for idx, subj in enumerate(ds_dict[ds]):
                data.append(idx+1)
                col.append(col_idx)
                row.append(self.subject_to_idx[subj])
                subj_to_check.discard(subj)

        # finalize subj to remove (subjects not in any datasets)
        subj_to_remove = []
        for subj in subj_to_check:
            if not self.subject_to_idx[subj] in row:
                subj_to_remove.append(subj)

        # remove those subjects and re-arrange subject indices
        row = self._remove_subjects_by_re_indexing(subj_to_remove, row)

        # re-construct the matrix
        self.matrix = sp.coo_matrix((data, (row, col)),
                                    shape=(self.num_subj, self.num_dataset))
        self.dok_matrix = self.matrix.todok(copy=True)
        return

    def _remove_subjects_from_data(self, subjects, data, row, col):
        """Remove some subjects by deleting their data

        Parameters
        ----------

        subjects : set or list of string,
        Name of subjects to be removed

        data, row, col : list of int,
        Data extracted from sparse matrix self.matrix

        Returns
        -------

        data, row, col : list of int,
        Data can be used to construct a sparse matrix after removal of those
        subjects

        Note
        ----

        Subjects are not re-indexed. Need to call
        _remove_subjects_by_re_indexing() afterwards to re-index.
        """
        len_data = len(data)
        # subject indices (row indices) to remove
        row_to_remove = set()
        subjects = set(subjects)
        for subj in subjects:
            row_to_remove.add(self.subject_to_idx[subj])
        # data indices to remove from data, row, col lists
        idx_to_remove = []
        for idx, row_idx in enumerate(row):
            if row_idx in row_to_remove:
                idx_to_remove.append(idx)
                self.num_subj_dataset[self.idx_to_dataset[col[idx]]] -= 1
        for ds in self.subj_in_dataset.keys():
            for idx in range(len(self.subj_in_dataset[ds])):
                if self.subj_in_dataset[ds][idx] in subjects:
                    self.subj_in_dataset[ds][idx] = None

        # remove data
        data = [data[i] for i in range(len_data) if i not in idx_to_remove]
        row = [row[i] for i in range(len_data) if i not in idx_to_remove]
        col = [col[i] for i in range(len_data) if i not in idx_to_remove]

        return data, row, col

    def _remove_datasets_from_data(self, datasets, data, row, col):
        """Remove some datasets by deleting their data

        Parameters
        ----------

        datasets : set or list of string,
        Name of datasets to be removed

        data, row, col : list of int,
        Data extracted from sparse matrix self.matrix

        Returns
        -------

        data, row, col : list of int,
        Data can be used to construct a sparse matrix after removal of those
        datasets

        subj_to_check : set of string,
        Name of subjects that are possibly not in any datasets (and thus
        need to be removed) after removal of those datasets.

        Note
        ----

        Datasets are not re-indexed. Need to call
        _remove_datasets_by_re_indexing() afterwards to re-index.
        """
        len_data = len(data)
        col_to_remove = set()  # dataset indices (column indices) to remove
        for ds in datasets:
            col_to_remove.add(self.dataset_to_idx[ds])
        # data indices to remove from data, row, col lists
        idx_to_remove = []
        # possible subject indices to remove after removing datasets
        subj_to_check = set()
        for idx, col_idx in enumerate(col):
            if col_idx in col_to_remove:
                idx_to_remove.append(idx)
                subj_to_check.add(self.idx_to_subject[row[idx]])
        # remove info in dict
        for ds in datasets:
            del self.subj_in_dataset[ds]
            del self.num_subj_dataset[ds]
        # remove data
        data = [data[i] for i in range(len_data) if i not in idx_to_remove]
        row = [row[i] for i in range(len_data) if i not in idx_to_remove]
        col = [col[i] for i in range(len_data) if i not in idx_to_remove]

        return data, row, col, subj_to_check

    def _remove_subjects_by_re_indexing(self, subjects, row):
        """Re-index all subjects after removal of data of some subjects
            so that the subject indexing are still contiguous.

        Parameters
        ----------

        subjects : set or list of string,
        Name of subjects where their data in self.matrix are removed
        already and need to be removed from indexing

        row : list of int, row indices as in a sparse matrix
        Row (subject) indices before re-indexing of subjects

        Returns
        -------

        row : list of int, row indices as in a sparse matrix
        Row (subject) indices after re-indexing of subjects

        Note
        ----

        Data of subjects 'subjects' must be removed already. If not,
        need to call _remove_subjects_from_data() beforehand
        """
        # remaining subjects after moving 'subjects'
        remained = set(self.subject_to_idx.keys()) - set(subjects)
        # re-indexing
        new_subject_to_idx, new_idx_to_subject = {}, {}
        for idx, subj in enumerate(remained):
            new_idx_to_subject[idx] = subj
            new_subject_to_idx[subj] = idx
        # map indices based on new indexing
        for idx, r in enumerate(row):
            subj = self.idx_to_subject[r]
            new_r = new_subject_to_idx[subj]
            row[idx] = new_r
        # update mapping
        self.subject_to_idx, self.idx_to_subject = new_subject_to_idx, \
            new_idx_to_subject
        # update total number of subjects
        self.num_subj -= len(subjects)
        return row

    def _remove_datasets_by_re_indexing(self, datasets, col):
        """Re-index all datasets after removal of data of some datasets
            so that the dataset indexing are still contiguous.

        Parameters
        ----------

        datasets : set or list of string,
        Name of datasets where their data in self.matrix are removed
        already and need to be removed from indexing

        col : list of int, col indices as in a sparse matrix
        Col (dataset) indices before re-indexing of datasets

        Returns
        -------

        col : list of int, col indices as in a sparse matrix
        Col (dataset) indices after re-indexing of datasets

        Note
        ----

        Data of datasets 'datasets' must be removed already. If not,
        need to call _remove_datasets_from_data() beforehand
        """
        # remaining datasets after moving 'datasets'
        remained = set(self.dataset_to_idx.keys()) - set(datasets)
        # re-indexing
        new_dataset_to_idx, new_idx_to_dataset = {}, {}
        for idx, ds in enumerate(remained):
            new_idx_to_dataset[idx] = ds
            new_dataset_to_idx[ds] = idx
        # map indices based on new indexing
        for idx, c in enumerate(col):
            ds = self.idx_to_dataset[c]
            new_c = new_dataset_to_idx[ds]
            col[idx] = new_c
        # update mapping
        self.dataset_to_idx, self.idx_to_dataset = new_dataset_to_idx, \
            new_idx_to_dataset
        # update total number of datasets
        self.num_dataset -= len(datasets)
        return col
