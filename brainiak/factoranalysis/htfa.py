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
"""Hierarchical Topographical Factor Analysis (HTFA)

This implementation is based on the work in [Manning2014-1]_, [Manning2014-2]_,
and [AndersonMJ2016]_.

.. [Manning2014-1] "Topographic factor analysis: a bayesian model for
   inferring brain networks from neural data", J. R. Manning,
   R. Ranganath, K. A. Norman, and D. M. Blei. PLoS One, vol. 9, no. 5,
   2014.

.. [Manning2014-2] "Hierarchical topographic factor analysis", Jeremy. R.
   Manning, R. Ranganath, W. Keung, N. B. Turk-Browne, J. D.Cohen,
   K. A. Norman, and D. M. Blei. Pattern Recognition in Neuroimaging,
   2014 International Workshop on, June 2014.

.. [AndersonMJ2016] "Enabling Factor Analysis on Thousand-Subject Neuroimaging
   Datasets",
   Michael J. Anderson, Mihai Capotă, Javier S. Turek, Xia Zhu, Theodore L.
   Willke, Yida Wang, Po-Hsuan Chen, Jeremy R. Manning, Peter J. Ramadge,
   Kenneth A. Norman,
   IEEE International Conference on Big Data, 2016.
   https://doi.org/10.1109/BigData.2016.7840719
"""

# Authors: Xia Zhu (Intel Labs), Jeremy Manning (Dartmouth College) 2015~2016

import numpy as np
from mpi4py import MPI
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
import logging
from .tfa import TFA
from ..utils.utils import from_tri_2_sym, from_sym_2_tri

__all__ = [
    "HTFA",
]

logger = logging.getLogger(__name__)


class HTFA(TFA):
    """Hierarchical Topographical Factor Analysis (HTFA)

    Given multi-subject data, HTFA factorizes data from each subject as a
    spatial factor F and a weight matrix W per subject. Also at top
    level, it estimates global template across subjects:


    Parameters
    ----------

    K : int
        Number of factors to compute.

    n_subj : int
        Total number of subjects in dataset.

    max_global_iter : int, default: 10
        Number of global iterations to run the algorithm.

    max_local_iter : int, default: 10
        Number of local iterations to run on each subject within each
        global interation.

    threshold : float, default: 1.0
        Tolerance for terminate the parameter estimation

    nlss_method : {'trf', 'dogbox', 'lm'}, default: 'trf'
        Non-Linear Least Square (NLSS) algorithm used by scipy.least_suqares to
        perform minimization. More information at
        http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html

    nlss_loss: str or callable, default: 'linear'
        Loss function used by scipy.least_squares.
        More information at
        http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html

    jac : {'2-point', '3-point', 'cs', callable}, default: '2-point'
        Method of computing the Jacobian matrix.
        More information at
        http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html

    x_scale : float or array_like or 'jac', default: 1.0
        Characteristic scale of each variable for scipy.least_suqares.
        More information at
        http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html

    tr_solver: {None, 'exact', 'lsmr'}, default: None
        Method for solving trust-region subproblems, relevant only for 'trf'
        and 'dogbox' methods.
        More information at
        http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html

    weight_method : {'rr','ols'}, default: 'rr'
        Method for estimating weight matrix W given X and F.
        'rr' means ridge regression, 'ols' means ordinary least square.

    upper_ratio : float, default: 1.8
        The upper bound of the ratio between factor's width and brain diameter.

    lower_ratio : float, default: 0.02
        The lower bound of the ratio between factor's width and brain diameter.

    voxel_ratio : float, default: 0.25
        The percentage of voxels to sample in each inner iteration.

    tr_ratio : float, default: 0.1
        The percentage of trs to sample in each inner iteration.

    max_voxel : int, default: 5000
        The maximum number of voxels to sample in each inner iteration.

    max_tr : int, default: 500
        The maximum number of trs to sample in each inner iteration.

    comm : Intracomm
        MPI communication group, default MPI.COMM_WORLD

    verbose : boolean, default: False
        Verbose mode flag.


    Attributes
    ----------
    global_prior_ : 1D array
        The global prior on mean and variance of centers and widths.

    global_posterior_ : 1D array
        The global posterior on mean and variance of centers and widths.

    local_posterior_ : 1D array
        Local posterior on centers and widths of subjects allocated
        to this process.

    local_weights_ : 1D array
        Local posterior on weights allocated to this process.


    Notes
    -----
    We recommend to use data in MNI space to better interpret global template

    """

    def __init__(self, K, n_subj, max_global_iter=10, max_local_iter=10,
                 threshold=0.01, nlss_method='trf', nlss_loss='soft_l1',
                 jac='2-point', x_scale='jac', tr_solver=None,
                 weight_method='rr', upper_ratio=1.8, lower_ratio=0.02,
                 voxel_ratio=0.25, tr_ratio=0.1, max_voxel=5000, max_tr=500,
                 comm=MPI.COMM_WORLD, verbose=False):
        self.K = K
        self.n_subj = n_subj
        self.max_global_iter = max_global_iter
        self.max_local_iter = max_local_iter
        self.threshold = threshold
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.jac = jac
        self.x_scale = x_scale
        self.tr_solver = tr_solver
        self.weight_method = weight_method
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.voxel_ratio = voxel_ratio
        self.tr_ratio = tr_ratio
        self.max_voxel = max_voxel
        self.max_tr = max_tr
        self.comm = comm
        self.verbose = verbose

    def _converged(self):
        """Check convergence based on maximum absolute difference

        Returns
        -------

        converged : boolean
            Whether the parameter estimation converged.

        max_diff : float
            Maximum absolute difference between prior and posterior.

        """

        prior = self.global_prior_[0:self.prior_size]
        posterior = self.global_posterior_[0:self.prior_size]
        diff = prior - posterior
        max_diff = np.max(np.fabs(diff))
        if self.verbose:
            _, mse = self._mse_converged()
            diff_ratio = np.sum(diff ** 2) / np.sum(posterior ** 2)
            logger.info(
                'htfa prior posterior max diff %f mse %f diff_ratio %f' %
                ((max_diff, mse, diff_ratio)))

        if max_diff > self.threshold:
            return False, max_diff
        else:
            return True, max_diff

    def _mse_converged(self):
        """Check convergence based on mean squared difference between
            prior and posterior

        Returns
        -------

        converged : boolean
            Whether the parameter estimation converged.

        mse : float
            Mean squared error between prior and posterior.

        """

        prior = self.global_prior_[0:self.prior_size]
        posterior = self.global_posterior_[0:self.prior_size]
        mse = mean_squared_error(prior, posterior,
                                 multioutput='uniform_average')
        if mse > self.threshold:
            return False, mse
        else:
            return True, mse

    def _map_update(
            self,
            prior_mean,
            prior_cov,
            global_cov_scaled,
            new_observation):
        """Maximum A Posterior (MAP) update of a parameter

        Parameters
        ----------

        prior_mean : float or 1D array
            Prior mean of parameters.

        prior_cov : float or 1D array
            Prior variance of scalar parameter, or
            prior covariance of multivariate parameter

        global_cov_scaled : float or 1D array
            Global prior variance of scalar parameter, or
            global prior covariance of multivariate parameter

        new_observation : 1D or 2D array, with shape [n_dim, n_subj]
            New observations on parameters.

        Returns
        -------

        posterior_mean : float or 1D array
            Posterior mean of parameters.

        posterior_cov : float or 1D array
            Posterior variance of scalar parameter, or
            posterior covariance of multivariate parameter

        """
        common = np.linalg.inv(prior_cov + global_cov_scaled)
        observation_mean = np.mean(new_observation, axis=1)
        posterior_mean = prior_cov.dot(common.dot(observation_mean)) +\
            global_cov_scaled.dot(common.dot(prior_mean))
        posterior_cov =\
            prior_cov.dot(common.dot(global_cov_scaled))
        return posterior_mean, posterior_cov

    def _map_update_posterior(self):
        """Maximum A Posterior (MAP) update of HTFA parameters

        Returns
        -------
        HTFA
            Returns the instance itself.
        """
        self.global_posterior_ = self.global_prior_.copy()
        prior_centers = self.get_centers(self.global_prior_)
        prior_widths = self.get_widths(self.global_prior_)
        prior_centers_mean_cov = self.get_centers_mean_cov(self.global_prior_)
        prior_widths_mean_var = self.get_widths_mean_var(self.global_prior_)
        center_size = self.K * self.n_dim
        posterior_size = center_size + self.K
        for k in np.arange(self.K):
            next_centers = np.zeros((self.n_dim, self.n_subj))
            next_widths = np.zeros(self.n_subj)
            for s in np.arange(self.n_subj):
                center_start = s * posterior_size
                width_start = center_start + center_size
                start_idx = center_start + k * self.n_dim
                end_idx = center_start + (k + 1) * self.n_dim
                next_centers[:, s] = self.gather_posterior[start_idx:end_idx]\
                    .copy()
                next_widths[s] = self.gather_posterior[width_start + k].copy()

            # centers
            posterior_mean, posterior_cov = self._map_update(
                prior_centers[k].T.copy(),
                from_tri_2_sym(prior_centers_mean_cov[k], self.n_dim),
                self.global_centers_cov_scaled,
                next_centers)
            self.global_posterior_[k * self.n_dim:(k + 1) * self.n_dim] =\
                posterior_mean.T
            start_idx = self.map_offset[2] + k * self.cov_vec_size
            end_idx = self.map_offset[2] + (k + 1) * self.cov_vec_size
            self.global_posterior_[start_idx:end_idx] =\
                from_sym_2_tri(posterior_cov)

            # widths
            common = 1.0 /\
                (prior_widths_mean_var[k] + self.global_widths_var_scaled)
            observation_mean = np.mean(next_widths)
            tmp = common * self.global_widths_var_scaled
            self.global_posterior_[self.map_offset[1] + k] = \
                prior_widths_mean_var[k] * common * observation_mean +\
                tmp * prior_widths[k]
            self.global_posterior_[self.map_offset[3] + k] = \
                prior_widths_mean_var[k] * tmp

        return self

    def _get_gather_offset(self, size):
        """Calculate the offset for gather result from this process

        Parameters
        ----------

        size : int
            The total number of process.

        Returns
        -------

        tuple_size : tuple_int
            Number of elements to send from each process
            (one integer for each process)

        tuple_offset : tuple_int
            Number of elements away from the first element
            in the array at which to begin the new, segmented
            array for a process
            (one integer for each process)

        subject_map : dictionary
            Mapping between global subject id to local id

        """

        gather_size = np.zeros(size).astype(int)
        gather_offset = np.zeros(size).astype(int)
        num_local_subjs = np.zeros(size).astype(int)
        subject_map = {}

        for idx, s in enumerate(np.arange(self.n_subj)):
            cur_rank = idx % size
            gather_size[cur_rank] += self.prior_size
            subject_map[idx] = (cur_rank, num_local_subjs[cur_rank])
            num_local_subjs[cur_rank] += 1

        for idx in np.arange(size - 1) + 1:
            gather_offset[idx] = gather_offset[idx - 1] + gather_size[idx - 1]

        tuple_size = tuple(gather_size)
        tuple_offset = tuple(gather_offset)
        return tuple_size, tuple_offset, subject_map

    def _get_weight_size(self, data, n_local_subj):
        """Calculate the size of weight for this process

        Parameters
        ----------

        data : a list of 2D array, each in shape [n_voxel, n_tr]
            The fMRI data from multi-subject.

        n_local_subj : int
            Number of subjects allocated to this process.


        Returns
        -------

        weight_size : 1D array
            The size of total subject weight on this process.

        local_weight_offset : 1D array
            Number of elements away from the first element
            in the combined weight array at which to begin
            the new, segmented array for a subject

        """

        weight_size = np.zeros(1).astype(int)
        local_weight_offset = np.zeros(n_local_subj).astype(int)
        for idx, subj_data in enumerate(data):
            if idx > 0:
                local_weight_offset[idx] = weight_size[0]
            weight_size[0] += self.K * subj_data.shape[1]
        return weight_size, local_weight_offset

    def _get_subject_info(self, n_local_subj, data):
        """Calculate metadata for subjects allocated to this process

        Parameters
        ----------

        n_local_subj : int
            Number of subjects allocated to this process.

        data : list of 2D array. Each in shape [n_voxel, n_tr]
            Total number of MPI process.

        Returns
        -------

        max_sample_tr : 1D array
            Maximum number of TR to subsample for each subject

        max_sample_voxel : 1D array
            Maximum number of voxel to subsample for each subject

        """

        max_sample_tr = np.zeros(n_local_subj).astype(int)
        max_sample_voxel = np.zeros(n_local_subj).astype(int)
        for idx in np.arange(n_local_subj):
            nvoxel = data[idx].shape[0]
            ntr = data[idx].shape[1]
            max_sample_voxel[idx] =\
                min(self.max_voxel, int(self.voxel_ratio * nvoxel))
            max_sample_tr[idx] = min(self.max_tr, int(self.tr_ratio * ntr))
        return max_sample_tr, max_sample_voxel

    def _get_mpi_info(self):
        """get basic MPI info

        Returns
        -------
        comm : Intracomm
            Returns MPI communication group

        rank : integer
            Returns the rank of this process

        size : integer
            Returns total number of processes

        """

        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        return rank, size

    def _init_prior_posterior(self, rank, R, n_local_subj):
        """set prior for this subject

        Parameters
        ----------

        rank : integer
            The rank of this process

        R : list of 2D arrays, element i has shape=[n_voxel, n_dim]
            Each element in the list contains the scanner coordinate matrix
            of fMRI data of one subject.

        n_local_subj : integer
            The number of subjects allocated to this process.


        Returns
        -------
        HTFA
            Returns the instance itself.

        """

        if rank == 0:
            idx = np.random.choice(n_local_subj, 1)
            self.global_prior_, self.global_centers_cov,\
                self.global_widths_var = self.get_template(R[idx[0]])
            self.global_centers_cov_scaled =\
                self.global_centers_cov / float(self.n_subj)
            self.global_widths_var_scaled =\
                self.global_widths_var / float(self.n_subj)
            self.gather_posterior = np.zeros(self.n_subj * self.prior_size)
            self.global_posterior_ = np.zeros(self.prior_size)
        else:
            self.global_prior_ = np.zeros(self.prior_bcast_size)
            self.global_posterior_ = None
            self.gather_posterior = None
        return self

    def _gather_local_posterior(self, use_gather,
                                gather_size, gather_offset):
        """Gather/Gatherv local posterior


        Parameters
        ----------
        comm : object
            MPI communication group

        use_gather : boolean
            Whether to use Gather or Gatherv

        gather_size : 1D array
            The size of each local posterior

        gather_offset : 1D array
            The offset of each local posterior


        Returns
        -------
        HTFA
            Returns the instance itself.


        Notes
        -----
        We use numpy array rather than generic Python objects for MPI
        communication because Gatherv is only supported for the former.
        https://pythonhosted.org/mpi4py/usrman/tutorial.html

        """
        if use_gather:
            self.comm.Gather(self.local_posterior_,
                             self.gather_posterior, root=0)
        else:
            target = [
                self.gather_posterior,
                gather_size,
                gather_offset,
                MPI.DOUBLE]
            self.comm.Gatherv(self.local_posterior_, target)
        return self

    def _assign_posterior(self):
        """assign posterior to the right prior based on
           Hungarian algorithm

        Returns
        -------
        HTFA
            Returns the instance itself.
        """

        prior_centers = self.get_centers(self.global_prior_)
        posterior_centers = self.get_centers(self.global_posterior_)
        posterior_widths = self.get_widths(self.global_posterior_)
        posterior_centers_mean_cov =\
            self.get_centers_mean_cov(self.global_posterior_)
        posterior_widths_mean_var =\
            self.get_widths_mean_var(self.global_posterior_)
        # linear assignment on centers
        cost = distance.cdist(prior_centers, posterior_centers, 'euclidean')
        _, col_ind = linear_sum_assignment(cost)
        # reorder centers/widths based on cost assignment
        self.set_centers(self.global_posterior_, posterior_centers)
        self.set_widths(self.global_posterior_, posterior_widths)
        # reorder cov/var based on cost assignment
        self.set_centers_mean_cov(
            self.global_posterior_,
            posterior_centers_mean_cov[col_ind])
        self.set_widths_mean_var(
            self.global_posterior_,
            posterior_widths_mean_var[col_ind])
        return self

    def _update_global_posterior(
            self, rank, m, outer_converged):
        """Update global posterior and then check convergence

        Parameters
        ----------

        rank : integer
            The rank of current process.

        m : integer
            The outer iteration number of HTFA.

        outer_converged : 1D array
            Record whether HTFA loop converged


        Returns
        -------
        1D array, contains only 1 element for MPI
            1 means HTFA converged, 0 means not converged.

        """
        if rank == 0:
            self._map_update_posterior()
            self._assign_posterior()
            is_converged, _ = self._converged()
            if is_converged:
                logger.info("converged at %d outer iter" % (m))
                outer_converged[0] = 1
            else:
                self.global_prior_ = self.global_posterior_
        return outer_converged

    def _update_weight(self, data, R, n_local_subj, local_weight_offset):
        """update local weight

        Parameters
        ----------

        data : list of 2D array, element i has shape=[n_voxel, n_tr]
            Subjects' fMRI data.

        R : list of 2D arrays, element i has shape=[n_voxel, n_dim]
            Each element in the list contains the scanner coordinate matrix
            of fMRI data of one subject.

        n_local_subj : integer
            Number of subjects allocated to this process.

        local_weight_offset : 1D array
            Offset of each subject's weights on this process.


        Returns
        -------
        HTFA
            Returns the instance itself.

        """
        for s, subj_data in enumerate(data):
            base = s * self.prior_size
            centers = self.local_posterior_[base:base + self.K * self.n_dim]\
                .reshape((self.K, self.n_dim))
            start_idx = base + self.K * self.n_dim
            end_idx = base + self.prior_size
            widths = self.local_posterior_[start_idx:end_idx]\
                .reshape((self.K, 1))
            unique_R, inds = self.get_unique_R(R[s])
            F = self.get_factors(unique_R, inds, centers, widths)
            start_idx = local_weight_offset[s]
            if s == n_local_subj - 1:
                self.local_weights_[start_idx:] =\
                    self.get_weights(subj_data, F).ravel()
            else:
                end_idx = local_weight_offset[s + 1]
                self.local_weights_[start_idx:end_idx] =\
                    self.get_weights(subj_data, F).ravel()
        return self

    def _fit_htfa(self, data, R):
        """HTFA main algorithm

        Parameters
        ----------

        data : list of 2D array. Each in shape [n_voxel, n_tr]
            The fMRI data from multiple subjects.

        R : list of 2D arrays, element i has shape=[n_voxel, n_dim]
            Each element in the list contains the scanner coordinate matrix
            of fMRI data of one subject.

        Returns
        -------
        HTFA
            Returns the instance itself.
        """

        rank, size = self._get_mpi_info()
        use_gather = True if self.n_subj % size == 0 else False
        n_local_subj = len(R)
        max_sample_tr, max_sample_voxel =\
            self._get_subject_info(n_local_subj, data)

        tfa = []
        # init tfa for each subject
        for s, subj_data in enumerate(data):
            tfa.append(TFA(
                max_iter=self.max_local_iter,
                threshold=self.threshold,
                K=self.K,
                nlss_method=self.nlss_method,
                nlss_loss=self.nlss_loss,
                x_scale=self.x_scale,
                tr_solver=self.tr_solver,
                weight_method=self.weight_method,
                upper_ratio=self.upper_ratio,
                lower_ratio=self.lower_ratio,
                verbose=self.verbose,
                max_num_tr=max_sample_tr[s],
                max_num_voxel=max_sample_voxel[s]))

        # map data to processes
        gather_size, gather_offset, subject_map =\
            self._get_gather_offset(size)
        self.local_posterior_ = np.zeros(n_local_subj * self.prior_size)
        self._init_prior_posterior(rank, R, n_local_subj)
        node_weight_size, local_weight_offset =\
            self._get_weight_size(data, n_local_subj)
        self.local_weights_ = np.zeros(node_weight_size[0])

        m = 0
        outer_converged = np.array([0])
        while m < self.max_global_iter and not outer_converged[0]:
            if(self.verbose):
                logger.info("HTFA global iter %d " % (m))
            # root broadcast first 4 fields of global_prior to all nodes
            self.comm.Bcast(self.global_prior_, root=0)
            # each node loop over its data
            for s, subj_data in enumerate(data):
                # update tfa with current local prior
                tfa[s].set_prior(self.global_prior_[0:self.prior_size].copy())
                tfa[s].set_seed(m * self.max_local_iter)
                tfa[s].fit(
                    subj_data,
                    R=R[s],
                    template_prior=self.global_prior_.copy())
                tfa[s]._assign_posterior()
                start_idx = s * self.prior_size
                end_idx = (s + 1) * self.prior_size
                self.local_posterior_[start_idx:end_idx] =\
                    tfa[s].local_posterior_

            self._gather_local_posterior(
                use_gather,
                gather_size,
                gather_offset)

            # root updates global_posterior
            outer_converged =\
                self._update_global_posterior(rank, m, outer_converged)
            self.comm.Bcast(outer_converged, root=0)
            m += 1

        # update weight matrix for each subject
        self._update_weight(
            data,
            R,
            n_local_subj,
            local_weight_offset)

        return self

    def _check_input(self, X, R):
        """Check whether input data and coordinates in right type

        Parameters
        ----------
        X :  list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        R : list of 2D arrays, element i has shape=[n_voxel, n_dim]
            Each element in the list contains the scanner coordinate matrix
            of fMRI data of one subject.

        Returns
        -------
        HTFA
            Returns the instance itself.
        """
        # Check data type
        if not isinstance(X, list):
            raise TypeError("Input data should be a list")

        if not isinstance(R, list):
            raise TypeError("Coordinates should be a list")

        # Check the number of subjects
        if len(X) < 1:
            raise ValueError("Need at leat one subject to train the model.\
                              Got {0:d}".format(len(X)))

        for idx, x in enumerate(X):
            if not isinstance(x, np.ndarray):
                raise TypeError("Each subject data should be an array")
            if x.ndim != 2:
                raise TypeError("Each subject data should be 2D array")
            if not isinstance(R[idx], np.ndarray):
                raise TypeError(
                    "Each scanner coordinate matrix should be an array")
            if R[idx].ndim != 2:
                raise TypeError(
                    "Each scanner coordinate matrix should be 2D array")
            if x.shape[0] != R[idx].shape[0]:
                raise TypeError(
                    "n_voxel should be the same in X[idx] and R[idx]")
        return self

    def fit(self, X, R):
        """Compute Hierarchical Topographical Factor Analysis Model
           [Manning2014-1][Manning2014-2]

        Parameters
        ----------
        X :  list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        R : list of 2D arrays, element i has shape=[n_voxel, n_dim]
            Each element in the list contains the scanner coordinate matrix
            of fMRI data of one subject.

        Returns
        -------
        HTFA
            Returns the instance itself.
        """
        self._check_input(X, R)
        if self.verbose:
            logger.info("Start to fit HTFA")
        self.n_dim = R[0].shape[1]
        self.cov_vec_size = np.sum(np.arange(self.n_dim) + 1)
        # centers,widths
        self.prior_size = self.K * (self.n_dim + 1)
        # centers,widths,centerCov,widthVar
        self.prior_bcast_size =\
            self.K * (self.n_dim + 2 + self.cov_vec_size)
        self.get_map_offset()
        self._fit_htfa(X, R)
        return self
