"""Hierarchical Topographical Factor Analysis (HTFA)

This implementation is based on the work:
.. [1] "Topographic factor analysis: a bayesian model for inferring brain
        networks from neural data"
   J. R. Manning, R. Ranganath, K. A. Norman, and D. M. Blei
   PLoS One, vol. 9, no. 5, p. e94914,2014

.. [2] "Hierarchical topographic factor analysis"
   J. R. Manning, R. Ranganath, W. Keung, N. B. Turk-Browne, J. D.Cohen,
   K. A. Norman, and D. M. Blei
   Pattern Recognition in Neuroimaging, 2014 International Workshop on,
   June 2014, pp. 1–4.

.. [3] "Scaling Up Multi-Subject Neuroimaging Factor Analysis"
   Michael J. Anderson, Mihai Capota, Javier S. Turek, Xia Zhu,
   Theodore L. Willke, Yida Wang, Po-Hsuan Chen, Jeremy R. Manning,
   Peter J. Ramadge, and Kenneth A. Norman
   2016.

"""

# Authors: Xia Zhu (Intel Labs), Jeremy Manning (Dartmouth College) 2015~2016

import numpy as np
from mpi4py import MPI
import time
import os
import sys
from .tfa import TFA
from .utils import fast_inv, from_tri_2_sym, from_sym_2_tri


class HTFA(TFA):

    """Hierarchical Topographical Factor Analysis (HTFA)

    Given multi-subject data, factorize it as a spatial factor F and
    a weight matrix W per subject.
    Also estimate global template across subjects:

m sklearn.base import BaseEstimator
    .. math:: X_i \\approx F_i W_i ,~for~all~i=1\dots N

    Parameters
    ----------

    R : list of 2D arrays, element i has shape=[n_voxel, n_dim]
        Each element in the list contains the coordinate matrix
        of fMRI data of one subject.

    K : int, default: 50
          Number of factors to compute.

    max_outer_iter : int, default: 10
        Number of outer iterations to run the algorithm.

    max_inner_iter : int, default: 10
        Number of inner iterations to run the algorithm.

    n_subj : int, default: 1
        Number of subjects in dataset.

    threshold : float, default: 1.0
       Tolerance for terminate the parameter estimation

    nlss_method : {‘trf’, ‘dogbox’, ‘lm’}, default: 'trf'
       Alogirthm used by scipy.least_suqares to perform minimization.
       More information at
http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html

    nlss_loss: str or callable, default: 'linear'
       Loss function used by scipy.least_squares.
       More information at
http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html

    x_scale : float or array_like or ‘jac’, default: 1.0
       Characteristic scale of each variable for scipy.least_suqares.
       More information at
http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.least_squares.html

    tr_solver: {None, ‘exact’, ‘lsmr’}, default: None
       Method for solving trust-region subproblems, relevant only for ‘trf'
       and ‘dogbox’ methods.
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

    output_path : str, default: None
       The directory to save results.

    output_prefix : list of str, default: None
       The prefix to use for each subject when saving results

    verbose : boolean, default: False
        Verbose mode flag.


    .. note::
    The number of voxels and the number of samples may be different
    between subjects.

    --------

    """

    K = 50
    n_subj = 1
    max_outer_iter = 10
    max_inner_iter = 10
    threshold = 0.01
    nlss_method = 'trf'
    nlss_loss = 'soft_l1'
    x_scale = 'jac'
    tr_solver = None
    weight_method = 'rr'
    upper_ratio = 1.8
    lower_ratio = 0.02
    voxel_ratio = 0.25
    tr_ratio = 0.1
    max_voxel = 5000
    max_tr = 500
    output_path = None
    output_prefix = None
    verbose = False

    def __init__(self, R, K, n_subj=1, max_outer_iter=10, max_inner_iter=10,
                 threshold=0.01, nlss_method='trf', nlss_loss='soft_l1',
                 x_scale='jac', tr_solver=None, weight_method='rr',
                 upper_ratio=1.8, lower_ratio=0.02, voxel_ratio=0.25,
                 tr_ratio=0.1, max_voxel=5000, max_tr=500,
                 output_path=None, output_prefix=None, verbose=False):
        self.K = K
        self.n_subj = n_subj
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.threshold = threshold
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.x_scale = x_scale
        self.tr_solver = tr_solver
        self.weight_method = weight_method
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.voxel_ratio = voxel_ratio
        self.tr_ratio = tr_ratio
        self.max_voxel = max_voxel
        self.max_tr = max_tr
        self.output_path = output_path
        self.output_prefix = output_prefix

    def _map_update(
            self,
            prior_mean,
            prior_cov,
            global_cov,
            new_observation,
            n_dim):
        """Maximum A Posterior (MAP) update of a parameter

        Parameters
        ----------

        prior_mean : float or 1D array
            Prior mean of parameters.

        prior_cov : float or 1D array
            Prior variance of scalar parameter, or
            prior covariance of multivariate parameter

        global_cov : float or 1D array
            Global prior variance of scalar parameter, or
            global prior covariance of multivariate parameter

        new_observation : 1D or 2D array, with shape [n_dim, n_subj]
            New observations on parameters.

        n_dim : int
            Then_dimensionality of parameter to be estimated.

        Returns

        -------

        posterior_mean : float or 1D array
            Posterior mean of parameters.

        posterior_cov : float or 1D array
            Posterior variance of scalar parameter, or
            posterior covariance of multivariate parameter

        """
        scaled = global_cov / float(self.n_subj)
        common = fast_inv(prior_cov + scaled)
        observation_mean = np.mean(new_observation, axis=1)
        posterior_mean = prior_cov.dot(
            common.dot(observation_mean)) + scaled.dot(common.dot(prior_mean))
        posterior_cov = prior_cov.dot(common.dot(scaled))
        return posterior_mean, posterior_cov

    def _map_update_posterior(
            self,
            global_prior,
            global_const,
            gather_posterior,
            n_dim,
            map_offset,
            cov_vec_size):
        """Maximum A Posterior (MAP) update of HTFA parameters

        Parameters
        ----------

        global_prior : 1D array
            Global prior of parameters.

        global_const : 1D array
            Constant part of global prior. Namely the covariance
            of centers' mean.  The varaince of widths' mean.

        gather_posterior : 1D array
            Latest posterior gathered from all subjects

        n_dim : int
            Then_dimensionality of parameter to be estimated.

        map_offset : 1D array
            The offset to different fields in global prior

        cov_vec_size : 1D array
            The size of flattened 1D covaraince matrix

        Returns
        -------

        global_posterior : 1D array
            Updated global posterior based on Bayesian
            maximum a posterior (MAP) estimation.

        """
        global_posterior = global_prior.copy()
        prior_centers = global_prior[
            0:map_offset[1]].copy().reshape(
            self.K,
            n_dim)
        prior_widths = global_prior[
            map_offset[1]:map_offset[2]].copy().reshape(
            self.K,
            1)
        prior_centers_mean_cov = global_prior[
            map_offset[2]:map_offset[3]].copy().reshape(
            self.K, cov_vec_size)
        prior_widths_mean_var = \
            global_prior[map_offset[3]:].copy().reshape(self.K, 1)
        global_centers_cov = global_const[
            0:self.K *
            cov_vec_size].copy().reshape(
            self.K,
            cov_vec_size)
        global_widths_var = \
            global_const[self.K * cov_vec_size:].copy().reshape(self.K, 1)
        center_size = self.K * n_dim
        posterior_size = center_size + self.K
        for k in np.arange(self.K):
            next_centers = np.zeros((n_dim, self.n_subj))
            next_widths = np.zeros(self.n_subj)
            for s in np.arange(self.n_subj):
                center_start = s * posterior_size
                width_start = center_start + center_size
                next_centers[:, s] = gather_posterior[
                    center_start + k * n_dim:center_start + (k + 1) * n_dim]\
                    .copy()
                next_widths[s] = gather_posterior[width_start + k].copy()

            # centers
            posterior_mean, posterior_cov = self._map_update(
                prior_centers[k].T.copy(), from_tri_2_sym(
                    prior_centers_mean_cov[k], n_dim), from_tri_2_sym(
                    global_centers_cov[k], n_dim), next_centers, self.n_subj,
                n_dim)
            global_posterior[k * n_dim:(k + 1) * n_dim] = posterior_mean.T
            global_posterior[map_offset[2] +
                             k *
                             cov_vec_size:map_offset[2] +
                             (k +
                              1) *
                             cov_vec_size] = from_sym_2_tri(posterior_cov)

            # widths
            scaled = global_widths_var[k] / float(self.n_subj)
            common = 1.0 / (prior_widths_mean_var[k] + scaled)
            observation_mean = np.mean(next_widths)
            tmp = common * scaled
            global_posterior[map_offset[1] + k] = prior_widths_mean_var[k] * \
                common * observation_mean + tmp * prior_widths[k]
            global_posterior[map_offset[3] + k] = \
                prior_widths_mean_var[k] * tmp

        return global_posterior

    def _get_gather_offset(self, size, prior_size):
        """Calculate the offset for gather result from this process

        Parameters
        ----------

        size : int
            The total number of process.

        prior_size : int
            The size of subject prior in bytes.


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
            gather_size[cur_rank] += prior_size
            subject_map[idx] = (cur_rank, num_local_subjs[cur_rank])
            num_local_subjs[cur_rank] += 1

        for idx in np.arange(size - 1) + 1:
            gather_offset[idx] = gather_offset[idx - 1] + gather_size[idx - 1]

        tuple_size = tuple(map(tuple, gather_size.reshape((1, size))))[0]
        tuple_offset = tuple(map(tuple, gather_offset.reshape((1, size))))[0]
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

    def _get_weight_offset(self, weight_size, size):
        """Calculate the offset for this process's weight

        Parameters
        ----------

        weight_size : int
            The total size of weight for all subjects on this process.

        size : int
            Total number of MPI process.


        Returns
        -------

        weight_offset : 1D array
            Number of elements away from the first element
            in the combined weight array at which to begin
            the new, segmented array for a subject

        """

        weight_offset = np.zeros(size)
        for idx in np.arange(size - 1) + 1:
            weight_offset[idx] = weight_offset[idx - 1] + weight_size[idx - 1]
        return weight_offset

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

        sample_scaling : 1D array
            Subsampling coefficient for each subject

        center_width_bounds : list of tuple
            Upper and lower bounds for each subject's centers and widths

        """

        max_sample_tr = np.zeros(n_local_subj).astype(int)
        max_sample_voxel = np.zeros(n_local_subj).astype(int)
        sample_scaling = np.zeros(n_local_subj)
        center_width_bounds = []
        for idx in np.arange(n_local_subj):
            nvoxel = data[idx].shape[0]
            ntr = data[idx].shape[1]
            max_sample_voxel[idx] = min(self.max_voxel,
                                        int(self.voxel_ratio * nvoxel))
            max_sample_tr[idx] = min(self.max_tr, int(self.tr_ratio * ntr))
            sample_scaling[idx] = 0.5 * float(
                max_sample_voxel[idx] *
                max_sample_tr[idx]) / float(nvoxel * ntr)
            center_width_bounds.append(
                self._get_bounds(
                    self.R[idx],
                    self.K,
                    self.upper_ratio,
                    self.lower_ratio))
        return max_sample_tr, max_sample_voxel, sample_scaling,\
            center_width_bounds

    def _fit_htfa(self, data):
        """HTFA main algorithm

        Parameters
        ----------

        data : list of 2D array. Each in shape [n_voxel, n_tr]
            Total number of MPI process.

        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        comm.barrier()
        if rank == 0:
            start_time = time.time()

        if self.n_subj % size == 0:
            use_gather = True
        else:
            use_gather = False

        n_local_subj = len(self.R)
        K = self.K
        n_dim = self.R[0].shape[1]
        cov_vec_size = np.sum(np.arange(n_dim) + 1)
        # centers,widths,centerCov,widthVar
        prior_bcast_size = K * (n_dim + 2 + cov_vec_size)
        # centers,widths
        prior_size = K * (n_dim + 1)
        # map data to processes
        gather_size, gather_offset, subject_map = self._get_gather_offset(
            self.n_subj, size, prior_size)
        max_sample_tr, max_sample_voxel, sample_scaling, center_width_bounds =\
            self._get_subject_info(n_local_subj, data, self.R)

        n_local_subj = len(data)
        local_posterior = np.zeros(n_local_subj * prior_size)
        local_prior = np.zeros(n_local_subj * prior_size)
        weight_size = np.zeros(size).astype(int)

        if rank == 0:
            idx = np.random.choice(n_local_subj, 1)
            global_prior, map_offset, global_const = self._get_global_prior(
                self.R[idx], self.K, n_dim, cov_vec_size)
            gather_posterior = np.zeros(self.n_subj * prior_size)
            global_posterior = np.zeros(prior_size)
        else:
            global_prior = np.zeros(prior_bcast_size)
            map_offset = self._get_map_offset(self.K, n_dim, cov_vec_size)
            gather_posterior = None
            global_posterior = None

        node_weight_size, local_weight_offset =\
            self._get_weight_size(data, K, n_local_subj)
        local_weights = np.zeros(node_weight_size[0])

        comm.Gather(node_weight_size, weight_size, root=0)

        if rank == 0:
            weight_offset = self._get_weight_offset(weight_size, size)
        else:
            weight_offset = np.zeros(size).astype(int)

        comm.Bcast(weight_offset, root=0)
        comm.Bcast(weight_size, root=0)
        weight_offset = tuple(map(tuple, weight_offset.reshape((1, size))))[0]
        weight_size = tuple(map(tuple, weight_size.reshape((1, size))))[0]

        init_global_prior = global_prior.copy()
        m = 0
        outer_converged = np.array([0])
        while m < self.max_outer_iter and not outer_converged[0]:
            # root broadcast first 4 fields of global_prior to all nodes
            comm.Bcast(global_prior, root=0)
            # each node loop over its data
            for s, subj_data in enumerate(data):
                local_prior[s * prior_size:(s + 1) * prior_size] =\
                    global_prior[0:prior_size].copy()
                TFA(self.max_inner_iter,
                    self.threshold,
                    K,
                    self.nlss_method,
                    self.nlss_loss,
                    self.x_scale,
                    self.tr_solver,
                    self.weight_method,
                    self.upper_ratio,
                    self.lower_ratio,
                    max_sample_tr[s],
                    max_sample_voxel[s],
                    sample_scaling[s],
                    center_width_bounds[s],
                    m * self.max_inner_iter)
                local_posterior[
                    s *
                    prior_size:(
                        s +
                        1) *
                    prior_size] = self._fit_tfa(
                    local_prior[
                        s *
                        prior_size:(
                            s +
                            1) *
                        prior_size],
                    global_prior,
                    map_offset,
                    subj_data,
                    self.R[s])
                local_posterior = self._assign_posterior(
                    global_prior[
                        0:K *
                        n_dim].reshape(
                        (K,
                         n_dim)),
                    local_posterior,
                    map_offset,
                    K,
                    n_dim,
                    cov_vec_size,
                    True)
                base = s * prior_size

            if use_gather:
                comm.Gather(local_posterior, gather_posterior, root=0)
            else:
                comm.Gatherv(
                    local_posterior, [
                        gather_posterior, gather_size, gather_offset,
                        MPI.DOUBLE])

            # root updates update global_posterior
            if rank == 0:
                global_posterior = self._map_update_posterior(
                    global_prior,
                    global_const,
                    gather_posterior,
                    K,
                    self.n_subj,
                    n_dim,
                    map_offset,
                    cov_vec_size)
                global_posterior = self._assign_posterior(
                    global_prior[
                        0:K *
                        n_dim].reshape(
                        (K,
                         n_dim)),
                    global_posterior,
                    map_offset,
                    K,
                    n_dim,
                    cov_vec_size,
                    False)
                is_converged, _ = self._converged(
                    global_prior[
                        0:prior_size], global_posterior[
                        0:prior_size], self.threshold)
                if is_converged:
                    print("converged at %d outer iter" % (m))
                    outer_converged[0] = 1
                else:
                    global_prior = global_posterior
            comm.Bcast(outer_converged, root=0)
            print('+')
            m += 1

        # update weight matrix for each subject
        for s, subj_data in enumerate(data):
            base = s * prior_size
            centers = local_posterior[
                base:base +
                K *
                n_dim].copy().reshape(
                (K,
                 n_dim))
            widths = local_posterior[
                base +
                K *
                n_dim:base +
                prior_size].copy().reshape(
                (K,
                 1))
            unique_R, inds = self._get_unique_R(self.R[s])
            F = self._get_factors(unique_R, inds, centers, widths)
            if s == n_local_subj - 1:
                local_weights[
                    local_weight_offset[s]:] = self._get_weights(
                    subj_data,
                    F,
                    self.weight_method).ravel()
            else:
                local_weights[
                    local_weight_offset[s]:local_weight_offset[
                        s +
                        1]] = self._get_weights(
                    subj_data,
                    F,
                    self.weight_method).ravel()

        comm.barrier()
        if rank == 0:
            print("htfa exe time: %s seconds" % (time.time() - start_time))
            sys.stdout.flush()
            start_time = time.time()

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        for s, subj_data in enumerate(data):
            base = s * prior_size
            np.save(self.output_path +
                    '/' +
                    self.output_prefix[s] +
                    '_posterior.npy', local_posterior[base:base +
                                                      K *
                                                      (n_dim +
                                                       1)])
            if s == n_local_subj - 1:
                np.save(
                    self.output_path +
                    '/' +
                    self.output_prefix[s] +
                    '_weights.npy',
                    local_weights[
                        local_weight_offset[s]:])
            else:
                np.save(
                    self.output_path +
                    '/' +
                    self.output_prefix[s] +
                    '_weights.npy',
                    local_weights[
                        local_weight_offset[s]:local_weight_offset[
                            s +
                            1]])

        comm.barrier()
        if rank == 0:
            np.save(
                self.output_path +
                '/global_posterior.npy',
                global_posterior[
                    0:prior_size])
            print("write result time: %s seconds" % (time.time() - start_time))
            sys.stdout.flush()

        if rank == 0:
            return init_global_prior, global_posterior[
                0:prior_size], gather_posterior, local_weights, subject_map
        else:
            return local_weights, local_posterior

    def fit(self, X, y=None):
        """Computes the probabilistic Shared Response Model

        Parameters
        ----------
        X :  list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used
        """
        if self.verbose:
            print('Start to fit HTFA ')

        # Check the number of subjects
        if len(X) < 1:
            raise ValueError("Need at leat one subject to train the model.\
                              Got {0:d}".format(len(X)))

        # main algorithm
        self._fit_htfa(X)
