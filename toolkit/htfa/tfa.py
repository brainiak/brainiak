"""Topographical Factor Analysis (TFA)

This implementation is based on the work:
.. [1] "Topographic factor analysis: a bayesian model for inferring brain
        networks from neural data"
   J. R. Manning, R. Ranganath,self.K. A. Norman, and D. M. Blei
   PLoS One, vol. 9, no. 5, p. e94914,2014

.. [2] "Scaling Up Multi-Subject Neuroimaging Factor Analysis"
   Michael J. Anderson, Mihai Capota, Javier S. Turek, Xia Zhu,
   Theodore L. Willke, Yida Wang, Po-Hsuan Chen, Jeremy R. Manning,
   Peter J. Ramadge, andself.Kenneth A. Norman
   2016.

"""

# Authors: Xia Zhu (Intel Labs), Jeremy Manning (Dartmouth College) 2015~2016

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from .utils import from_tri_2_sym, from_sym_2_tri
import numpy as np
import sys
import math
import tfa_extension
import gc


class TFA(BaseEstimator):

    """Topographical Factor Analysis (TFA)

    Given a subject data, factorize it as a spatial factor F and
    a weight matrix W.

    .. math:: X \\approx FW

    Parameters
    ----------

    R : 2D array, in shape [n_voxel, n_dim]
        The coordinate matrix of fMRI data.

    sample_scaling : float
       The subsampling coefficient.
       0.5*max_num_voxel*max_num_tr/(n_voxel*n_tr)

   self.K : int, default: 50
          Number of factors to compute.

    max_iter : int, default: 10
        Number of inner iterations to run the algorithm.

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

    max_num_voxel : int, default: 5000
       The maximum number of voxels to subsample.

    max_num_tr : int, default: 500
       The maximum number of TRs to subsample.

    bounds : 2-tuple of array_like, default: None
       The lower and upper bounds on factor's centers and widths.

    seed : int, default: 100
       Seed for subsample voxels and trs.

    verbose : boolean, default: False
        Verbose mode flag.


    .. note::
    The number of voxels and the number of samples may be different
    between subjects.

    --------

    """

    max_iter = 10
    threshold = 0.01
    K = 50
    nlss_method = 'trf'
    nlss_loss = 'soft_l1'
    x_scale = 'jac'
    tr_solver = None
    weight_method = 'rr'
    upper_ratio = 1.8
    lower_ratio = 0.02
    max_num_voxel = 5000
    max_num_tr = 500
    bounds = None
    seed = 100
    verbose = False

    def __init__(
            self,
            R,
            sample_scaling,
            max_iter,
            threshold,
            K,
            nlss_method,
            nlss_loss,
            x_scale,
            tr_solver,
            weight_method,
            upper_ratio,
            lower_ratio,
            max_num_tr,
            max_num_voxels,
            bounds,
            seed,
            verbose):
        self.R = R
        self.sample_scaling = sample_scaling
        self.miter = max_iter
        self.threshold = threshold
        self.K = K
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.x_scale = x_scale
        self.tr_solver = tr_solver
        self.weight_method = weight_method
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.max_num_tr = max_num_tr
        self.max_num_voxels = max_num_voxels
        self.bounds = bounds
        self.seed = seed
        self.verbose = verbose

    def _assign_posterior(
            self,
            prior_centers,
            posterior,
            map_offset,
            n_dim,
            cov_vec_size,
            is_local):
        """Minimum weight matching between prior and posterior,
           assign posterior to the right prior.

        Parameters
        ----------

        prior_centers : 2D array, with shape [K, n_dim]
            Prior of factors' centers.

        posterior : 1D array
            New posterior to be assigned/aligned.

        map_offset : 1D array
            The offset to different fields in global prior

        n_dim : int
            Then_dimensionality of parameter to be estimated.

        cov_vec_size : int
            The size of flattened 1D covaraince matrix

        is_local : boolean
            Whether it is to assign global posterior or subject
            posterior.

        Returns
        -------

        posterior : 1D array
            Posterior that is assigned to prior with minimum
            weight matching.

        """

        posterior_centers = posterior[0:map_offset[1]].reshape(self.K, n_dim)
        posterior_widths = posterior[map_offset[1]:map_offset[2]]\
            .reshape(self.K, 1)
        # linear assignment on centers
        cost = distance.cdist(prior_centers, posterior_centers, 'euclidean')
        _, col_ind = linear_sum_assignment(cost)
        posterior[0:map_offset[1]] = posterior_centers[col_ind].ravel()
        posterior[map_offset[1]:map_offset[2]] = \
            posterior_widths[col_ind].ravel()
        if not is_local:
            posterior_centers_mean_cov = posterior[
                map_offset[2]:map_offset[3]].reshape(
                self.K, cov_vec_size)
            posterior_widths_mean_var = posterior[map_offset[3]:]\
                .reshape(self.K, 1)
            posterior[map_offset[2]:map_offset[3]] = \
                posterior_centers_mean_cov[col_ind].ravel()
            posterior[map_offset[3]:] = \
                posterior_widths_mean_var[col_ind].ravel()
        return posterior

    def _converged(self, prior, posterior, threshold):
        """Check convergence based on maximum absolute difference

        Parameters
        ----------

        prior : 1D array
            Paramter prior.

        posterior : 1D array
            Parameter posterior.

        threshold : float
            Tolerance for terminate the parameter estimation

        Returns
        -------

        converged : boolean
            Whether the parameter estimation converged.

        max_diff : float
            Maximum absolute difference between prior and posterior.

        """

        max_diff = np.max(np.fabs(prior - posterior))
        if max_diff > threshold:
            return False, max_diff
        else:
            return True, max_diff

    def _mse_converged(self, prior, posterior, threshold):
        """Check convergence based on mean squared error

        Parameters
        ----------

        prior : 1D array
            Paramter prior.

        posterior : 1D array
            Parameter posterior.

        threshold : float
            Tolerance for terminate the parameter estimation

        Returns
        -------

        converged : boolean
            Whether the parameter estimation converged.

        mse : float
            Mean squared error between prior and posterior.

        """

        mse = mean_squared_error(prior, posterior,
                                 multioutput='uniform_average')
        if mse > threshold:
            return False, mse
        else:
            return True, mse

    def _get_map_offset(self, n_dim, cov_vec_size):
        """Compute offset of global prior

        Parameters
        ----------

        n_dim : int
            Dimension of the voxel space.

        cov_vec_size : int
            The size of flattened 1D covaraince matrix

        Returns
        -------

        map_offest : 1D array
            The offset to different fields in global prior

        """

        nfield = 4
        map_offset = np.zeros(nfield).astype(int)
        field_size = self.K * np.array([n_dim, 1, cov_vec_size, 1])
        for i in np.arange(nfield - 1) + 1:
            map_offset[i] = map_offset[i - 1] + field_size[i - 1]
        return map_offset

    def _get_global_prior(self, R, n_dim, cov_vec_size):
        """Compute global prior

        Parameters
        ----------

        R : 2D array, with shape [n_voxel, n_dim]
            Coordinate matrix of selected subject.

        n_dim : int
            Dimension of the voxel space.

        cov_vec_size : int
            The size of flattened 1D covaraince matrix

        Returns
        -------

        global_prior : 1D array
            Global prior of parameters and hyper parameters.

        map_offest : 1D array
            The offset to different fields in global prior

        """

        global_prior = np.zeros(self.K * (n_dim + cov_vec_size + 2))
        global_const = np.zeros(self.K * (cov_vec_size + 1))
        centers, widths = self._init_centers_widths(R, self.K)
        center_cov = np.cov(R.T) * math.pow(self.K, -2 / 3.0)
        # print center_cov
        center_cov_all = np.tile(from_sym_2_tri(center_cov), self.K)
        width_var = math.pow(np.nanmax(np.std(R, axis=0)), 2)
        width_var_all = np.tile(width_var, self.K)
        map_offset = self._get_map_offset(self.K, n_dim, cov_vec_size)
        # center mean mean
        global_prior[0:map_offset[1]] = centers.ravel()
        # width mean mean
        global_prior[map_offset[1]:map_offset[2]] = widths.ravel()
        # center mean cov
        global_prior[map_offset[2]:map_offset[3]] = center_cov_all.ravel()
        # width mean var
        global_prior[map_offset[3]:] = width_var_all.ravel()
        # center cov
        global_const[0:self.K * cov_vec_size] = center_cov_all.ravel()
        # width var
        global_const[self.K * cov_vec_size:] = width_var_all.ravel()
        return global_prior, map_offset, global_const

    def _init_centers_widths(self, R):
        """Initialize global prior of centers and widths

        Parameters
        ----------

        R : 2D array, with shape [n_voxel, n_dim]
            Coordinate matrix of selected subject.

        Returns
        -------

        centers : 2D array, with shape [K, n_dim]
            Global prior of factors' centers.

        widths : 1D array, with shape [K, 1]
            Global prior of factors' widths.

        """

        kmeans = KMeans(
            init='k-means++',
            n_clusters=self.K,
            n_init=10,
            random_state=100)
        kmeans.fit(R)
        centers = kmeans.cluster_centers_
        widths = self._get_diameter(R) * np.ones((self.K, 1))
        return centers, widths

    def _get_factors(self, unique_R, inds, centers, widths):
        """Calculate factors based on centers and widths

        Parameters
        ----------

        unique_R : a list of array,
            Each element contains unique value in one dimension of
            coordinate matrix R.

        inds : a list of array,
            Each element contains the indices to reconstruct one
            dimension of original cooridnate matrix from the unique
            array.

        centers : 2D array, with shape [K, n_dim]
            The centers of  factors.

        widths : 1D array, with shape [K, 1]
            The widths of factors.


        Returns
        -------

        F : 2D array, with shape [n_voxel,self.K]
            The latent factors from fMRI data.

        """

        n_dim = len(unique_R)
        F = np.zeros((len(inds[0]), self.K))
        for k in np.arange(self.K):
            unique_dist = []
            # first build up unique dist table
            for d in np.arange(n_dim):
                unique_dist.append((unique_R[d] - centers[k, d])**2)
            # RBF calculation based on looking up the unique table
            F[:, k] = np.exp(-
                             1.0 /
                             widths[k] *
                             (unique_dist[0][inds[0]] +
                              unique_dist[1][inds[1]] +
                                 unique_dist[2][inds[2]]))
        return F

    def _get_weights(self, data, F, weight_method='rr'):
        """Calculate weight matrix based on fMRI data and factors

        Parameters
        ----------

        data : 2D array, with shape [n_voxel, n_tr]
            fMRI data from one subject

        F : 2D array, with shape [n_voxel,self.K]
            The latent factors from fMRI data.


        weight_method : 'rr' or 'ols'. default: 'rr'
            The method to estimate weight matrix.
            'rr' stands for ridge regression.
            'ols' stands for ordinary least square.


        Returns
        -------

        W : 2D array, with shape [K, n_tr]
            The weight matrix from fMRI data.

        """

        beta = np.var(data)
        trans_F = F.T.copy()
        W = np.zeros((self.K, data.shape[1]))
        if weight_method == 'rr':
            W = np.linalg.solve(
                trans_F.dot(F) +
                beta *
                np.identity(self.K),
                trans_F.dot(data))
        elif weight_method == 'ols':
            W = np.linalg.solve(trans_F.dot(F), trans_F.dot(data))
        return W

    def _get_diameter(self, R):
        """Calculate diameter of volume data

        Parameters
        ----------

        R : 2D array, with shape [n_voxel, n_dim]
            The coordinate matrix of fMRI data from one subject

        Returns
        -------

        diameter : float
            The diameter of volume data.

        """

        diameter = np.max(np.ptp(R, axis=0))
        return diameter

    def _get_bounds(self, R, upper_ratio, lower_ratio):
        """Calculate lower and upper bounds for centers and widths

        Parameters
        ----------

        R : 2D array, with shape [n_voxel, n_dim]
            The coordinate matrix of fMRI data from one subject

        upper_ratio : float
            The upper bound of the ratio between factor's width
            and brain diameter.

        lower_ratio : float
            The lower bound of the ratio between factor's width
            and brain diameter.

        Returns
        -------

        bounds : 2-tuple of array_like, default: None
            The lower and upper bounds on factor's centers and widths.

        """

        n_dim = R.shape[1]
        diameter = self._get_diameter(R)
        final_lower = np.zeros(self.K * (n_dim + 1))
        final_lower[0:self.K * n_dim] = np.tile(np.nanmin(R, axis=0), self.K)
        final_lower[
            self.K *
            n_dim:] = np.repeat(
            lower_ratio *
            diameter,
            self.K)
        final_upper = np.zeros(self.K * (n_dim + 1))
        final_upper[0:self.K * n_dim] = np.tile(np.nanmax(R, axis=0), self.K)
        final_upper[
            self.K *
            n_dim:] = np.repeat(
            upper_ratio *
            diameter,
            self.K)
        bounds = (final_lower, final_upper)
        return bounds

    def _residual_center_multivariate(
            self,
            estimate,
            widths,
            n_dim,
            cov_size,
            map_offset,
            unique_R,
            inds,
            X,
            W,
            global_centers,
            global_center_mean_cov,
            sample_scaling,
            data_sigma):
        """Residual function for estimating centers

        Parameters
        ----------

        estimate : 1D array
            Initial estimation on centers

        widths : 1D array
            Current estimation of widths.

        n_dim : int
            The dimension of voxel space.

        cov_size: 1D array
            The size of flattened 1D covaraince matrix

        map_offset: 1D array
            The offset to different fields in global prior

        unique_R : a list of array,
            Each element contains unique value in one dimension of
            coordinate matrix R.

        inds : a list of array,
            Each element contains the indices to reconstruct one
            dimension of original cooridnate matrix from the unique
            array.

        X : 2D array, with shape [n_voxel, n_tr]
            fMRI data from one subject.

        W : 2D array, with shape [K, n_tr]
            The weight matrix.

        global_centers: 2D array, with shape [K, n_dim]
            The global prior on centers

        global_center_mean_cov: 2D array, with shape [K, cov_size]
            The global prior on covariance of centers' mean

        sample_scaling: float
            The subsampling coefficient

        data_sigma: float
            The variance of X.


        Returns
        -------

        final_err : 1D array
            The residual function for estimating centers.

        """

        centers = estimate.reshape((self.K, n_dim))
        F = np.zeros((len(inds[0]), self.K))

        recon_type = 1
        if recon_type == 0:
            F = self._get_factors(unique_R, inds, centers, widths)
            recon = X.size
            final_err = np.zeros(recon + self.K)
            final_err[0:recon] = (data_sigma * (X - F.dot(W))).ravel()
        else:
            recon = X.size
            tfa_extension.factor(
                F,
                centers,
                widths,
                unique_R[0],
                unique_R[1],
                unique_R[2],
                inds[0],
                inds[1],
                inds[2])
            sigma = np.zeros((1,))
            sigma[0] = data_sigma
            if recon_type == 1:
                recon = X.size
                final_err = np.zeros(recon + self.K)
                tfa_extension.recon(final_err[0:recon], X, F, W, sigma)
            else:
                C = F.dot(W)
                if recon_type == 2:
                    recon = X.shape[0]
                    final_err = np.zeros(recon + self.K)
                    tfa_extension.recon_voxel(final_err[0:recon], X, C, sigma)
                elif recon_type == 3:
                    recon = X.shape[1]
                    final_err = np.zeros(recon + self.K)
                    tfa_extension.recon_tr(final_err[0:recon], X, C, sigma)
                elif recon_type == 4:
                    block_size = np.zeros((1,))
                    num_block = np.zeros((1,))
                    nvoxel = X.shape[0]
                    block_ratio = 0.001
                    block_size[0] = int(nvoxel * block_ratio)
                    num_block[0] = math.ceil(float(nvoxel) / block_size)
                    recon = X.shape[1] * int(num_block[0])
                    final_err = np.zeros(recon + self.K)
                    tfa_extension.recon_block(
                        final_err[
                            0:recon],
                        X,
                        C,
                        sigma,
                        block_size,
                        num_block)
                else:
                    block_size = np.zeros((1,))
                    num_block = np.zeros((1,))
                    nvoxel = X.shape[0]
                    block_ratio = 0.001
                    block_size[0] = int(nvoxel * block_ratio)
                    num_block[0] = math.ceil(float(nvoxel) / block_size)
                    recon = int(num_block[0])
                    final_err = np.zeros(recon + self.K)
                    tfa_extension.recon_vblock(
                        final_err[
                            0:recon],
                        X,
                        C,
                        sigma,
                        block_size,
                        num_block)

        # center error
        for k in np.arange(self.K):
            diff = (centers[k] - global_centers[k])
            cov = from_tri_2_sym(global_center_mean_cov[k], n_dim)
            final_err[
                recon +
                k] = math.sqrt(
                sample_scaling *
                diff.dot(
                    np.linalg.solve(
                        cov,
                        diff.T)))
        return final_err

    def _residual_width_multivariate(
            self,
            estimate,
            centers,
            n_dim,
            cov_size,
            map_offset,
            unique_R,
            inds,
            X,
            W,
            global_widths,
            global_width_mean_var_reci,
            sample_scaling,
            data_sigma):
        """Residual function for estimating widths

        Parameters
        ----------

        estimate : 1D array
            Initial estimation on widths

        centers : 1D array
            Current estimation of centers.

        n_dim : int
            The dimension of voxel space.

        cov_size: 1D array
            The size of flattened 1D covaraince matrix

        map_offset: 1D array
            The offset to different fields in global prior

        unique_R : a list of array,
            Each element contains unique value in one dimension of
            coordinate matrix R.

        inds : a list of array,
            Each element contains the indices to reconstruct one
            dimension of original cooridnate matrix from the unique
            array.

        X : 2D array, with shape [n_voxel, n_tr]
            fMRI data from one subject.

        W : 2D array, with shape [K, n_tr]
            The weight matrix.

        global_widths: 1D array
            The global prior on widths

        global_width_mean_var_reci: 1D array
            The reciprocal of global prior on variance of widths' mean

        sample_scaling: float
            The subsampling coefficient

        data_sigma: float
            The variance of X.


        Returns
        -------

        final_err : 1D array
            The residual function for estimating widths.

        """
        widths = estimate.reshape((self.K, 1))
        F = np.zeros((len(inds[0]), self.K))

        recon_type = 1
        if recon_type == 0:
            F = self._get_factors(unique_R, inds, centers, widths)
            recon = X.size
            final_err = np.zeros(recon + self.K)
            final_err[0:recon] = (data_sigma * (X - F.dot(W))).ravel()
        else:
            recon = X.size
            tfa_extension.factor(
                F,
                centers,
                widths,
                unique_R[0],
                unique_R[1],
                unique_R[2],
                inds[0],
                inds[1],
                inds[2])

            sigma = np.zeros((1,))
            sigma[0] = data_sigma
            if recon_type == 1:
                recon = X.size
                final_err = np.zeros(recon + self.K)
                tfa_extension.recon(final_err[0:recon], X, F, W, sigma)
            else:
                C = F.dot(W)
                if recon_type == 2:
                    recon = X.shape[0]
                    final_err = np.zeros(recon + self.K)
                    tfa_extension.recon_voxel(final_err[0:recon], X, C, sigma)
                elif recon_type == 3:
                    recon = X.shape[1]
                    final_err = np.zeros(recon + self.K)
                    tfa_extension.recon_tr(final_err[0:recon], X, C, sigma)
                elif recon_type == 4:
                    block_size = np.zeros((1,))
                    num_block = np.zeros((1,))
                    nvoxel = X.shape[0]
                    block_ratio = 0.001
                    block_size[0] = int(nvoxel * block_ratio)
                    num_block[0] = math.ceil(float(nvoxel) / block_size)
                    recon = X.shape[1] * int(num_block[0])
                    final_err = np.zeros(recon + self.K)
                    tfa_extension.recon_block(
                        final_err[
                            0:recon],
                        X,
                        C,
                        sigma,
                        block_size,
                        num_block)
                else:
                    block_size = np.zeros((1,))
                    num_block = np.zeros((1,))
                    nvoxel = X.shape[0]
                    block_ratio = 0.001
                    block_size[0] = int(nvoxel * block_ratio)
                    num_block[0] = math.ceil(float(nvoxel) / block_size)
                    recon = int(num_block[0])
                    final_err = np.zeros(recon + self.K)
                    tfa_extension.recon_vblock(
                        final_err[
                            0:recon],
                        X,
                        C,
                        sigma,
                        block_size,
                        num_block)

        # width error
        final_err[recon:] = np.sqrt(sample_scaling *
                                    (global_width_mean_var_reci *
                                     (widths -
                                      global_widths)**2).ravel())
        return final_err

    def __get_centers_widths(
            self,
            unique_R,
            inds,
            X,
            W,
            init_centers,
            init_widths,
            global_centers,
            global_widths,
            global_center_mean_cov,
            global_width_mean_var_reci,
            cov_size,
            map_offset):
        """Estimate centers and widths

        Parameters
        ----------

        unique_R : a list of array,
            Each element contains unique value in one dimension of
            coordinate matrix R.

        inds : a list of array,
            Each element contains the indices to reconstruct one
            dimension of original cooridnate matrix from the unique
            array.

        X : 2D array, with shape [n_voxel, n_tr]
            fMRI data from one subject.

        W : 2D array, with shape [K, n_tr]
            The weight matrix.

        init_centers : 2D array, with shape [K, n_dim]
            The initial values of centers.

        init_widths : 1D array
            The initial values of widths.

        global_centers: 1D array
            The global prior on centers

        global_widths: 1D array
            The global prior on widths

        global_center_mean_cov: 2D array, with shape [K, cov_size]
            The global prior on centers' mean

        global_width_mean_var_reci: 1D array
            The reciprocal of global prior on variance of widths' mean

        cov_size: 1D array
            The size of flattened 1D covaraince matrix

        map_offset: 1D array
            The offset to different fields in global prior

        Returns
        -------

        final_estimate.x: 1D array
            The newly estimated centers and widths.

        final_estimate.cost: float
            The cost value.

        """
        n_dim = init_centers.shape[1]
        # least_squares only accept x in 1D format
        init_estimate = np.hstack(
            (init_centers.ravel(), init_widths.ravel())).copy()
        data_sigma = 1.0 / math.sqrt(2.0) * np.std(X)
        final_estimate = least_squares(
            self._residual_multivariate,
            init_estimate,
            self=(
                n_dim,
                cov_size,
                map_offset,
                unique_R,
                inds,
                X,
                W,
                global_centers,
                global_widths,
                global_center_mean_cov,
                global_width_mean_var_reci,
                self.sample_scaling,
                data_sigma),
            method=self.nlss_method,
            loss=self.nlss_loss,
            bounds=self.bounds,
            verbose=0,
            x_scale=self.x_scale,
            tr_solver=self.tr_solver)
        return final_estimate.x, final_estimate.cost

    def _get_centers(
            self,
            unique_R,
            inds,
            X,
            W,
            init_centers,
            init_widths,
            global_centers,
            global_center_mean_cov,
            cov_size,
            map_offset):
        """Estimate centers

        Parameters
        ----------

        unique_R : a list of array,
            Each element contains unique value in one dimension of
            coordinate matrix R.

        inds : a list of array,
            Each element contains the indices to reconstruct one
            dimension of original cooridnate matrix from the unique
            array.

        X : 2D array, with shape [n_voxel, n_tr]
            fMRI data from one subject.

        W : 2D array, with shape [K, n_tr]
            The weight matrix.

        init_centers : 2D array, with shape [K, n_dim]
            The initial values of centers.

        init_widths : 1D array
            The initial values of widths.

        global_centers: 1D array
            The global prior on centers

        global_center_mean_cov: 2D array, with shape [K, cov_size]
            The global prior on centers' mean

        cov_size: 1D array
            The size of flattened 1D covaraince matrix

        map_offset: 1D array
            The offset to different fields in global prior

        Returns
        -------

        center_estimate.x: 1D array
            The newly estimated centers.

        center_estimate.cost: float
            The cost value.

        """
        n_dim = init_centers.shape[1]
        # least_squares only accept x in 1D format
        init_estimate = init_centers.ravel()
        data_sigma = 1.0 / math.sqrt(2.0) * np.std(X)
        center_estimate = least_squares(
            self._residual_center_multivariate,
            init_estimate,
            self=(
                init_widths,
                self.K,
                n_dim,
                cov_size,
                map_offset,
                unique_R,
                inds,
                X,
                W,
                global_centers,
                global_center_mean_cov,
                self.sample_scaling,
                data_sigma),
            method=self.nlss_method,
            loss=self.nlss_loss,
            bounds=(
                self.bounds[0][
                    0:self.K * n_dim],
                self.bounds[1][
                    0:self.K * n_dim]),
            verbose=0,
            x_scale=self.x_scale,
            tr_solver=self.tr_solver)
        return center_estimate.x, center_estimate.cost

    def _get_widths(
            self,
            unique_R,
            inds,
            X,
            W,
            init_centers,
            init_widths,
            global_widths,
            global_width_mean_var_reci,
            cov_size,
            map_offset):
        """Estimate widths

        Parameters
        ----------

        unique_R : a list of array,
            Each element contains unique value in one dimension of
            coordinate matrix R.

        inds : a list of array,
            Each element contains the indices to reconstruct one
            dimension of original cooridnate matrix from the unique
            array.

        X : 2D array, with shape [n_voxel, n_tr]
            fMRI data from one subject.

        W : 2D array, with shape [K, n_tr]
            The weight matrix.

        init_centers : 2D array, with shape [K, n_dim]
            The initial values of centers.

        init_widths : 1D array
            The initial values of widths.

        global_widths: 1D array
            The global prior on widths

        global_width_mean_var_reci: 1D array
            The reciprocal of global prior on variance of widths' mean

        cov_size: 1D array
            The size of flattened 1D covaraince matrix

        map_offset: 1D array
            The offset to different fields in global prior

        Returns
        -------

        width_estimate.x: 1D array
            The newly estimated widths.

        width_estimate.cost: float
            The cost value.

        """
        n_dim = init_centers.shape[1]
        # least_squares only accept x in 1D format
        init_estimate = init_widths.ravel()
        data_sigma = 1.0 / math.sqrt(2.0) * np.std(X)
        width_estimate = least_squares(
            self._residual_width_multivariate,
            init_estimate,
            self=(
                init_centers,
                self.K,
                n_dim,
                cov_size,
                map_offset,
                unique_R,
                inds,
                X,
                W,
                global_widths,
                global_width_mean_var_reci,
                self.sample_scaling,
                data_sigma),
            method=self.nlss_method,
            loss=self.nlss_loss,
            bounds=(
                self.bounds[0][
                    self.K * n_dim:],
                self.bounds[1][
                    self.K * n_dim:]),
            verbose=0,
            x_scale=self.x_scale,
            tr_solver=self.tr_solver)
        return width_estimate.x, width_estimate.cost

    def _fit_tfa(self, local_prior, global_prior, map_offset, data):
        """fit TFA

        Parameters
        ----------

        local_prior : 1D array,
            Subject's prior on centers and widths.

        global_prior : 1D array,
            The global prior on centers and widths.

        map_offset: 1D array
            The offset to different fields in global prior

        data: 2D array, in shape [n_voxel, n_tr]
            The fMRI data from one subject.

        Returns
        -------

        local_posterior: 1D array
            The posterior for this subject.


        """
        n_dim = self.R.shape[1]
        cov_size = (map_offset[3] - map_offset[2]) / self.K
        global_centers = global_prior[0:map_offset[1]].copy()\
            .reshape(self.K, n_dim)
        global_widths = global_prior[
            map_offset[1]:map_offset[2]].copy().reshape(
            self.K,
            1)
        global_center_mean_cov = global_prior[
            map_offset[2]:map_offset[3]].copy().reshape(
            self.K,
            cov_size)
        global_width_mean_var_reci = 1.0 / \
            (global_prior[map_offset[3]:].copy().reshape(self.K, 1))
        inner_converged = False
        np.random.seed(self.seed)
        n = 0
        while n < self.miter and not inner_converged:
            local_posterior = self._fit_tfa_inner(
                local_prior,
                global_centers,
                global_widths,
                global_center_mean_cov,
                global_width_mean_var_reci,
                data,
                self.R,
                self,
                cov_size,
                map_offset)
            local_posterior = self._assign_posterior(
                local_prior[
                    0:self.K *
                    n_dim].reshape(
                    (self.K,
                     n_dim)),
                local_posterior,
                map_offset,
                self.K,
                n_dim,
                cov_size,
                True)
            inner_converged, _ = self._converged(
                local_prior, local_posterior, self.threshold)
            if not inner_converged:
                local_prior = local_posterior
            n += 1
            print('.')
            gc.collect()
        return local_posterior

    def _get_unique_R(self, R):
        """Get unique vlaues from coordinate matrix

        Parameters
        ----------
        R :  2D array
           The coordinate matrix of a subject's fMRI data

        Return
        ------

        unique_R : a list of array,
            Each element contains unique value in one dimension of
            coordinate matrix R.

        inds : a list of array,
            Each element contains the indices to reconstruct one
            dimension of original cooridnate matrix from the unique
            array.


        """
        n_dim = R.shape[1]
        unique_R = []
        inds = []
        for d in np.arange(n_dim):
            tmp_unique, tmp_inds = np.unique(R[:, d], return_inverse=True)
            unique_R.append(tmp_unique)
            inds.append(tmp_inds)
        return unique_R, inds

    def _fit_tfa_inner(
            self,
            local_prior,
            global_centers,
            global_widths,
            global_center_mean_cov,
            global_width_mean_var_reci,
            data,
            R,
            cov_size,
            map_offset):
        """Fit TFA model, the inner loop part

        Parameters
        ----------

        local_prior : 1D array,
            The subject's prior

        global_centers: 1D array
            The global prior on centers

        global_widths: 1D array
            The global prior on widths

        global_center_mean_cov: 2D array, with shape [K, cov_size]
            The global prior on centers' mean

        global_width_mean_var_reci: 1D array
            The reciprocal of global prior on variance of widths' mean

        data: 2D array, in shape [n_voxel, n_tr]
            The fMRI data of a subject

        R: 2D array, in shape [n_voxel, n_dim]
            The coordinate matrix of the fMRI data

        cov_size: 1D array
            The size of flattened 1D covaraince matrix

        map_offset: 1D array
            The offset to different fields in global prior

        Returns
        -------

        local_posterior: 1D array
            The subject's posterior


        """
        n_dim = R.shape[1]
        nfeature = data.shape[0]
        nsample = data.shape[1]
        feature_indices = np.random.choice(
            nfeature,
            self.max_num_voxels,
            replace=False)
        sample_features = np.zeros(nfeature).astype(bool)
        sample_features[feature_indices] = True
        samples_indices = np.random.choice(
            nsample,
            self.max_num_tr,
            replace=False)
        sample_samples = np.zeros(nsample).astype(bool)
        sample_samples[samples_indices] = True
        curr_data = np.zeros(self.max_num_voxels, self.max_num_tr)
        curr_data = data[feature_indices]
        curr_data = curr_data[:, samples_indices].copy()
        curr_R = R[feature_indices].copy()
        centers = local_prior[0:self.K * n_dim].reshape(self.K, n_dim)
        widths = local_prior[self.K * n_dim:].reshape(self.K, 1)
        unique_R, inds = self._get_unique_R(curr_R)
        F = np.zeros((len(inds[0]), self.K))
        recon_type = 1
        if recon_type == 0:
            F = self._get_factors(unique_R, inds, centers, widths)
        else:
            tfa_extension.factor(
                F,
                centers,
                widths,
                unique_R[0],
                unique_R[1],
                unique_R[2],
                inds[0],
                inds[1],
                inds[2])
        W = self._get_weights(curr_data, F, self.weight_method)
        local_centers, center_cost = self._get_centers(
            unique_R, inds, curr_data, W, centers, widths, self,
            global_centers, global_center_mean_cov, cov_size, map_offset)
        local_widths, width_cost = self._get_widths(
            unique_R, inds, curr_data, W, local_centers.reshape(
                self.K, n_dim), widths, self, global_widths,
            global_width_mean_var_reci, cov_size, map_offset)
        local_posterior = np.hstack((local_centers, local_widths))

        sys.stdout.flush()
        return local_posterior

    def fit(self, X, y=None):
        """Computes the probabilistic Shared Response Model

        Parameters
        ----------
        X : 2D array, in shape [n_voxel, n_sample]
            The fMRI data of one subject

        y : not used
        """
        if self.verbose:
            print('Start to fit TFA ')

        # Check the number of subjects
        if len(X) != 1:
            raise ValueError("Need one subject to train the model.\
                              Got {0:d}".format(len(X)))

        # main algorithm
        self._fit_tfa(X)
