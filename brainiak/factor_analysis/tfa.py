"""Topographical Factor Analysis (TFA)

This implementation is based on the work:
.. [1] "Topographic factor analysis: a bayesian model for inferring brain
        networks from neural data"
   J. R. Manning, R. Ranganath, K. A. Norman, and D. M. Blei
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
from ..utils.utils import from_tri_2_sym, from_sym_2_tri
import numpy as np
import math
#import tfa_extension
import gc

__all__ = [
    "TFA",
]


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

    max_iter : int, default: 10
        Number of inner iterations to run the algorithm.

    threshold : float, default: 1.0
       Tolerance for terminate the parameter estimation

    K : int, default: 50
       Number of factors to compute

    nlss_method : {'trf', 'dogbox', 'lm'}, default: 'trf'
       Non-Linear least sqaure (NLSS) algorithm used by scipy.least_suqares to
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

    max_num_voxel : int, default: 5000
       The maximum number of voxels to subsample.

    max_num_tr : int, default: 500
       The maximum number of TRs to subsample.

    seed : int, default: 100
       Seed for subsample voxels and trs.

    two_step : boolean, default: True
       Whether to estimate centers/widths in two steps.

    verbose : boolean, default: False
       Verbose mode flag.


    --------

    """

    def __init__(
            self,
            R,
            sample_scaling,
            max_iter=10,
            threshold=0.01,
            K=50,
            nlss_method='trf',
            nlss_loss='soft_l1',
            jac='2-point',
            x_scale='jac',
            tr_solver=None,
            weight_method='rr',
            upper_ratio=1.8,
            lower_ratio=0.02,
            max_num_tr=500,
            max_num_voxel=5000,
            seed=100,
            two_step=True,
            verbose=False):
        self.R = R
        self.sample_scaling = sample_scaling
        self.miter = max_iter
        self.threshold = threshold
        self.K = K
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.jac = jac
        self.x_scale = x_scale
        self.tr_solver = tr_solver
        self.weight_method = weight_method
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.max_num_tr = max_num_tr
        self.max_num_voxel = max_num_voxel
        self.seed = seed
        self.two_step = two_step
        self.verbose = verbose
        self.n_dim = self.R.shape[1]
        self.cov_vec_size = np.sum(np.arange(self.n_dim) + 1)
        self.map_offset = self.get_map_offset()
        self.bounds = self.get_bounds(self.R)

    def set_K(self, K):
        """set K for this subject

        Parameters
        ----------

        K : integer
            Number of latent factor.


        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.K = K
        return self

    def set_R(self, R):
        """set R for this subject

        Parameters
        ----------

        R : 2D arary, in shape [n_voxel, n_dim]
            The coordinate matrix of subject's fMRI data.


        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.R = R
        return self

    def set_prior(self, prior):
        """set prior for this subject

        Parameters
        ----------

        prior : 1D array, with K*(n_dim+1) elements
            Subject prior of centers and widths.


        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.local_prior = prior
        return self

    def set_seed(self, seed):
        """set seed for this subject

        Parameters
        ----------

        seed : int
            Seed for subsample voxels and trs

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.seed = seed
        return self

    def init_prior(self):
        """initialize prior for this subject

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        centers, widths = self.init_centers_widths(self.R)
        # update prior
        prior = np.zeros(self.K * (self.n_dim + 1))
        print(centers.shape)
        print(self.map_offset[1])
        prior[0:self.map_offset[1]] = centers.ravel()
        prior[self.map_offset[1]:self.map_offset[2]] = widths.ravel()
        self.set_prior(prior)
        return self

    def _assign_posterior(self):
        """Minimum weight matching between prior and posterior,
           assign posterior to the right prior.

        Returns
        -------

        self : object
            Returns the instance itself.
        """

        prior_centers = self.local_prior[
            0:self.map_offset[1]].reshape(
            self.K,
            self.n_dim)
        posterior_centers = self.local_posterior[
            0:self.map_offset[1]].reshape(
            self.K,
            self.n_dim)
        posterior_widths = self.local_posterior[
            self.map_offset[1]:self.map_offset[2]] .reshape(
            self.K,
            1)
        # linear assignment on centers
        cost = distance.cdist(prior_centers, posterior_centers, 'euclidean')
        _, col_ind = linear_sum_assignment(cost)
        # reorder centers/widths based on cost assignment
        self.local_posterior[
            0:self.map_offset[1]] = posterior_centers[col_ind].ravel()
        self.local_posterior[self.map_offset[1]:self.map_offset[2]] = \
            posterior_widths[col_ind].ravel()
        return self

    def _converged(self):
        """Check convergence based on maximum absolute difference

        Returns
        -------

        converged : boolean
            Whether the parameter estimation converged.

        max_diff : float
            Maximum absolute difference between prior and posterior.

        """
        diff = self.local_prior - self.local_posterior
        max_diff = np.max(np.fabs(diff))
        if self.verbose:
            _, mse = self._mse_converged()
            diff_ratio = np.sum(diff**2) / np.sum(self.local_posterior**2)
            print(
                'tfa prior posterior max diff %f mse %f diff_ratio %f' %
                ((max_diff, mse, diff_ratio)))
        if max_diff > self.threshold:
            return False, max_diff
        else:
            return True, max_diff

    def _mse_converged(self):
        """Check convergence based on mean squared error

        Returns
        -------

        converged : boolean
            Whether the parameter estimation converged.

        mse : float
            Mean squared error between prior and posterior.

        """

        mse = mean_squared_error(self.local_prior, self.local_posterior,
                                 multioutput='uniform_average')
        if mse > self.threshold:
            return False, mse
        else:
            return True, mse

    def get_map_offset(self):
        """Compute offset of global prior


        Returns
        -------

        map_offest : 1D array
            The offset to different fields in global prior

        """

        nfield = 4
        self.map_offset = np.zeros(nfield).astype(int)
        field_size = self.K * np.array([self.n_dim, 1, self.cov_vec_size, 1])
        for i in np.arange(nfield - 1) + 1:
            self.map_offset[i] = self.map_offset[i - 1] + field_size[i - 1]
        return self.map_offset

    def init_centers_widths(self, R):
        """Initialize global prior of centers and widths

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

    def get_global_prior(self, R):
        """Compute global prior

        Parameters
        ----------
        R : 2D array, in format [n_voxel, n_dim]
            The coordinate matrix of one subject's fMRI data

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        centers, widths = self.init_centers_widths(R)
        self.global_prior = np.zeros(
            self.K * (self.n_dim + 2 + self.cov_vec_size))
        self.global_const = np.zeros(self.K * (self.cov_vec_size + 1))
        center_cov = np.cov(R.T) * math.pow(self.K, -2 / 3.0)
        # print center_cov
        center_cov_all = np.tile(from_sym_2_tri(center_cov), self.K)
        width_var = math.pow(np.nanmax(np.std(R, axis=0)), 2)
        width_var_all = np.tile(width_var, self.K)
        # center mean mean
        self.global_prior[0:self.map_offset[1]] = centers.ravel()
        # width mean mean
        self.global_prior[
            self.map_offset[1]:self.map_offset[2]] = widths.ravel()
        # center mean cov
        self.global_prior[
            self.map_offset[2]:self.map_offset[3]] = center_cov_all.ravel()
        # width mean var
        self.global_prior[self.map_offset[3]:] = width_var_all.ravel()
        # center cov
        self.global_const[
            0:self.K *
            self.cov_vec_size] = center_cov_all.ravel()
        # width var
        self.global_const[self.K * self.cov_vec_size:] = width_var_all.ravel()
        return self

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

    def _get_weights(self, data, F):
        """Calculate weight matrix based on fMRI data and factors

        Parameters
        ----------

        data : 2D array, with shape [n_voxel, n_tr]
            fMRI data from one subject

        F : 2D array, with shape [n_voxel,self.K]
            The latent factors from fMRI data.


        Returns
        -------

        W : 2D array, with shape [K, n_tr]
            The weight matrix from fMRI data.

        """

        beta = np.var(data)
        trans_F = F.T.copy()
        W = np.zeros((self.K, data.shape[1]))
        if self.weight_method == 'rr':
            W = np.linalg.solve(
                trans_F.dot(F) +
                beta *
                np.identity(self.K),
                trans_F.dot(data))
        elif self.weight_method == 'ols':
            W = np.linalg.solve(trans_F.dot(F), trans_F.dot(data))
        else:
            print("unknow weight_method")
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

    def get_bounds(self, R):
        """Calculate lower and upper bounds for centers and widths

        Parameters
        ----------

        R : 2D array, with shape [n_voxel, n_dim]
            The coordinate matrix of fMRI data from one subject


        Returns
        -------

        bounds : 2-tuple of array_like, default: None
            The lower and upper bounds on factor's centers and widths.

        """

        diameter = self._get_diameter(R)
        final_lower = np.zeros(self.K * (self.n_dim + 1))
        final_lower[
            0:self.K *
            self.n_dim] = np.tile(
            np.nanmin(
                R,
                axis=0),
            self.K)
        final_lower[
            self.K *
            self.n_dim:] = np.repeat(
            self.lower_ratio *
            diameter,
            self.K)
        final_upper = np.zeros(self.K * (self.n_dim + 1))
        final_upper[
            0:self.K *
            self.n_dim] = np.tile(
            np.nanmax(
                R,
                axis=0),
            self.K)
        final_upper[
            self.K *
            self.n_dim:] = np.repeat(
            self.upper_ratio *
            diameter,
            self.K)
        bounds = (final_lower, final_upper)
        return bounds

    def _residual_multivariate(
            self,
            estimate,
            unique_R,
            inds,
            X,
            W,
            global_centers,
            global_center_mean_cov,
            global_widths,
            global_width_mean_var_reci,
            data_sigma):
        """Residual function for estimating centers and widths

        Parameters
        ----------

        estimate : 1D array
            Initial estimation on centers

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

        global_widths: 1D array
            The global prior on widths

        global_width_mean_var_reci: 1D array
            The reciprocal of global prior on variance of widths' mean

        data_sigma: float
            The variance of X.


        Returns
        -------

        final_err : 1D array
            The residual function for estimating centers.

        """

        centers = estimate[:self.K * self.n_dim].reshape((self.K, self.n_dim))
        widths = estimate[self.K * self.n_dim:].reshape((self.K, 1))
        F = np.zeros((len(inds[0]), self.K))

        recon = X.size
        other_err = 0 if global_centers is None else (2 * self.K)
        final_err = np.zeros(recon + other_err)

        F = self._get_factors(unique_R, inds, centers, widths)
        final_err[0:recon] = (data_sigma * (X - F.dot(W))).ravel()
        """
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
        tfa_extension.recon(final_err[0:recon], X, F, W, sigma)
        """

        if other_err > 0:
            # center error
            for k in np.arange(self.K):
                diff = (centers[k] - global_centers[k])
                cov = from_tri_2_sym(global_center_mean_cov[k], self.n_dim)
                final_err[
                    recon +
                    k] = math.sqrt(
                    self.sample_scaling *
                    diff.dot(
                        np.linalg.solve(
                            cov,
                            diff.T)))

            # width error
            base = recon + self.K
            final_err[base:] = np.sqrt(self.sample_scaling *
                                       (global_width_mean_var_reci *
                                        (widths -
                                         global_widths)**2).ravel())

        return final_err

    def _residual_center_multivariate(
            self,
            estimate,
            widths,
            unique_R,
            inds,
            X,
            W,
            global_centers,
            global_center_mean_cov,
            data_sigma):
        """Residual function for estimating centers

        Parameters
        ----------

        estimate : 1D array
            Initial estimation on centers

        widths : 1D array
            Current estimation of widths.

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

        data_sigma: float
            The variance of X.


        Returns
        -------

        final_err : 1D array
            The residual function for estimating centers.

        """

        centers = estimate.reshape((self.K, self.n_dim))
        F = np.zeros((len(inds[0]), self.K))
        recon = X.size
        other_err = 0 if global_centers is None else self.K
        final_err = np.zeros(recon + other_err)
        F = self._get_factors(unique_R, inds, centers, widths)
        final_err[0:recon] = (data_sigma * (X - F.dot(W))).ravel()
        """
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
        tfa_extension.recon(final_err[0:recon], X, F, W, sigma)
        """
        if other_err > 0:
            # center error
            for k in np.arange(self.K):
                diff = (centers[k] - global_centers[k])
                cov = from_tri_2_sym(global_center_mean_cov[k], self.n_dim)
                final_err[
                    recon +
                    k] = math.sqrt(
                    self.sample_scaling *
                    diff.dot(
                        np.linalg.solve(
                            cov,
                            diff.T)))
        return final_err

    def _residual_width_multivariate(
            self,
            estimate,
            centers,
            unique_R,
            inds,
            X,
            W,
            global_widths,
            global_width_mean_var_reci,
            data_sigma):
        """Residual function for estimating widths

        Parameters
        ----------

        estimate : 1D array
            Initial estimation on widths

        centers : 1D array
            Current estimation of centers.

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

        data_sigma: float
            The variance of X.


        Returns
        -------

        final_err : 1D array
            The residual function for estimating widths.

        """
        widths = estimate.reshape((self.K, 1))
        F = np.zeros((len(inds[0]), self.K))
        recon = X.size
        other_err = 0 if global_widths is None else self.K
        final_err = np.zeros(recon + other_err)
        F = self._get_factors(unique_R, inds, centers, widths)
        final_err[0:recon] = (data_sigma * (X - F.dot(W))).ravel()
        """
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
        tfa_extension.recon(final_err[0:recon], X, F, W, sigma)
        """
        if other_err > 0:
            # width error
            final_err[recon:] = np.sqrt(self.sample_scaling *
                                        (global_width_mean_var_reci *
                                         (widths -
                                          global_widths)**2).ravel())
        return final_err

    def _get_centers_widths(
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
            global_width_mean_var_reci):
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


        Returns
        -------

        final_estimate.x: 1D array
            The newly estimated centers and widths.

        final_estimate.cost: float
            The cost value.

        """
        # least_squares only accept x in 1D format
        init_estimate = np.hstack(
            (init_centers.ravel(), init_widths.ravel())).copy()
        data_sigma = 1.0 / math.sqrt(2.0) * np.std(X)
        final_estimate = least_squares(
            self._residual_multivariate,
            init_estimate,
            args=(
                unique_R,
                inds,
                X,
                W,
                global_centers,
                global_widths,
                global_center_mean_cov,
                global_width_mean_var_reci,
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
            global_center_mean_cov):
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
            args=(
                init_widths,
                unique_R,
                inds,
                X,
                W,
                global_centers,
                global_center_mean_cov,
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
            global_width_mean_var_reci):
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
            args=(
                init_centers,
                unique_R,
                inds,
                X,
                W,
                global_widths,
                global_width_mean_var_reci,
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

    def _fit_tfa(self, data, global_prior=None):
        """TFA main algorithm

        Parameters
        ----------

        data: 2D array, in shape [n_voxel, n_tr]
            The fMRI data from one subject.

        global_prior : 1D array,
            The global prior on centers and widths.


        Returns
        -------

        self : object
            Returns the instance itself.

        """
        if global_prior is None:
            global_centers = None
            global_widths = None
            global_center_mean_cov = None
            global_width_mean_var_reci = None
        else:
            global_centers = global_prior[0:self.map_offset[1]].copy()\
                .reshape(self.K, self.n_dim)
            global_widths = global_prior[
                self.map_offset[1]:self.map_offset[2]].copy().reshape(
                self.K,
                1)
            global_center_mean_cov = global_prior[
                self.map_offset[2]:self.map_offset[3]].copy().reshape(
                self.K,
                self.cov_vec_size)
            global_width_mean_var_reci = 1.0 / \
                (global_prior[self.map_offset[3]:].copy().reshape(self.K, 1))
        inner_converged = False
        np.random.seed(self.seed)
        n = 0
        while n < self.miter and not inner_converged:
            self._fit_tfa_inner(
                data,
                global_centers,
                global_widths,
                global_center_mean_cov,
                global_width_mean_var_reci)
            self._assign_posterior()
            inner_converged, _ = self._converged()
            if not inner_converged:
                self.local_prior = self.local_posterior
            else:
                print("TFA converged at %d iteration." % (n))
            n += 1
            print('.')
            gc.collect()
        return self

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
        unique_R = []
        inds = []
        for d in np.arange(self.n_dim):
            tmp_unique, tmp_inds = np.unique(R[:, d], return_inverse=True)
            unique_R.append(tmp_unique)
            inds.append(tmp_inds)
        return unique_R, inds

    def _fit_tfa_inner(
            self,
            data,
            global_centers,
            global_widths,
            global_center_mean_cov,
            global_width_mean_var_reci):
        """Fit TFA model, the inner loop part

        Parameters
        ----------

        data: 2D array, in shape [n_voxel, n_tr]
            The fMRI data of a subject

        global_centers: 1D array
            The global prior on centers

        global_widths: 1D array
            The global prior on widths

        global_center_mean_cov: 2D array, with shape [K, cov_size]
            The global prior on centers' mean

        global_width_mean_var_reci: 1D array
            The reciprocal of global prior on variance of widths' mean


        Returns
        -------

        self : object
            Returns the instance itself.

        """
        nfeature = data.shape[0]
        nsample = data.shape[1]
        feature_indices = np.random.choice(
            nfeature,
            self.max_num_voxel,
            replace=False)
        sample_features = np.zeros(nfeature).astype(bool)
        sample_features[feature_indices] = True
        samples_indices = np.random.choice(
            nsample,
            self.max_num_tr,
            replace=False)
        sample_samples = np.zeros(nsample).astype(bool)
        sample_samples[samples_indices] = True
        curr_data = np.zeros(
            (self.max_num_voxel, self.max_num_tr)).astype(float)
        curr_data = data[feature_indices]
        curr_data = curr_data[:, samples_indices].copy()
        curr_R = self.R[feature_indices].copy()
        centers = self.local_prior[
            0:self.K *
            self.n_dim].reshape(
            self.K,
            self.n_dim)
        widths = self.local_prior[self.K * self.n_dim:].reshape(self.K, 1)
        unique_R, inds = self._get_unique_R(curr_R)
        F = np.zeros((len(inds[0]), self.K))
        F = self._get_factors(unique_R, inds, centers, widths)
        """
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
        """
        W = self._get_weights(curr_data, F)
        if self.two_step:
            local_centers, center_cost = self._get_centers(
                unique_R, inds, curr_data, W, centers, widths,
                global_centers, global_center_mean_cov)
            local_widths, width_cost = self._get_widths(
                unique_R, inds, curr_data, W, local_centers.reshape(
                    self.K, self.n_dim), widths, global_widths,
                global_width_mean_var_reci)
            self.local_posterior = np.hstack((local_centers, local_widths))
        else:
            self.local_posterior, total_cost = self._get_centers_widths(
                unique_R, inds, curr_data, W, centers, widths,
                global_centers, global_center_mean_cov,
                global_widths, global_width_mean_var_reci)
        return self

    def fit(self, X, y=None):
        """ Topographical Factor Analysis (TFA)

        Parameters
        ----------
        X : 2D array, in shape [n_voxel, n_sample]
            The fMRI data of one subject

        y : None or 1D array
            The global prior when fitting TFA for HTFA
            None when fitting TFA alone
        """
        if self.verbose:
            print('Start to fit TFA ')

        if not isinstance(X, np.ndarray):
            raise TypeError("Input data should be an array")
        if X.ndim != 2:
            raise ValueError("Input data should be 2D array")

        # main algorithm
        self._fit_tfa(X, y)
        return self
