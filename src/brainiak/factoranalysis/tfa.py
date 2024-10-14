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
"""Topographical Factor Analysis (TFA)

This implementation is based on the work in [Manning2014]_ and
[AndersonM2016]_.

.. [Manning2014] "Topographic factor analysis: a bayesian model for inferring
   brain networks from neural data", J. R. Manning, R. Ranganath, K. A. Norman,
   and D. M. Blei.PLoS One, vol. 9, no. 5, 2014.

.. [AndersonM2016] "Scaling Up Multi-Subject Neuroimaging Factor Analysis"
   Michael J. Anderson, Mihai Capota, Javier S. Turek, Xia Zhu,
   Theodore L. Willke, Yida Wang, Po-Hsuan Chen, Jeremy R. Manning,
   Peter J. Ramadge, and Kenneth A. Norman
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
from . import tfa_extension  # type: ignore
import numpy as np
import math
import gc
import logging

__all__ = [
    "TFA",
]

logger = logging.getLogger(__name__)


class TFA(BaseEstimator):

    """Topographical Factor Analysis (TFA)

    Given data from one subject, factorize it as a spatial factor F and
    a weight matrix W.

    Parameters
    ----------

    max_iter : int, default: 10
        Number of iterations to run the algorithm.

    threshold : float, default: 1.0
       Tolerance for terminating the parameter estimation

    K : int, default: 50
       Number of factors to compute

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
       The upper bound of the ratio between factor's width
       and maximum sigma of scanner coordinates.

    lower_ratio : float, default: 0.02
       The lower bound of the ratio between factor's width
       and maximum sigma of scanner coordinates.

    max_num_voxel : int, default: 5000
       The maximum number of voxels to subsample.

    max_num_tr : int, default: 500
       The maximum number of TRs to subsample.

    seed : int, default: 100
       Seed for subsampling voxels and trs.

    verbose : boolean, default: False
       Verbose mode flag.


    Attributes
    ----------
    local_posterior_ : 1D array
        Local posterior on subject's centers and widths

    F_ : 2D array, in shape [n_voxel, K]
        Latent factors of the subject

    W_ : 2D array, in shape [K, n_tr]
        Weight matrix of the subject


    """

    def __init__(
            self,
            max_iter=10,
            threshold=1.0,
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
            verbose=False):
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
        self.verbose = verbose

    def set_K(self, K):
        """set K for the subject

        Parameters
        ----------

        K : integer
            Number of latent factor.


        Returns
        -------
        TFA
            Returns the instance itself.

        """
        self.K = K
        return self

    def set_prior(self, prior):
        """set prior for the subject

        Parameters
        ----------

        prior : 1D array, with K*(n_dim+1) elements
            Subject prior of centers and widths.


        Returns
        -------
        TFA
            Returns the instance itself.

        """
        self.local_prior = prior
        return self

    def set_seed(self, seed):
        """set seed for the subject

        Parameters
        ----------

        seed : int
            Seed for subsampling voxels and trs

        Returns
        -------
        TFA
            Returns the instance itself.

        """
        self.seed = seed
        return self

    def init_prior(self, R):
        """initialize prior for the subject

        Returns
        -------
        TFA
            Returns the instance itself.

        """
        centers, widths = self.init_centers_widths(R)
        # update prior
        prior = np.zeros(self.K * (self.n_dim + 1))
        self.set_centers(prior, centers)
        self.set_widths(prior, widths)
        self.set_prior(prior)
        return self

    def _assign_posterior(self):
        """assign posterior to prior based on Hungarian algorithm

        Returns
        -------
        TFA
            Returns the instance itself.
        """

        prior_centers = self.get_centers(self.local_prior)
        posterior_centers = self.get_centers(self.local_posterior_)
        posterior_widths = self.get_widths(self.local_posterior_)
        # linear assignment on centers
        cost = distance.cdist(prior_centers, posterior_centers, 'euclidean')
        _, col_ind = linear_sum_assignment(cost)
        # reorder centers/widths based on cost assignment
        self.set_centers(self.local_posterior_, posterior_centers[col_ind])
        self.set_widths(self.local_posterior_, posterior_widths[col_ind])
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
        diff = self.local_prior - self.local_posterior_
        max_diff = np.max(np.fabs(diff))
        if self.verbose:
            _, mse = self._mse_converged()
            diff_ratio = np.sum(diff ** 2) / np.sum(self.local_posterior_ ** 2)
            logger.info(
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

        mse = mean_squared_error(self.local_prior, self.local_posterior_,
                                 multioutput='uniform_average')
        if mse > self.threshold:
            return False, mse
        else:
            return True, mse

    def get_map_offset(self):
        """Compute offset of prior/posterior


        Returns
        -------

        map_offest : 1D array
            The offset to different fields in prior/posterior

        """

        nfield = 4
        self.map_offset = np.zeros(nfield).astype(int)
        field_size = self.K * np.array([self.n_dim, 1, self.cov_vec_size, 1])
        for i in np.arange(nfield - 1) + 1:
            self.map_offset[i] = self.map_offset[i - 1] + field_size[i - 1]
        return self.map_offset

    def init_centers_widths(self, R):
        """Initialize prior of centers and widths

        Returns
        -------

        centers : 2D array, with shape [K, n_dim]
            Prior of factors' centers.

        widths : 1D array, with shape [K, 1]
            Prior of factors' widths.

        """

        kmeans = KMeans(
            init='k-means++',
            n_clusters=self.K,
            n_init=10,
            random_state=100)
        kmeans.fit(R)
        centers = kmeans.cluster_centers_
        widths = self._get_max_sigma(R) * np.ones((self.K, 1))
        return centers, widths

    def get_template(self, R):
        """Compute a template on latent factors

        Parameters
        ----------
        R : 2D array, in format [n_voxel, n_dim]
            The scanner coordinate matrix of one subject's fMRI data

        Returns
        -------
        template_prior : 1D array
            The template prior.

        template_centers_cov:  2D array,  in shape [n_dim, n_dim]
            The template on centers' covariance.

        template_widths_var: float
            The template on widths' variance
        """

        centers, widths = self.init_centers_widths(R)
        template_prior =\
            np.zeros(self.K * (self.n_dim + 2 + self.cov_vec_size))
        # template centers cov and widths var are const
        template_centers_cov = np.cov(R.T) * math.pow(self.K, -2 / 3.0)
        template_widths_var = self._get_max_sigma(R)
        centers_cov_all = np.tile(from_sym_2_tri(template_centers_cov), self.K)
        widths_var_all = np.tile(template_widths_var, self.K)
        # initial mean of centers' mean
        self.set_centers(template_prior, centers)
        self.set_widths(template_prior, widths)
        self.set_centers_mean_cov(template_prior, centers_cov_all)
        self.set_widths_mean_var(template_prior, widths_var_all)
        return template_prior, template_centers_cov, template_widths_var

    def set_centers(self, estimation, centers):
        """Set estimation on centers

        Parameters
        ----------
        estimation : 1D arrary
            Either prior or posterior estimation

        centers : 2D array, in shape [K, n_dim]
            Estimation on centers

        """
        estimation[0:self.map_offset[1]] = centers.ravel()

    def set_widths(self, estimation, widths):
        """Set estimation on widths

        Parameters
        ----------
        estimation : 1D arrary
            Either prior of posterior estimation

        widths : 2D array, in shape [K, 1]
            Estimation on widths

        """
        estimation[self.map_offset[1]:self.map_offset[2]] = widths.ravel()

    def set_centers_mean_cov(self, estimation, centers_mean_cov):
        """Set estimation on centers

        Parameters
        ----------
        estimation : 1D arrary
            Either prior of posterior estimation

        centers : 2D array, in shape [K, n_dim]
            Estimation on centers

        """
        estimation[self.map_offset[2]:self.map_offset[3]] =\
            centers_mean_cov.ravel()

    def set_widths_mean_var(self, estimation, widths_mean_var):
        """Set estimation on centers

        Parameters
        ----------
        estimation : 1D arrary
            Either prior of posterior estimation

        centers : 2D array, in shape [K, n_dim]
            Estimation on centers

        """
        estimation[self.map_offset[3]:] = widths_mean_var.ravel()

    def get_centers(self, estimation):
        """Get estimation on centers

        Parameters
        ----------
        estimation : 1D arrary
            Either prior of posterior estimation


        Returns
        -------
        centers : 2D array, in shape [K, n_dim]
            Estimation on centers

        """
        centers = estimation[0:self.map_offset[1]]\
            .reshape(self.K, self.n_dim)
        return centers

    def get_widths(self, estimation):
        """Get estimation on widths


        Parameters
        ----------
        estimation : 1D arrary
             Either prior of posterior estimation


        Returns
        -------
        fields : 2D array, in shape [K, 1]
            Estimation of widths

        """
        widths = estimation[self.map_offset[1]:self.map_offset[2]]\
            .reshape(self.K, 1)
        return widths

    def get_centers_mean_cov(self, estimation):
        """Get estimation on the covariance of centers' mean


        Parameters
        ----------
        estimation : 1D arrary
             Either prior of posterior estimation


        Returns
        -------

        centers_mean_cov : 2D array, in shape [K, cov_vec_size]
            Estimation of the covariance of centers' mean

        """
        centers_mean_cov = estimation[self.map_offset[2]:self.map_offset[3]]\
            .reshape(self.K, self.cov_vec_size)
        return centers_mean_cov

    def get_widths_mean_var(self, estimation):
        """Get estimation on the variance of widths' mean


        Parameters
        ----------
        estimation : 1D arrary
             Either prior of posterior estimation


        Returns
        -------

        widths_mean_var : 2D array, in shape [K, 1]
            Estimation on variance of widths' mean

        """
        widths_mean_var = \
            estimation[self.map_offset[3]:].reshape(self.K, 1)
        return widths_mean_var

    def get_factors(self, unique_R, inds, centers, widths):
        """Calculate factors based on centers and widths

        Parameters
        ----------

        unique_R : a list of array,
            Each element contains unique value in one dimension of
            scanner coordinate matrix R.

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

        F = np.zeros((len(inds[0]), self.K))
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

        return F

    def get_weights(self, data, F):
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
            W = np.linalg.solve(trans_F.dot(F) + beta * np.identity(self.K),
                                trans_F.dot(data))
        else:
            W = np.linalg.solve(trans_F.dot(F), trans_F.dot(data))
        return W

    def _get_max_sigma(self, R):
        """Calculate maximum sigma of scanner RAS coordinates

        Parameters
        ----------

        R : 2D array, with shape [n_voxel, n_dim]
            The coordinate matrix of fMRI data from one subject

        Returns
        -------

        max_sigma : float
            The maximum sigma of scanner coordinates.

        """

        max_sigma = 2.0 * math.pow(np.nanmax(np.std(R, axis=0)), 2)
        return max_sigma

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

        max_sigma = self._get_max_sigma(R)
        final_lower = np.zeros(self.K * (self.n_dim + 1))
        final_lower[0:self.K * self.n_dim] =\
            np.tile(np.nanmin(R, axis=0), self.K)
        final_lower[self.K * self.n_dim:] =\
            np.repeat(self.lower_ratio * max_sigma, self.K)
        final_upper = np.zeros(self.K * (self.n_dim + 1))
        final_upper[0:self.K * self.n_dim] =\
            np.tile(np.nanmax(R, axis=0), self.K)
        final_upper[self.K * self.n_dim:] =\
            np.repeat(self.upper_ratio * max_sigma, self.K)
        bounds = (final_lower, final_upper)
        return bounds

    def _residual_multivariate(
            self,
            estimate,
            unique_R,
            inds,
            X,
            W,
            template_centers,
            template_centers_mean_cov,
            template_widths,
            template_widths_mean_var_reci,
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

        template_centers: 2D array, with shape [K, n_dim]
            The template prior on centers

        template_centers_mean_cov: 2D array, with shape [K, cov_size]
            The template prior on covariance of centers' mean

        template_widths: 1D array
            The template prior on widths

        template_widths_mean_var_reci: 1D array
            The reciprocal of template prior on variance of widths' mean

        data_sigma: float
            The variance of X.


        Returns
        -------

        final_err : 1D array
            The residual function for estimating centers.

        """

        centers = self.get_centers(estimate)
        widths = self.get_widths(estimate)
        recon = X.size
        other_err = 0 if template_centers is None else (2 * self.K)
        final_err = np.zeros(recon + other_err)
        F = self.get_factors(unique_R, inds, centers, widths)
        sigma = np.zeros((1,))
        sigma[0] = data_sigma
        tfa_extension.recon(final_err[0:recon], X, F, W, sigma)

        if other_err > 0:
            # center error
            for k in np.arange(self.K):
                diff = (centers[k] - template_centers[k])
                cov = from_tri_2_sym(template_centers_mean_cov[k], self.n_dim)
                final_err[recon + k] = math.sqrt(
                    self.sample_scaling *
                    diff.dot(np.linalg.solve(cov, diff.T)))

            # width error
            base = recon + self.K
            dist = template_widths_mean_var_reci *\
                (widths - template_widths) ** 2
            final_err[base:] = np.sqrt(self.sample_scaling * dist).ravel()

        return final_err

    def _estimate_centers_widths(
            self,
            unique_R,
            inds,
            X,
            W,
            init_centers,
            init_widths,
            template_centers,
            template_widths,
            template_centers_mean_cov,
            template_widths_mean_var_reci):
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

        template_centers: 1D array
            The template prior on centers

        template_widths: 1D array
            The template prior on widths

        template_centers_mean_cov: 2D array, with shape [K, cov_size]
            The template prior on centers' mean

        template_widths_mean_var_reci: 1D array
            The reciprocal of template prior on variance of widths' mean


        Returns
        -------

        final_estimate.x: 1D array
            The newly estimated centers and widths.

        final_estimate.cost: float
            The cost value.

        """
        # least_squares only accept x in 1D format
        init_estimate = np.hstack(
            (init_centers.ravel(), init_widths.ravel()))  # .copy()
        data_sigma = 1.0 / math.sqrt(2.0) * np.std(X)
        final_estimate = least_squares(
            self._residual_multivariate,
            init_estimate,
            args=(
                unique_R,
                inds,
                X,
                W,
                template_centers,
                template_widths,
                template_centers_mean_cov,
                template_widths_mean_var_reci,
                data_sigma),
            method=self.nlss_method,
            loss=self.nlss_loss,
            bounds=self.bounds,
            verbose=0,
            x_scale=self.x_scale,
            tr_solver=self.tr_solver)
        return final_estimate.x, final_estimate.cost

    def _fit_tfa(self, data, R, template_prior=None):
        """TFA main algorithm

        Parameters
        ----------

        data: 2D array, in shape [n_voxel, n_tr]
            The fMRI data from one subject.

        R : 2D array, in shape [n_voxel, n_dim]
            The voxel coordinate matrix of fMRI data

        template_prior : 1D array,
            The template prior on centers and widths.


        Returns
        -------
        TFA
            Returns the instance itself.

        """
        if template_prior is None:
            template_centers = None
            template_widths = None
            template_centers_mean_cov = None
            template_widths_mean_var_reci = None
        else:
            template_centers = self.get_centers(template_prior)
            template_widths = self.get_widths(template_prior)
            template_centers_mean_cov =\
                self.get_centers_mean_cov(template_prior)
            template_widths_mean_var_reci = 1.0 /\
                self.get_widths_mean_var(template_prior)
        inner_converged = False
        np.random.seed(self.seed)
        n = 0
        while n < self.miter and not inner_converged:
            self._fit_tfa_inner(
                data,
                R,
                template_centers,
                template_widths,
                template_centers_mean_cov,
                template_widths_mean_var_reci)
            self._assign_posterior()
            inner_converged, _ = self._converged()
            if not inner_converged:
                self.local_prior = self.local_posterior_
            else:
                logger.info("TFA converged at %d iteration." % (n))
            n += 1
            gc.collect()
        return self

    def get_unique_R(self, R):
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
            R,
            template_centers,
            template_widths,
            template_centers_mean_cov,
            template_widths_mean_var_reci):
        """Fit TFA model, the inner loop part

        Parameters
        ----------

        data: 2D array, in shape [n_voxel, n_tr]
            The fMRI data of a subject

        R : 2D array, in shape [n_voxel, n_dim]
            The voxel coordinate matrix of fMRI data

        template_centers: 1D array
            The template prior on centers

        template_widths: 1D array
            The template prior on widths

        template_centers_mean_cov: 2D array, with shape [K, cov_size]
            The template prior on covariance of centers' mean

        template_widths_mean_var_reci: 1D array
            The reciprocal of template prior on variance of widths' mean


        Returns
        -------
        TFA
            Returns the instance itself.

        """
        nfeature = data.shape[0]
        nsample = data.shape[1]
        feature_indices =\
            np.random.choice(nfeature, self.max_num_voxel, replace=False)
        sample_features = np.zeros(nfeature).astype(bool)
        sample_features[feature_indices] = True
        samples_indices =\
            np.random.choice(nsample, self.max_num_tr, replace=False)
        curr_data = np.zeros((self.max_num_voxel, self.max_num_tr))\
            .astype(float)
        curr_data = data[feature_indices]
        curr_data = curr_data[:, samples_indices].copy()
        curr_R = R[feature_indices].copy()
        centers = self.get_centers(self.local_prior)
        widths = self.get_widths(self.local_prior)
        unique_R, inds = self.get_unique_R(curr_R)
        F = self.get_factors(unique_R, inds, centers, widths)
        W = self.get_weights(curr_data, F)
        self.local_posterior_, self.total_cost = self._estimate_centers_widths(
            unique_R, inds, curr_data, W, centers, widths,
            template_centers, template_centers_mean_cov,
            template_widths, template_widths_mean_var_reci)

        return self

    def fit(self, X, R, template_prior=None):
        """ Topographical Factor Analysis (TFA)[Manning2014]

        Parameters
        ----------
        X : 2D array, in shape [n_voxel, n_sample]
            The fMRI data of one subject

        R : 2D array, in shape [n_voxel, n_dim]
            The voxel coordinate matrix of fMRI data

        template_prior : None or 1D array
            The template prior as an extra constraint
            None when fitting TFA alone
        """
        if self.verbose:
            logger.info('Start to fit TFA ')

        if not isinstance(X, np.ndarray):
            raise TypeError("Input data should be an array")
        if X.ndim != 2:
            raise TypeError("Input data should be 2D array")
        if not isinstance(R, np.ndarray):
            raise TypeError("Input coordinate matrix should be an array")
        if R.ndim != 2:
            raise TypeError("Input coordinate matrix should be 2D array")
        if X.shape[0] != R.shape[0]:
            raise TypeError(
                "The number of voxels should be the same in X and R!")
        if self.weight_method != 'rr' and self.weight_method != 'ols':
            raise ValueError(
                "only 'rr' and 'ols' are accepted as weight_method!")

        # main algorithm
        self.n_dim = R.shape[1]
        self.cov_vec_size = np.sum(np.arange(self.n_dim) + 1)
        self.map_offset = self.get_map_offset()
        self.bounds = self.get_bounds(R)
        n_voxel = X.shape[0]
        n_tr = X.shape[1]
        self.sample_scaling = 0.5 * float(
            self.max_num_voxel * self.max_num_tr) / float(n_voxel * n_tr)
        if template_prior is None:
            self.init_prior(R)
        else:
            self.local_prior = template_prior[0: self.map_offset[2]]
        self._fit_tfa(X, R, template_prior)
        if template_prior is None:
            centers = self.get_centers(self.local_posterior_)
            widths = self.get_widths(self.local_posterior_)
            unique_R, inds = self.get_unique_R(R)
            self.F_ = self.get_factors(unique_R, inds, centers, widths)
            self.W_ = self.get_weights(X, self.F_)
        return self
