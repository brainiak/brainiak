#  Copyright 2016 Mingbo Cai, Princeton Neuroscience Instititute,
#  Princeton University
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
""" Bayesian Representational Similarity Analysis (BRSA)

    This implementation is based on the following publications:

 .. [Cai2016] "A Bayesian method for reducing bias in neural
    representational similarity analysis",
    M.B. Cai, N. Schuck, J. Pillow, Y. Niv,
    Neural Information Processing Systems 29, 2016.
    A preprint is available at
    https://doi.org/10.1101/073932
"""

# Authors: Mingbo Cai
# Princeton Neuroscience Institute, Princeton University, 2016

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import warnings
import time
from sklearn.base import BaseEstimator
from sklearn.utils import assert_all_finite
import logging
import brainiak.utils.utils as utils
import scipy.spatial.distance as spdist
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

__all__ = [
    "BRSA",
]


class BRSA(BaseEstimator):
    """Bayesian representational Similarity Analysis (BRSA)

    Given the time series of neural imaging data in a region of interest
    (ROI) and the hypothetical neural response (design matrix) to
    each experimental condition of interest,
    calculate the shared covariance matrix of
    the voxels(recording unit)' response to each condition,
    and the relative SNR of each voxels.
    The relative SNR could be considered as the degree of contribution
    of each voxel to this shared covariance matrix.
    A correlation matrix converted from the covariance matrix
    will be provided as a quantification of neural representational similarity.

    .. math::
        Y = X \\cdot \\beta + \\epsilon

        \\beta_i \\sim N(0,(s_{i} \\sigma_{i})^2 U)

    Parameters
    ----------
    n_iter : int, default: 200
        Number of maximum iterations to run the algorithm.
    rank : int, default: None
        The rank of the covariance matrix.
        If not provided, the covariance matrix will be assumed
        to be full rank. When you have many conditions
        (e.g., calculating the similarity matrix of responses to each event),
        you might want to start with specifying a lower rank and use metrics
        such as AIC or BIC to decide the optimal rank.
    auto_nuisance: boolean, default: True
        In order to model spatial correlation between voxels that cannot
        be accounted for by common response captured in the design matrix,
        we assume that a set of time courses not related to the task
        conditions are shared across voxels with unknown amplitudes.
        One approach is for users to provide time series which they consider
        as nuisance but exist in the noise (such as head motion).
        The other way is to take the first n_nureg principal components
        in the residual after subtracting the response to the design matrix
        from the data, and use these components as the nuisance regressor.
        If this flag is turned on, the nuisance regressor provided by the
        user is used only in the first round of fitting. The PCs from
        residuals will be used in the next round of fitting.
        Note that nuisance regressor is not required from user. If it is
        not provided, DC components for each run will be used as nuisance
        regressor in the initial fitting.
    n_nureg: int, default: 6
        Number of nuisance regressors to use in order to model signals
        shared across voxels not captured by the design matrix.
        This parameter will not be effective in the first round of fitting.
    GP_space: boolean, default: False
        Whether to impose a Gaussion Process (GP) prior on the log(pseudo-SNR).
        If true, the GP has a kernel defined over spatial coordinate.
        This is experimental and slow. I find that when SNR is genrally low,
        smoothness can be overestimated. But I think this is better than
        not imposing any regularization.
    GP_inten: boolean, defualt: False
        Whether to include a kernel definved over the intensity of image.
        GP_space should be True as well if you want to use this,
        because the smoothness should be primarily in space.
        Smoothness in intensity is just complementary.
    tol: tolerance parameter passed to the minimizer.
    verbose : boolean, default: False
        Verbose mode flag.
    eta: a small number added to the diagonal element of the
        covariance matrix in the Gaussian Process prior. This is
        to ensure that the matrix is invertible.
    space_smooth_range: the distance (in unit the same as what
        you would use when supplying the spatial coordiates of
        each voxel, typically millimeter) which you believe is
        the maximum range of the length scale parameter of
        Gaussian Process defined over voxel location. This is
        used to impose a half-Cauchy prior on the length scale.
        If not provided, the program will set it to half of the
        maximum distance between all voxels.
    inten_smooth_range: the difference in image intensity which
        you believe is the maximum range of plausible length
        scale for the Gaussian Process defined over image
        intensity. Length scales larger than this are allowed,
        but will be penalized. If not supplied, this parameter
        will be set to half of the maximal intensity difference.
    tau_range: the reasonable range of the standard deviation
        of the Gaussian Process. Since the Gaussian Process is
        imposed on the log(SNR), this range should not be too
        large. 10 is a pretty loose range. This parameter is
        used in a half-Cauchy prior on the standard deviation
    init_iter: how many initial iterations to fit the model
        without introducing the GP prior before fitting with it,
        if GP_space or GP_inten is requested. This initial
        fitting is to give the parameters a good starting point.
    optimizer: the optimizer to use for minimizing cost function.
        We use 'BFGS' as a default. Users can try other optimizer
        coming with scipy.optimize.minimize, or a custom
        optimizer.
    rand_seed : int, default: 0
        Seed for initializing the random number generator.

    Attributes
    ----------
    U_ : The shared covariance matrix, shape=[condition,condition].
    L_ : The Cholesky factor of the shared covariance matrix
        (lower-triangular matrix), shape=[condition,condition].
    C_: the correlation matrix derived from the shared covariance matrix,
        shape=[condition,condition]
    nSNR_ : array, shape=[voxels,]
        The pseuso-SNR of all voxels.
        They are normalized such that the geometric mean is 1
    sigma_ : array, shape=[voxels,]
        The estimated standard deviation of the noise in each voxel
        Assuming AR(1) model, this means the standard deviation
        of the refreshing noise.
    rho_ : array, shape=[voxels,]
        The estimated autoregressive coefficient of each voxel
    bGP_ : scalar, only if GP_space or GP_inten is True.
        the standard deviation of the GP prior
    lGPspace_ : scalar, only if GP_space or GP_inten is True
        the length scale of Gaussian Process prior of log(SNR)
    lGPinten_: scalar, only if GP_inten is True
        the length scale in fMRI intensity of the GP prior of log(SNR)
    beta_: array, shape=[conditions, voxels]
        The maximum a posterior estimation of the response amplitudes
        of each voxel to each task condition.
    beta0_: array, shape=[n_nureg, voxels]
        The loading weights of each voxel for the shared time courses
        not captured by the design matrix.

    """

    def __init__(
            self, n_iter=50, rank=None, GP_space=False, GP_inten=False,
            tol=2e-3, auto_nuisance=True, n_nureg=6, verbose=False,
            eta=0.0001, space_smooth_range=None, inten_smooth_range=None,
            tau_range=10.0, init_iter=20, optimizer='BFGS', rand_seed=0):
        self.n_iter = n_iter
        self.rank = rank
        self.GP_space = GP_space
        self.GP_inten = GP_inten
        self.tol = tol
        self.auto_nuisance = auto_nuisance
        self.n_nureg = n_nureg
        self.verbose = verbose
        self.eta = eta
        # This is a tiny ridge added to the Gaussian Process
        # covariance matrix template to gaurantee that it is invertible.
        # Mathematically it means we assume that this proportion of the
        # variance is always independent between voxels for the log(SNR2).
        self.space_smooth_range = space_smooth_range
        self.inten_smooth_range = inten_smooth_range
        # The kernel of the Gaussian Process is the product of a kernel
        # defined on spatial coordinate and a kernel defined on
        # image intensity.
        self.tau_range = tau_range
        self.init_iter = init_iter
        # When imposing smoothness prior, fit the model without this
        # prior for this number of iterations.
        self.optimizer = optimizer
        self.rand_seed = rand_seed
        return

    def fit(self, X, design, nuisance=None, scan_onsets=None, coords=None,
            inten=None):
        """Compute the Bayesian RSA

        Parameters
        ----------
        X: 2-D numpy array, shape=[time_points, voxels]
            If you have multiple scans of the same participants that you
            want to analyze together, you should concatenate them along
            the time dimension after proper preprocessing (e.g. spatial
            alignment), and specify the onsets of each scan in scan_onsets.
        design: 2-D numpy array, shape=[time_points, conditions]
            This is the design matrix. It should only include the hypothetic
            response for task conditions. You do not need to include
            regressors for a DC component or motion parameters, unless with
            a strong reason. If you want to model head motion,
            you should include them in nuisance regressors.
            If you have multiple run, the design matrix
            of all runs should be concatenated along the time dimension,
            with one column across runs for each condition.
        nuisance: optional, 2-D numpy array,
            shape=[time_points, nuisance_factors]
            The responses to these regressors will be marginalized out from
            each voxel, which means they are considered, but won't be assumed
            to share the same pseudo-SNR map with the design matrix.
            Therefore, the pseudo-SNR map will only reflect the
            relative contribution of design matrix to each voxel.
            You can provide time courses such as those for head motion
            to this parameter.
            Note that if auto_nuisance is set to True, this input
            will only be used in the first round of fitting. The first
            n_nureg principal components of residual (excluding the response
            to the design matrix) will be used as the nuisance regressor
            for the second round of fitting.
            If auto_nuisance is set to False, the nuisance regressors supplied
            by the users together with DC components will be used as
            nuisance time series.
        scan_onsets: optional, an 1-D numpy array, shape=[runs,]
            This specifies the indices of X which correspond to the onset
            of each scanning run. For example, if you have two experimental
            runs of the same subject, each with 100 TRs, then scan_onsets
            should be [0,100].
            If you do not provide the argument, the program will
            assume all data are from the same run.
            The effect of them is to make the inverse matrix
            of the temporal covariance matrix of noise block-diagonal.
        coords: optional, 2-D numpy array, shape=[voxels,3]
            This is the coordinate of each voxel,
            used for implementing Gaussian Process prior.
        inten: optional, 1-D numpy array, shape=[voxel,]
            This is the average fMRI intensity in each voxel.
            It should be calculated from your data without any preprocessing
            such as z-scoring. Because it should reflect
            whether a voxel is bright (grey matter) or dark (white matter).
            A Gaussian Process kernel defined on both coordinate and intensity
            imposes a smoothness prior on adjcent voxels
            but with the same tissue type. The Gaussian Process
            is experimental and has shown good performance on
            some visual datasets.
        """

        logger.info('Running Bayesian RSA')

        assert not self.GP_inten or (self.GP_inten and self.GP_space),\
            'You must speficiy GP_space to True'\
            'if you want to use GP_inten'

        # Check input data
        assert_all_finite(X)
        assert X.ndim == 2, 'The data should be 2 dimension ndarray'

        assert np.all(np.std(X, axis=0) > 0),\
            'The time courses of some voxels do not change at all.'\
            ' Please make sure all voxels are within the brain'

        # check design matrix
        assert_all_finite(design)
        assert design.ndim == 2,\
            'The design matrix should be 2 dimension ndarray'
        assert np.linalg.matrix_rank(design) == design.shape[1], \
            'Your design matrix has rank smaller than the number of'\
            ' columns. Some columns can be explained by linear '\
            'combination of other columns. Please check your design matrix.'
        assert np.size(design, axis=0) == np.size(X, axis=0),\
            'Design matrix and data do not '\
            'have the same number of time points.'
        assert self.rank is None or self.rank <= design.shape[1],\
            'Your design matrix has fewer columns than the rank you set'

        # Check the nuisance regressors.
        if nuisance is not None:
            assert_all_finite(nuisance)
            assert nuisance.ndim == 2,\
                'The nuisance regressor should be 2 dimension ndarray'
            assert np.linalg.matrix_rank(nuisance) == nuisance.shape[1], \
                'The nuisance regressor has rank smaller than the number of'\
                'columns. Some columns can be explained by linear '\
                'combination of other columns. Please check your nuisance' \
                'regressors.'
            assert np.size(nuisance, axis=0) == np.size(X, axis=0), \
                'Nuisance regressor and data do not have the same '\
                'number of time points.'
        # check scan_onsets validity
        assert scan_onsets is None or\
            (np.max(scan_onsets) <= X.shape[0] and np.min(scan_onsets) >= 0),\
            'Some scan onsets provided are out of the range of time points.'

        # check the size of coords and inten
        if self.GP_space:
            logger.info('Fitting with Gaussian Process prior on log(SNR)')
            assert coords is not None and coords.shape[0] == X.shape[1],\
                'Spatial smoothness was requested by setting GP_space. '\
                'But the voxel number of coords does not match that of '\
                'data X, or voxel coordinates are not provided. '\
                'Please make sure that coords is in the shape of '\
                '[ n_voxel x 3].'
            assert coords.ndim == 2,\
                'The coordinate matrix should be a 2-d array'
            if self.GP_inten:
                assert inten is not None and inten.shape[0] == X.shape[1],\
                    'The voxel number of intensity does not '\
                    'match that of data X, or intensity not provided.'
                assert np.var(inten) > 0,\
                    'All voxels have the same intensity.'
        if (not self.GP_space and coords is not None) or\
                (not self.GP_inten and inten is not None):
            logger.warning('Coordinates or image intensity provided'
                           ' but GP_space or GP_inten is not set '
                           'to True. The coordinates or intensity are'
                           ' ignored.')
        # Run Bayesian RSA
        # Note that we have a change of notation here. Within _fit_RSA_UV,
        # design matrix is named X and data is named Y, to reflect the
        # generative model that data Y is generated by mixing the response
        # X to experiment conditions and other neural activity.
        # However, in fit(), we keep the tradition of scikit-learn that
        # X is the input data to fit and y, a reserved name not used, is
        # the label to map to from X.
        if not self.GP_space:
            # If GP_space is not requested, then the model is fitted
            # without imposing any Gaussian Process prior on log(SNR^2)
            self.U_, self.L_, self.nSNR_, self.beta_, self.beta0_,\
                self.sigma_, self.rho_ = \
                self._fit_RSA_UV(X=design, Y=X, X0=nuisance,
                                 scan_onsets=scan_onsets)
        elif not self.GP_inten:
            # If GP_space is requested, but GP_inten is not, a GP prior
            # based on spatial locations of voxels will be imposed.
            self.U_, self.L_, self.nSNR_, self.beta_, self.beta0_,\
                self.sigma_, self.rho_, self.lGPspace_, self.bGP_ \
                = self._fit_RSA_UV(
                    X=design, Y=X, X0=nuisance,
                    scan_onsets=scan_onsets, coords=coords)
        else:
            # If both self.GP_space and self.GP_inten are True,
            # a GP prior based on both location and intensity is imposed.
            self.U_, self.L_, self.nSNR_, self.beta_, self.beta0_,\
                self.sigma_, self.rho_, self.lGPspace_, self.bGP_,\
                self.lGPinten_ = \
                self._fit_RSA_UV(X=design, Y=X, X0=nuisance,
                                 scan_onsets=scan_onsets,
                                 coords=coords, inten=inten)

        self.C_ = utils.cov2corr(self.U_)
        return self

    # The following 2 functions _D_gen and _F_gen generate templates used
    # for constructing inverse of covariance matrix of AR(1) noise
    # The inverse of covarian matrix is
    # (I - rho1 * D + rho1**2 * F) / sigma**2. D is a matrix where all the
    # elements adjacent to the diagonal are 1 and all others are 0. F is
    # a matrix which is 1 on all diagonal elements except for in the first
    # and last columns. We denote (I - rho1 * D + rho1**2 * F) with A.
    # In the function calculating likelihood function,
    # XTAX, YTAY_diag, YTAX all mean multiplying the inverse covariance matrix
    # in between either the design matrix or the data.
    # As one can see, even though rho1 and sigma2 might update as we keep
    # fitting parameters, several terms stay unchanged and do not need to
    # be re-calculated.
    # For example, in X'AX = X'(I + rho1*D + rho1**2*F)X / sigma2,
    # the products X'X, X'DX, X'FX, etc. can always be re-used if they
    # are pre-calculated. Therefore, _D_gen and _F_gen constructs matrices
    # D and F, and _prepare_data_* calculates these products that can be
    # re-used. In principle, once parameters have been fitted for a
    # dataset, they can be updated for new incoming data by adding the
    # products X'X, X'DX, X'FX, X'Y etc. from new data to those from
    # existing data, and refit the parameters starting from the ones
    # fitted from existing data.
    def _D_gen(self, TR):
        if TR > 0:
            return np.diag(np.ones(TR - 1), -1) \
                + np.diag(np.ones(TR - 1), 1)
        else:
            return np.empty([0, 0])

    def _F_gen(self, TR):
        if TR > 0:
            F = np.eye(TR)
            F[0, 0] = 0
            F[TR - 1, TR - 1] = 0
            return F
        else:
            return np.empty([0, 0])

    def _prepare_DF(self, n_T, scan_onsets=None):
        """ Prepare the essential template matrices D and F for
            pre-calculating some terms to be re-used.
            The inverse covariance matrix of AR(1) noise is
            sigma^-2 * (I - rho1*D + rho1**2 * F).
            And we denote A = I - rho1*D + rho1**2 * F"""
        if scan_onsets is None:
            # assume that all data are acquired within the same scan.
            D = np.diag(np.ones(n_T - 1), -1) + np.diag(np.ones(n_T - 1), 1)
            F = np.eye(n_T)
            F[0, 0] = 0
            F[n_T - 1, n_T - 1] = 0
            n_run = 1
            run_TRs = np.array([n_T])
        else:
            # Each value in the scan_onsets tells the index at which
            # a new scan starts. For example, if n_T = 500, and
            # scan_onsets = [0,100,200,400], this means that the time points
            # of 0-99 are from the first scan, 100-199 are from the second,
            # 200-399 are from the third and 400-499 are from the fourth
            run_TRs = np.diff(np.append(scan_onsets, n_T))
            run_TRs = np.delete(run_TRs, np.where(run_TRs == 0))
            n_run = run_TRs.size
            # delete run length of 0 in case of duplication in scan_onsets.
            logger.info('I infer that the number of volumes'
                        ' in each scan are: {}'.format(run_TRs))

            D_ele = map(self._D_gen, run_TRs)
            F_ele = map(self._F_gen, run_TRs)
            D = []
            for d_ele in D_ele:
                D = scipy.linalg.block_diag(D, d_ele)
            F = []
            for f_ele in F_ele:
                F = scipy.linalg.block_diag(F, f_ele)
            # D and F above are templates for constructing
            # the inverse of temporal covariance matrix of noise
        return D, F, run_TRs, n_run

    def _prepare_data_XY(self, X, Y, D, F):
        """Prepares different forms of products of design matrix X
            and data Y, or between themselves.
            These products are re-used a lot during fitting.
            So we pre-calculate them. Because these are reused,
            it is in principle possible to update the fitting
            as new data come in, by just incrementally adding
            the products of new data and their corresponding parts
            of design matrix to these pre-calculated terms.
        """
        XTY, XTDY, XTFY = self._make_templates(D, F, X, Y)

        YTY_diag = np.sum(Y * Y, axis=0)
        YTDY_diag = np.sum(Y * np.dot(D, Y), axis=0)
        YTFY_diag = np.sum(Y * np.dot(F, Y), axis=0)

        XTX, XTDX, XTFX = self._make_templates(D, F, X, X)

        return XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX, \
            XTDX, XTFX

    def _prepare_data_XYX0(self, X, Y, X0, D, F, run_TRs, no_DC=False):
        """Prepares different forms of products between design matrix X or
            data Y or nuisance regressors X0 and X0.
            These products are re-used a lot during fitting.
            So we pre-calculate them.
            no_DC means not inserting regressors for DC components
            into nuisance regressor.
            It will only take effect if X0 is not None.
        """

        X_base = []
        for r_l in run_TRs:
            X_base = scipy.linalg.block_diag(X_base, np.ones(r_l)[:, None])
        res = np.linalg.lstsq(X_base, X)
        if np.any(np.isclose(res[1], 0)):
            raise ValueError('Your design matrix appears to have '
                             'included baseline time series.'
                             'Either remove them, or indicates which'
                             ' columns in your design matrix are for '
                             ' conditions of interest.')
        if X0 is not None:
            if not no_DC:
                res0 = np.linalg.lstsq(X_base, X0)
                if not np.any(np.isclose(res0[1], 0)):
                    # No columns in X0 can be explained by the
                    # baseline regressors. So we insert them.
                    X0 = np.concatenate((X_base, X0), axis=1)
                else:
                    logger.warning('Provided regressors for non-interesting '
                                   'time series already include baseline. '
                                   'No additional baseline is inserted.')
        else:
            # If a set of regressors for non-interested signals is not
            # provided, then we simply include one baseline for each run.
            X0 = X_base
            logger.info('You did not provide time series of no interest '
                        'such as DC component. One trivial regressor of'
                        ' DC component is included for further modeling.'
                        ' The final covariance matrix won''t '
                        'reflet them.')
        n_base = X0.shape[1]
        X0TX0, X0TDX0, X0TFX0 = self._make_templates(D, F, X0, X0)
        XTX0, XTDX0, XTFX0 = self._make_templates(D, F, X, X0)
        X0TY, X0TDY, X0TFY = self._make_templates(D, F, X0, Y)

        return X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, \
            X0TY, X0TDY, X0TFY, X0, n_base

    def _make_sandwidge(self, XTX, XTDX, XTFX, rho1):
        return XTX - rho1 * XTDX + rho1**2 * XTFX

    def _make_sandwidge_grad(self, XTDX, XTFX, rho1):
        return - XTDX + 2 * rho1 * XTFX

    def _make_templates(self, D, F, X, Y):
        XTY = np.dot(X.T, Y)
        XTDY = np.dot(np.dot(X.T, D), Y)
        XTFY = np.dot(np.dot(X.T, F), Y)
        return XTY, XTDY, XTFY

    def _calc_sandwidge(self, XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
                        XTX, XTDX, XTFX, X0TX0, X0TDX0, X0TFX0,
                        XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                        L, rho1, n_V, n_base):
        # Calculate the sandwidge terms which put A between X, Y and X0
        # These terms are used a lot in the likelihood. But in the _fitV
        # step, they only need to be calculated once, since A is fixed.
        # In _fitU step, they need to be calculated at each iteration,
        # because rho1 changes.
        XTAY = self._make_sandwidge(XTY, XTDY, XTFY, rho1)
        # dimension: feature*space
        YTAY = self._make_sandwidge(YTY_diag, YTDY_diag, YTFY_diag, rho1)
        # dimension: space,
        # A/sigma2 is the inverse of noise covariance matrix in each voxel.
        # YTAY means Y'AY
        XTAX = XTX[np.newaxis, :, :] - rho1[:, np.newaxis, np.newaxis] \
            * XTDX[np.newaxis, :, :] \
            + rho1[:, np.newaxis, np.newaxis]**2 * XTFX[np.newaxis, :, :]
        # dimension: space*feature*feature
        X0TAX0 = X0TX0[np.newaxis, :, :] - rho1[:, np.newaxis, np.newaxis] \
            * X0TDX0[np.newaxis, :, :] \
            + rho1[:, np.newaxis, np.newaxis]**2 * X0TFX0[np.newaxis, :, :]
        # dimension: space*#baseline*#baseline
        XTAX0 = XTX0[np.newaxis, :, :] - rho1[:, np.newaxis, np.newaxis] \
            * XTDX0[np.newaxis, :, :] \
            + rho1[:, np.newaxis, np.newaxis]**2 * XTFX0[np.newaxis, :, :]
        # dimension: space*feature*#baseline
        X0TAY = self._make_sandwidge(X0TY, X0TDY, X0TFY, rho1)
        # dimension: #baseline*space
        X0TAX0_i = np.linalg.solve(X0TAX0, np.identity(n_base)[None, :, :])
        # dimension: space*#baseline*#baseline
        XTAcorrX = XTAX
        # dimension: space*feature*feature
        XTAcorrY = XTAY
        # dimension: feature*space
        for i_v in range(n_V):
            XTAcorrX[i_v, :, :] -= \
                np.dot(np.dot(XTAX0[i_v, :, :], X0TAX0_i[i_v, :, :]),
                       XTAX0[i_v, :, :].T)
            XTAcorrY[:, i_v] -= np.dot(np.dot(XTAX0[i_v, :, :],
                                              X0TAX0_i[i_v, :, :]),
                                       X0TAY[:, i_v])
        XTAcorrXL = np.dot(XTAcorrX, L)
        # dimension: space*feature*rank
        LTXTAcorrXL = np.tensordot(XTAcorrXL, L, axes=(1, 0))
        # dimension: rank*feature*rank
        LTXTAcorrY = np.dot(L.T, XTAcorrY)
        # dimension: rank*space
        YTAcorrY = YTAY - np.sum(X0TAY * np.einsum('ijk,ki->ji',
                                                   X0TAX0_i, X0TAY), axis=0)
        # dimension: space

        return X0TAX0, XTAX0, X0TAY, X0TAX0_i, \
            XTAcorrX, XTAcorrY, YTAcorrY, LTXTAcorrY, XTAcorrXL, LTXTAcorrXL

    def _calc_LL(self, rho1, LTXTAcorrXL, LTXTAcorrY, YTAcorrY, X0TAX0, SNR2,
                 n_V, n_T, n_run, rank, n_base):
        # Calculate the log likelihood (excluding the GP prior of log(SNR))
        # for both _loglike_AR1_diagV_fitU and _loglike_AR1_diagV_fitV,
        # in addition to a few other terms.
        LAMBDA_i = LTXTAcorrXL * SNR2[:, None, None] + np.eye(rank)
        # dimension: space*rank*rank

        LAMBDA = np.linalg.solve(LAMBDA_i, np.identity(rank)[None, :, :])
        # dimension: space*rank*rank
        # LAMBDA is essentially the inverse covariance matrix of the
        # posterior probability of alpha, which bears the relation with
        # beta by beta = L * alpha. L is the Cholesky factor of the
        # shared covariance matrix U. Refer to the explanation below
        # Equation 5 in the NIPS paper.

        YTAcorrXL_LAMBDA = np.einsum('ji,ijk->ik', LTXTAcorrY, LAMBDA)
        # dimension: space*rank
        sigma2 = (YTAcorrY - np.sum(LTXTAcorrY * YTAcorrXL_LAMBDA.T, axis=0)
                  * SNR2) / (n_T - n_base)
        # dimension: space
        LL = - np.sum(np.log(sigma2)) * (n_T - n_base) * 0.5 \
            + np.sum(np.log(1 - rho1**2)) * n_run * 0.5 \
            - np.sum(np.log(np.linalg.det(X0TAX0))) * 0.5 \
            - np.sum(np.log(np.linalg.det(LAMBDA_i))) * 0.5 \
            - (n_T - n_base) * n_V * (1 + np.log(2 * np.pi)) * 0.5
        # Log likelihood
        return LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, sigma2

    def _calc_dist2_GP(self, coords=None, inten=None,
                       GP_space=False, GP_inten=False):
        # calculate the square of difference between each voxel's location
        # coorinates and image intensity.
        if GP_space:
            assert coords is not None, 'coordinate is not provided'
            # square of spatial distance between every two voxels
            dist2 = spdist.squareform(spdist.pdist(coords, 'sqeuclidean'))
            # set the hyperparameter for the GP process:
            if self.space_smooth_range is None:
                space_smooth_range = np.max(dist2)**0.5 / 2.0
                # By default, we assume the length scale should be
                # within half the size of ROI.
            else:
                space_smooth_range = self.space_smooth_range

            if GP_inten:
                assert inten is not None, 'intensity is not provided'
                # squre of difference between intensities of
                # # every two voxels
                inten_diff2 = spdist.squareform(
                    spdist.pdist(inten[:, None], 'sqeuclidean'))
                # set the hyperparameter for the GP process:
                if self.inten_smooth_range is None:
                    inten_smooth_range = np.max(inten_diff2)**0.5 / 2.0
                    # By default, we assume the length scale should be
                    # within half the maximum difference of intensity.
                else:
                    inten_smooth_range = self.inten_smooth_range
                n_smooth = 2
            else:
                inten_diff2 = None
                inten_smooth_range = None
                n_smooth = 1
        else:
            n_smooth = 0
            dist2 = None
            inten_diff2 = None
            GP_inten = False
            space_smooth_range = None
            inten_smooth_range = None
        return dist2, inten_diff2, space_smooth_range, inten_smooth_range,\
            n_smooth

    def _build_index_param(self, n_l, n_V, n_smooth):
        """ Build dictionaries to retrieve each parameter
            from the combined parameters.
        """
        idx_param_sing = {'Cholesky': np.arange(n_l), 'a1': n_l}
        # for simplified fitting
        idx_param_fitU = {'Cholesky': np.arange(n_l),
                          'a1': np.arange(n_l, n_l + n_V)}
        # for the likelihood function when we fit U (the shared covariance).
        idx_param_fitV = {'log_SNR2': np.arange(n_V - 1),
                          'c_space': n_V - 1, 'c_inten': n_V,
                          'c_both': np.arange(n_V - 1, n_V - 1 + n_smooth)}
        # for the likelihood function when we fit V (reflected by SNR of
        # each voxel)
        return idx_param_sing, idx_param_fitU, idx_param_fitV

    def _fit_RSA_UV(self, X, Y, X0,
                    scan_onsets=None, coords=None, inten=None):
        """ The major utility of fitting Bayesian RSA.
            Note that there is a naming change of variable. X in fit()
            is changed to Y here, and design in fit() is changed to X here.
            This is because we follow the tradition that X expresses the
            variable defined (controlled) by the experimenter, i.e., the
            time course of experimental conditions convolved by an HRF,
            and Y expresses data.
            However, in wrapper function fit(), we follow the naming
            routine of scikit-learn.
        """
        GP_inten = self.GP_inten
        GP_space = self.GP_space
        rank = self.rank
        n_V = np.size(Y, axis=1)
        n_T = np.size(Y, axis=0)
        n_C = np.size(X, axis=1)
        l_idx = np.tril_indices(n_C)

        np.random.seed(self.rand_seed)
        # setting random seed
        t_start = time.time()

        if rank is not None:
            # The rank of covariance matrix is specified
            idx_rank = np.where(l_idx[1] < rank)
            l_idx = (l_idx[0][idx_rank], l_idx[1][idx_rank])
            logger.info('Using the rank specified by the user: '
                        '{}'.format(rank))
        else:
            rank = n_C
            # if not specified, we assume you want to
            # estimate a full rank matrix
            logger.warning('Please be aware that you did not specify the'
                           ' rank of covariance matrix to estimate.'
                           'I will assume that the covariance matrix '
                           'shared among voxels is of full rank.'
                           'Rank = {}'.format(rank))
            logger.warning('Please be aware that estimating a matrix of '
                           'high rank can be very slow.'
                           'If you have a good reason to specify a rank '
                           'lower than the number of experiment conditions,'
                           ' do so.')

        n_l = np.size(l_idx[0])  # the number of parameters for L

        D, F, run_TRs, n_run = self._prepare_DF(
            n_T, scan_onsets=scan_onsets)
        XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX, \
            XTDX, XTFX = self._prepare_data_XY(X, Y, D, F)

        X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, \
            X0TY, X0TDY, X0TFY, X0, n_base = self._prepare_data_XYX0(
                X, Y, X0, D, F, run_TRs, no_DC=False)
        # Prepare the data for fitting. These pre-calculated matrices
        # will be re-used a lot in evaluating likelihood function and
        # gradient.
        # DC component will be added to the nuisance regressors.
        # In later steps, we do not need to add DC components again

        dist2, inten_diff2, space_smooth_range, inten_smooth_range,\
            n_smooth = self._calc_dist2_GP(
                coords=coords, inten=inten,
                GP_space=GP_space, GP_inten=GP_inten)
        # Calculating the distance between voxel locations and betweeen
        # voxel intensities. These are used if a Gaussian Process prior
        # is requested to regularize log(SNR^2)

        idx_param_sing, idx_param_fitU, idx_param_fitV = \
            self._build_index_param(n_l, n_V, n_smooth)
        # Indexes to find each parameter in a combined parameter vector.

        current_GP = np.zeros(n_smooth)

        # We will perform the fitting in 2~3 steps:
        # (1) A preliminary fitting assuming all voxels share
        # exactly the same temporal covariance matrix for their noise.
        # SNR is assumed to be 1 for all voxels in this fitting.
        # Therefore, there are only n_l+2 free parameters.
        # (2) (optional) A fitting which allows each voxel to have their
        # own pseudo-SNR and AR(1) coefficients. But no Gaussian Process
        # prior is imposed on log(SNR). This step is neglected if GP
        # prior is not requested. This step allows the SNR parameters to
        # move closer to their correct values before GP is introduced.
        # This step alternately fits the shared covariance and voxel-
        # specific variance. It fits for init_iter steps and the
        # tolerance is also increased by a factor of 5 to speed up
        # fitting.
        # (3) Final fitting. If GP prior is requested, it will be
        # introduced in this step. Otherwise, just fit as the previous
        # step, but using un-altered tolerance setting, and n_iter
        # as the number of iteration.

        # Step 1 fitting, with a simplified model
        current_vec_U_chlsk_l, current_a1, current_logSigma2 = \
            self._initial_fit_singpara(
                XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                X, Y, X0, idx_param_sing,
                l_idx, n_C, n_T, n_V, n_l, n_run, n_base, rank)

        current_logSNR2 = -current_logSigma2
        norm_factor = np.mean(current_logSNR2)
        current_logSNR2 = current_logSNR2 - norm_factor

        # Step 2 fitting, which only happens if
        # GP prior is requested
        if GP_space:
            current_vec_U_chlsk_l, current_a1, current_logSNR2, X0 \
                = self._fit_diagV_noGP(
                    XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
                    XTX, XTDX, XTFX, X, Y, X0, D, F, run_TRs,
                    current_vec_U_chlsk_l,
                    current_a1, current_logSNR2,
                    idx_param_fitU, idx_param_fitV,
                    l_idx, n_C, n_T, n_V, n_l, n_run, n_base, rank)

            current_GP[0] = np.log(np.min(
                dist2[np.tril_indices_from(dist2, k=-1)]))
            # We start fitting the model with GP prior with a small
            # length scale: the size of voxels.
            # Alternatively, initialize with a large distance.
            # Further testing of initial parameters need to be done.

#             current_GP[0] = np.log(np.max(dist2)/4.0)

            logger.debug('current GP[0]:{}'.format(current_GP[0]))
            if GP_inten:
                current_GP[1] = np.log(np.maximum(
                    np.percentile(inten_diff2[np.tril_indices_from(
                        inten_diff2, k=-1)], 5), 0.5))
                logger.debug(
                    'current GP[1]:{}'.format(current_GP[1]))
                # We start the length scale for intensity with
                # a small value. A heuristic is 5 percentile of
                # all the square differences. But it should not be
                # smaller than 0.5. This limit is set in case
                # many voxels have close to equal intensities,
                # which might render 5 percential to 0.

        # Step 3 fitting. GP prior is imposed if requested.
        # In this step, unless auto_nuisance is set to False, X0
        # will be re-estimated from the residuals after each step
        # of fitting.
        logger.debug('indexing:{}'.format(idx_param_fitV))
        logger.debug('initial GP parameters:{}'.format(current_GP))
        current_vec_U_chlsk_l, current_a1, current_logSNR2,\
            current_GP, X0 = self._fit_diagV_GP(
                XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
                XTX, XTDX, XTFX, X, Y, X0, D, F, run_TRs,
                current_vec_U_chlsk_l,
                current_a1, current_logSNR2, current_GP, n_smooth,
                idx_param_fitU, idx_param_fitV,
                l_idx, n_C, n_T, n_V, n_l, n_run, n_base, rank,
                GP_space, GP_inten, dist2, inten_diff2,
                space_smooth_range, inten_smooth_range)

        estU_chlsk_l_AR1_UV = np.zeros([n_C, rank])
        estU_chlsk_l_AR1_UV[l_idx] = current_vec_U_chlsk_l

        est_cov_AR1_UV = np.dot(estU_chlsk_l_AR1_UV, estU_chlsk_l_AR1_UV.T)

        est_rho1_AR1_UV = 2 / np.pi * np.arctan(current_a1)
        est_SNR_AR1_UV = np.exp(current_logSNR2 / 2.0)

        # Calculating est_sigma_AR1_UV, est_sigma_AR1_UV,
        # est_beta_AR1_UV and est_beta0_AR1_UV
        X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, \
            X0TY, X0TDY, X0TFY, X0, n_base = self._prepare_data_XYX0(
                X, Y, X0, D, F, run_TRs, no_DC=True)

        X0TAX0, XTAX0, X0TAY, X0TAX0_i, \
            XTAcorrX, XTAcorrY, YTAcorrY, LTXTAcorrY, XTAcorrXL, LTXTAcorrXL\
            = self._calc_sandwidge(XTY, XTDY, XTFY,
                                   YTY_diag, YTDY_diag, YTFY_diag,
                                   XTX, XTDX, XTFX, X0TX0, X0TDX0, X0TFX0,
                                   XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                   estU_chlsk_l_AR1_UV, est_rho1_AR1_UV,
                                   n_V, n_base)
        LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, sigma2 \
            = self._calc_LL(est_rho1_AR1_UV, LTXTAcorrXL, LTXTAcorrY, YTAcorrY,
                            X0TAX0, est_SNR_AR1_UV**2,
                            n_V, n_T, n_run, rank, n_base)
        est_sigma_AR1_UV = sigma2**0.5
        est_beta_AR1_UV = est_sigma_AR1_UV * est_SNR_AR1_UV**2 \
            * np.dot(estU_chlsk_l_AR1_UV, YTAcorrXL_LAMBDA.T)
        est_beta0_AR1_UV = np.einsum(
            'ijk,ki->ji', X0TAX0_i,
            (X0TAY - np.einsum('ikj,ki->ji', XTAX0, est_beta_AR1_UV)))

        t_finish = time.time()
        logger.info(
            'total time of fitting: {} seconds'.format(t_finish - t_start))
        logger.debug('final GP parameters:{}'.format(current_GP))
        if GP_space:
            est_space_smooth_r = np.exp(current_GP[0] / 2.0)
            if GP_inten:
                est_intensity_kernel_r = np.exp(current_GP[1] / 2.0)
                K_major = np.exp(- (dist2 / est_space_smooth_r**2 +
                                 inten_diff2 / est_intensity_kernel_r**2)
                                 / 2.0)
                K = K_major + np.diag(np.ones(n_V) * self.eta)
                est_std_log_SNR = (np.dot(current_logSNR2, np.dot(
                    np.linalg.inv(K), current_logSNR2)) / n_V / 4)**0.5
                # divided by 4 because we used
                # log(SNR^2) instead of log(SNR)
                return est_cov_AR1_UV, estU_chlsk_l_AR1_UV, est_SNR_AR1_UV, \
                    est_beta_AR1_UV, est_beta0_AR1_UV, est_sigma_AR1_UV, \
                    est_rho1_AR1_UV, est_space_smooth_r, \
                    est_std_log_SNR, est_intensity_kernel_r
            # When GP_inten is True, the following lines won't be reached
            else:
                K_major = np.exp(- dist2 / est_space_smooth_r**2 / 2.0)
                K = K_major + np.diag(np.ones(n_V) * self.eta)
                est_std_log_SNR = (np.dot(current_logSNR2, np.dot(
                    np.linalg.inv(K), current_logSNR2)) / n_V / 4)**0.5
                return est_cov_AR1_UV, estU_chlsk_l_AR1_UV, est_SNR_AR1_UV, \
                    est_beta_AR1_UV, est_beta0_AR1_UV, \
                    est_sigma_AR1_UV, est_rho1_AR1_UV, est_space_smooth_r, \
                    est_std_log_SNR
        else:
            return est_cov_AR1_UV, estU_chlsk_l_AR1_UV, est_SNR_AR1_UV, \
                est_beta_AR1_UV, est_beta0_AR1_UV, \
                est_sigma_AR1_UV, est_rho1_AR1_UV

    def _initial_fit_singpara(self, XTX, XTDX, XTFX,
                              YTY_diag, YTDY_diag, YTFY_diag,
                              XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                              XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                              X, Y, X0, idx_param_sing, l_idx,
                              n_C, n_T, n_V, n_l, n_run, n_base, rank):
        """ Perform initial fitting of a simplified model, which assumes
            that all voxels share exactly the same temporal covariance
            matrix for their noise (the same noise variance and
            auto-correlation). The SNR is implicitly assumed to be 1
            for all voxels.
        """
        logger.info('Initial fitting assuming single parameter of '
                    'noise for all voxels')
        X_joint = np.concatenate((X0, X), axis=1)
        beta_hat = np.linalg.lstsq(X_joint, Y)[0]
        residual = Y - np.dot(X_joint, beta_hat)
        # point estimates of betas and fitting residuals without assuming
        # the Bayesian model underlying RSA.

        # There are several possible ways of initializing the covariance.
        # (1) start from the point estimation of covariance

        # cov_point_est = np.cov(beta_hat)
        # current_vec_U_chlsk_l = \
        #     np.linalg.cholesky(cov_point_est + \
        #     np.eye(n_C) * 1e-6)[l_idx]

        # We add a tiny diagonal element to the point
        # estimation of covariance, just in case
        # the user provides data in which
        # n_V is smaller than n_C

        # (2) start from identity matrix

        # current_vec_U_chlsk_l = np.eye(n_C)[l_idx]

        # (3) random initialization

        current_vec_U_chlsk_l = np.random.randn(n_l)
        # vectorized version of L, Cholesky factor of U, the shared
        # covariance matrix of betas across voxels.

        rho1 = np.sum(
            residual[0:-1, :] * residual[1:, :], axis=0) / \
            np.sum(residual[0:-1, :] * residual[0:-1, :], axis=0)
        # Estimate of auto correlation assuming data includes pure noise.
        log_sigma2 = np.log(np.var(
            residual[1:, :] - residual[0:-1, :] * rho1, axis=0))
        # log of estimates of the variance of the "refreshing" noise
        # of AR(1) process at each time point.
        param0 = np.empty(np.sum(np.size(v)
                                 for v in idx_param_sing.values()))
        # Initial parameter
        # Then we fill each part of the original guess of parameters
        param0[idx_param_sing['Cholesky']] = current_vec_U_chlsk_l
        param0[idx_param_sing['a1']] = np.median(np.tan(rho1 * np.pi / 2))

        # Fit it.
        res = scipy.optimize.minimize(
            self._loglike_AR1_singpara, param0,
            args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                  XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                  XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                  l_idx, n_C, n_T, n_V, n_run, n_base,
                  idx_param_sing, rank),
            method=self.optimizer, jac=True, tol=self.tol,
            options={'disp': self.verbose, 'maxiter': 100})
        current_vec_U_chlsk_l = res.x[idx_param_sing['Cholesky']]
        current_a1 = res.x[idx_param_sing['a1']] * np.ones(n_V)
        # log(sigma^2) assuming the data include no signal is returned,
        # as a starting point for the iteration in the next step.
        # Although it should overestimate the variance,
        # setting it this way might allow it to track log(sigma^2)
        # more closely for each voxel.
        return current_vec_U_chlsk_l, current_a1, log_sigma2

    def _fit_diagV_noGP(
            self, XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
            XTX, XTDX, XTFX, X, Y, X0, D, F, run_TRs,
            current_vec_U_chlsk_l,
            current_a1, current_logSNR2,
            idx_param_fitU, idx_param_fitV,
            l_idx, n_C, n_T, n_V, n_l, n_run, n_base, rank):
        """ (optional) second step of fitting, full model but without
            GP prior on log(SNR). This step is only done if GP prior
            is requested.
        """
        init_iter = self.init_iter
        logger.info('second fitting without GP prior'
                    ' for {} times'.format(init_iter))

        # Initial parameters
        param0_fitU = np.empty(
            np.sum(np.size(v) for v in idx_param_fitU.values()))
        param0_fitV = np.empty(np.size(idx_param_fitV['log_SNR2']))
        # We cannot use the same logic as the line above because
        # idx_param_fitV also includes entries for GP parameters.
        param0_fitU[idx_param_fitU['Cholesky']] = \
            current_vec_U_chlsk_l.copy()
        param0_fitU[idx_param_fitU['a1']] = current_a1.copy()
        param0_fitV[idx_param_fitV['log_SNR2']] = \
            current_logSNR2[:-1].copy()

        L = np.zeros((n_C, rank))
        tol = self.tol * 5
        for it in range(0, init_iter):
            X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, \
                X0TY, X0TDY, X0TFY, X0, n_base = self._prepare_data_XYX0(
                    X, Y, X0, D, F, run_TRs, no_DC=True)

            # fit U, the covariance matrix, together with AR(1) param
            param0_fitU[idx_param_fitU['Cholesky']] = \
                current_vec_U_chlsk_l
            param0_fitU[idx_param_fitU['a1']] = current_a1
            res_fitU = scipy.optimize.minimize(
                self._loglike_AR1_diagV_fitU, param0_fitU,
                args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                      XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                      XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                      current_logSNR2, l_idx, n_C,
                      n_T, n_V, n_run, n_base, idx_param_fitU, rank),
                method=self.optimizer, jac=True, tol=tol,
                options={'xtol': tol, 'disp': self.verbose,
                         'maxiter': 4})
            current_vec_U_chlsk_l = \
                res_fitU.x[idx_param_fitU['Cholesky']]
            current_a1 = res_fitU.x[idx_param_fitU['a1']]
            norm_fitUchange = np.linalg.norm(res_fitU.x - param0_fitU)
            logger.debug('norm of parameter change after fitting U: '
                         '{}'.format(norm_fitUchange))
            param0_fitU = res_fitU.x.copy()

            # fit V, reflected in the log(SNR^2) of each voxel
            rho1 = np.arctan(current_a1) * 2 / np.pi
            L[l_idx] = current_vec_U_chlsk_l
            X0TAX0, XTAX0, X0TAY, X0TAX0_i, \
                XTAcorrX, XTAcorrY, YTAcorrY, \
                LTXTAcorrY, XTAcorrXL, LTXTAcorrXL = \
                self._calc_sandwidge(XTY, XTDY, XTFY,
                                     YTY_diag, YTDY_diag, YTFY_diag,
                                     XTX, XTDX, XTFX,
                                     X0TX0, X0TDX0, X0TFX0,
                                     XTX0, XTDX0, XTFX0,
                                     X0TY, X0TDY, X0TFY,
                                     L, rho1, n_V, n_base)
            res_fitV = scipy.optimize.minimize(
                self._loglike_AR1_diagV_fitV, param0_fitV,
                args=(X0TAX0, XTAX0, X0TAY,
                      X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY,
                      LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                      current_vec_U_chlsk_l,
                      current_a1, l_idx, n_C, n_T, n_V, n_run,
                      n_base, idx_param_fitV, rank,
                      False, False),
                method=self.optimizer, jac=True, tol=tol,
                options={'xtol': tol, 'disp': self.verbose,
                         'maxiter': 4})

            current_logSNR2[0:n_V - 1] = res_fitV.x
            current_logSNR2[-1] = - np.sum(current_logSNR2[0:n_V - 1])

            norm_fitVchange = np.linalg.norm(res_fitV.x - param0_fitV)
            logger.debug('norm of parameter change after fitting V: '
                         '{}'.format(norm_fitVchange))
            logger.debug('E[log(SNR2)^2]: {}'.format(
                    np.mean(current_logSNR2**2)))

            # The lines below are for debugging purpose.
            # If any voxel's log(SNR^2) gets to non-finite number,
            # something might be wrong -- could be that the data has
            # nothing to do with the design matrix.
            if np.any(np.logical_not(np.isfinite(current_logSNR2))):
                logger.warning('Initial fitting: iteration {}'.format(it))
                logger.warning('current log(SNR^2): '
                               '{}'.format(current_logSNR2))
                logger.warning('log(sigma^2) has non-finite number')

            param0_fitV = res_fitV.x.copy()

            # Re-estimating X0 from residuals
            current_SNR2 = np.exp(current_logSNR2)
            if self.auto_nuisance:
                LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, current_sigma2 \
                    = self._calc_LL(rho1, LTXTAcorrXL, LTXTAcorrY, YTAcorrY,
                                    X0TAX0, current_SNR2,
                                    n_V, n_T, n_run, rank, n_base)
                betas = current_sigma2**0.5 * current_SNR2 \
                    * np.dot(L, YTAcorrXL_LAMBDA.T)
                residuals = Y - np.dot(X, betas)
                u, s, v = np.linalg.svd(residuals)
                X0 = u[:, :self.n_nureg]

            if norm_fitVchange / np.sqrt(param0_fitV.size) < tol \
                    and norm_fitUchange / np.sqrt(param0_fitU.size) \
                    < tol:
                break
        return current_vec_U_chlsk_l, current_a1, current_logSNR2, X0

    def _fit_diagV_GP(
            self, XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
            XTX, XTDX, XTFX, X, Y, X0, D, F, run_TRs,
            current_vec_U_chlsk_l,
            current_a1, current_logSNR2, current_GP, n_smooth,
            idx_param_fitU, idx_param_fitV, l_idx,
            n_C, n_T, n_V, n_l, n_run, n_base, rank, GP_space, GP_inten,
            dist2, inten_diff2, space_smooth_range, inten_smooth_range):
        """ Last step of fitting. If GP is not requested, this step will
            still be done, just without GP prior on log(SNR).
        """
        tol = self.tol
        n_iter = self.n_iter
        logger.info('Last step of fitting.'
                    ' for maximum {} times'.format(n_iter))

        # Initial parameters
        param0_fitU = np.empty(
            np.sum(np.size(v) for v in idx_param_fitU.values()))
        param0_fitV = np.empty(np.size(idx_param_fitV['log_SNR2'])
                               + np.size(idx_param_fitV['c_both']))
        # We cannot use the same logic as the line above because
        # idx_param_fitV also includes entries for GP parameters.
        param0_fitU[idx_param_fitU['Cholesky']] = \
            current_vec_U_chlsk_l.copy()
        param0_fitU[idx_param_fitU['a1']] = current_a1.copy()
        param0_fitV[idx_param_fitV['log_SNR2']] = \
            current_logSNR2[:-1].copy()
        L = np.zeros((n_C, rank))
        L[l_idx] = current_vec_U_chlsk_l
        if self.GP_space:
            param0_fitV[idx_param_fitV['c_both']] = current_GP.copy()

        for it in range(0, n_iter):
            X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, \
                X0TY, X0TDY, X0TFY, X0, n_base = self._prepare_data_XYX0(
                    X, Y, X0, D, F, run_TRs, no_DC=True)

            # fit U

            param0_fitU[idx_param_fitU['Cholesky']] = \
                current_vec_U_chlsk_l
            param0_fitU[idx_param_fitU['a1']] = current_a1

            res_fitU = scipy.optimize.minimize(
                self._loglike_AR1_diagV_fitU, param0_fitU,
                args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                      XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                      XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                      current_logSNR2, l_idx, n_C, n_T, n_V,
                      n_run, n_base, idx_param_fitU, rank),
                method=self.optimizer, jac=True,
                tol=tol,
                options={'xtol': tol,
                         'disp': self.verbose, 'maxiter': 6})
            current_vec_U_chlsk_l = \
                res_fitU.x[idx_param_fitU['Cholesky']]
            current_a1 = res_fitU.x[idx_param_fitU['a1']]
            L[l_idx] = current_vec_U_chlsk_l
            fitUchange = res_fitU.x - param0_fitU
            norm_fitUchange = np.linalg.norm(fitUchange)
            logger.debug('norm of parameter change after fitting U: '
                         '{}'.format(norm_fitUchange))
            param0_fitU = res_fitU.x.copy()

            # fit V
            rho1 = np.arctan(current_a1) * 2 / np.pi
            X0TAX0, XTAX0, X0TAY, X0TAX0_i, \
                XTAcorrX, XTAcorrY, YTAcorrY, \
                LTXTAcorrY, XTAcorrXL, LTXTAcorrXL = \
                self._calc_sandwidge(XTY, XTDY, XTFY,
                                     YTY_diag, YTDY_diag, YTFY_diag,
                                     XTX, XTDX, XTFX,
                                     X0TX0, X0TDX0, X0TFX0,
                                     XTX0, XTDX0, XTFX0,
                                     X0TY, X0TDY, X0TFY,
                                     L, rho1, n_V, n_base)
            res_fitV = scipy.optimize.minimize(
                self._loglike_AR1_diagV_fitV, param0_fitV, args=(
                    X0TAX0, XTAX0, X0TAY, X0TAX0_i,
                    XTAcorrX, XTAcorrY, YTAcorrY,
                    LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                    current_vec_U_chlsk_l, current_a1,
                    l_idx, n_C, n_T, n_V, n_run, n_base,
                    idx_param_fitV, rank,
                    GP_space, GP_inten, dist2, inten_diff2,
                    space_smooth_range, inten_smooth_range),
                method=self.optimizer, jac=True,
                tol=tol,
                options={'xtol': tol,
                         'disp': self.verbose, 'maxiter': 6})

            current_logSNR2[0:n_V - 1] = \
                res_fitV.x[idx_param_fitV['log_SNR2']]
            current_logSNR2[n_V - 1] = -np.sum(current_logSNR2[0:n_V - 1])
            current_GP = res_fitV.x[idx_param_fitV['c_both']]

            fitVchange = res_fitV.x - param0_fitV
            norm_fitVchange = np.linalg.norm(fitVchange)

            param0_fitV = res_fitV.x.copy()
            logger.debug('norm of parameter change after fitting V: '
                         '{}'.format(norm_fitVchange))
            logger.debug('E[log(SNR2)^2]: {}'.format(
                np.mean(current_logSNR2**2)))

            # Re-estimating X0 from residuals
            current_SNR2 = np.exp(current_logSNR2)
            if self.auto_nuisance:
                LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, current_sigma2 \
                    = self._calc_LL(rho1, LTXTAcorrXL, LTXTAcorrY, YTAcorrY,
                                    X0TAX0, current_SNR2,
                                    n_V, n_T, n_run, rank, n_base)
                betas = current_sigma2**0.5 * current_SNR2 \
                    * np.dot(L, YTAcorrXL_LAMBDA.T)
                residuals = Y - np.dot(X, betas)
                u, s, v = np.linalg.svd(residuals)
                X0 = u[:, :self.n_nureg]

            if GP_space:
                logger.debug('current GP[0]: {}'.format(current_GP[0]))
                logger.debug('gradient for GP[0]: {}'.format(
                    res_fitV.jac[idx_param_fitV['c_space']]))
                if GP_inten:
                    logger.debug('current GP[1]: {}'.format(current_GP[1]))
                    logger.debug('gradient for GP[1]: {}'.format(
                        res_fitV.jac[idx_param_fitV['c_inten']]))
            if np.max(np.abs(fitVchange)) < tol and \
                    np.max(np.abs(fitUchange)) < tol:
                break

        return current_vec_U_chlsk_l, current_a1, current_logSNR2,\
            current_GP, X0

    # We fit two parts of the parameters iteratively.
    # The following are the corresponding negative log likelihood functions.

    def _loglike_AR1_diagV_fitU(self, param, XTX, XTDX, XTFX, YTY_diag,
                                YTDY_diag, YTFY_diag, XTY, XTDY, XTFY,
                                X0TX0, X0TDX0, X0TFX0,
                                XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                log_SNR2, l_idx, n_C, n_T, n_V, n_run, n_base,
                                idx_param_fitU, rank):
        # This function calculates the log likelihood of data given cholesky
        # decomposition of U and AR(1) parameters of noise as free parameters.
        # Free parameters are in param.
        # The log of the square of signal to noise level in each voxel
        # (the ratio of the diagonal elements in V and
        # the noise variance) are fixed. This likelihood is iteratively
        # optimized with the one with suffix _fitV.
        #
        # The meaing of U and V follow this wiki page of matrix normal
        # distribution:
        # https://en.wikipedia.org/wiki/Matrix_normal_distribution
        #
        # We assume betas of all voxels as a matrix follow this distribution.
        # U describe the covariance between conditions. V describe the
        # covariance between voxels.
        #
        # In this version, we assume that beta is independent between voxels
        # and noise is also independent.
        # By the assumption that noise is independent, we only need to pass
        # the products X'X, X'Y and Y'Y, instead of X and Y
        # Y'Y is passed in the form of its diagonal elements.
        # DiagV means we assume that the variance of beta can be different
        # between voxels. This means that V is a diagonal matrix instead of
        # an identity matrix. The parameter includes the lower triangular
        # part of the cholesky decomposition
        # of U (flattened), then tan(rho1*pi/2) where rho1 is
        # each voxel's autoregressive coefficient (assumging AR(1) model).
        # Such parametrization avoids the need of boundaries
        # for parameters.

        L = np.zeros([n_C, rank])
        # lower triagular matrix L, cholesky decomposition of U
        L[l_idx] = param[idx_param_fitU['Cholesky']]

        a1 = param[idx_param_fitU['a1']]
        rho1 = 2.0 / np.pi * np.arctan(a1)  # auto-regressive coefficients

        SNR2 = np.exp(log_SNR2)
        # each element of SNR2 is the ratio of the diagonal element on V
        # to the variance of the fresh noise in that voxel

        X0TAX0, XTAX0, X0TAY, X0TAX0_i, \
            XTAcorrX, XTAcorrY, YTAcorrY, \
            LTXTAcorrY, XTAcorrXL, LTXTAcorrXL = \
            self._calc_sandwidge(XTY, XTDY, XTFY,
                                 YTY_diag, YTDY_diag, YTFY_diag,
                                 XTX, XTDX, XTFX, X0TX0, X0TDX0, X0TFX0,
                                 XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                 L, rho1, n_V, n_base)

        # Only starting from this point, SNR2 is involved

        LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, sigma2 \
            = self._calc_LL(rho1, LTXTAcorrXL, LTXTAcorrY, YTAcorrY,
                            X0TAX0, SNR2, n_V, n_T, n_run, rank, n_base)
        if not np.isfinite(LL):
            logger.warning('NaN detected!')
            logger.warning('LL: {}'.format(LL))
            logger.warning('sigma2: {}'.format(sigma2))
            logger.warning('YTAcorrY: {}'.format(YTAcorrY))
            logger.warning('LTXTAcorrY: {}'.format(LTXTAcorrY))
            logger.warning('YTAcorrXL_LAMBDA: {}'.format(YTAcorrXL_LAMBDA))
            logger.warning('SNR2: {}'.format(SNR2))

        YTAcorrXL_LAMBDA_LT = np.dot(YTAcorrXL_LAMBDA, L.T)
        # dimension: space*feature (feature can be larger than rank)
        deriv_L = -np.einsum('ijk,ikl,i', XTAcorrXL, LAMBDA, SNR2) \
            - np.dot(np.einsum('ijk,ik->ji', XTAcorrXL, YTAcorrXL_LAMBDA)
                     * SNR2**2 / sigma2, YTAcorrXL_LAMBDA) \
            + np.dot(XTAcorrY / sigma2 * SNR2, YTAcorrXL_LAMBDA)
        # dimension: feature*rank

        # The following are for calculating the derivative to a1
        deriv_a1 = np.empty(n_V)
        dXTAX_drho1 = -XTDX + 2 * rho1[:, np.newaxis, np.newaxis] * XTFX
        # dimension: space*feature*feature
        dXTAY_drho1 = self._make_sandwidge_grad(XTDY, XTFY, rho1)
        # dimension: feature*space
        dYTAY_drho1 = self._make_sandwidge_grad(YTDY_diag, YTFY_diag, rho1)
        # dimension: space,

        dX0TAX0_drho1 = - X0TDX0 \
            + 2 * rho1[:, np.newaxis, np.newaxis] * X0TFX0
        # dimension: space*rank*rank
        dXTAX0_drho1 = - XTDX0 \
            + 2 * rho1[:, np.newaxis, np.newaxis] * XTFX0
        # dimension: space*feature*rank
        dX0TAY_drho1 = self._make_sandwidge_grad(X0TDY, X0TFY, rho1)
        # dimension: rank*space

        # The following are executed for each voxel.
        for i_v in range(n_V):
            # All variables with _ele as suffix are for data of just one voxel
            invX0TAX0_X0TAX_ele = np.dot(X0TAX0_i[i_v, :, :],
                                         XTAX0[i_v, :, :].T)
            invX0TAX0_X0TAY_ele = np.dot(X0TAX0_i[i_v, :, :], X0TAY[:, i_v])
            dXTAX0_drho1_invX0TAX0_X0TAX_ele = np.dot(dXTAX0_drho1[i_v, :, :],
                                                      invX0TAX0_X0TAX_ele)
            # preparation for the variable below
            dXTAcorrX_drho1_ele = dXTAX_drho1[i_v, :, :] \
                - dXTAX0_drho1_invX0TAX0_X0TAX_ele \
                - dXTAX0_drho1_invX0TAX0_X0TAX_ele.T \
                + np.dot(np.dot(invX0TAX0_X0TAX_ele.T,
                                dX0TAX0_drho1[i_v, :, :]),
                         invX0TAX0_X0TAX_ele)
            dXTAcorrY_drho1_ele = dXTAY_drho1[:, i_v] \
                - np.dot(invX0TAX0_X0TAX_ele.T, dX0TAY_drho1[:, i_v]) \
                - np.dot(dXTAX0_drho1[i_v, :, :], invX0TAX0_X0TAY_ele) \
                + np.dot(np.dot(invX0TAX0_X0TAX_ele.T,
                                dX0TAX0_drho1[i_v, :, :]),
                         invX0TAX0_X0TAY_ele)
            dYTAcorrY_drho1_ele = dYTAY_drho1[i_v] \
                - np.dot(dX0TAY_drho1[:, i_v], invX0TAX0_X0TAY_ele) * 2\
                + np.dot(np.dot(invX0TAX0_X0TAY_ele, dX0TAX0_drho1[i_v, :, :]),
                         invX0TAX0_X0TAY_ele)
            deriv_a1[i_v] = 2 / np.pi / (1 + a1[i_v]**2) \
                * (- n_run * rho1[i_v] / (1 - rho1[i_v]**2)
                   - np.einsum('ij,ij', X0TAX0_i[i_v, :, :],
                               dX0TAX0_drho1[i_v, :, :]) * 0.5
                   - np.einsum('ij,ij', LAMBDA[i_v, :, :],
                               np.dot(np.dot(
                                L.T, dXTAcorrX_drho1_ele), L))
                   * (SNR2[i_v] * 0.5)
                   - dYTAcorrY_drho1_ele * 0.5 / sigma2[i_v]
                   + SNR2[i_v] / sigma2[i_v]
                   * np.dot(dXTAcorrY_drho1_ele,
                            YTAcorrXL_LAMBDA_LT[i_v, :])
                   - (0.5 * SNR2[i_v]**2 / sigma2[i_v])
                   * np.dot(np.dot(YTAcorrXL_LAMBDA_LT[i_v, :],
                                   dXTAcorrX_drho1_ele),
                            YTAcorrXL_LAMBDA_LT[i_v, :]))

        deriv = np.empty(np.size(param))
        deriv[idx_param_fitU['Cholesky']] = deriv_L[l_idx]
        deriv[idx_param_fitU['a1']] = deriv_a1

        return -LL, -deriv

    def _loglike_AR1_diagV_fitV(self, param,
                                X0TAX0, XTAX0, X0TAY, X0TAX0_i,
                                XTAcorrX, XTAcorrY, YTAcorrY,
                                LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                                L_l, a1, l_idx, n_C, n_T, n_V, n_run,
                                n_base, idx_param_fitV, rank=None,
                                GP_space=False, GP_inten=False,
                                dist2=None, inten_dist2=None,
                                space_smooth_range=None,
                                inten_smooth_range=None):

        # This function calculates the log likelihood of data given
        # the log of the square of pseudo signal to noise ratio in each voxel.
        # The free parameter log(SNR^2) is in param
        # This likelihood is iteratively optimized with the one with _fitU.
        # The cholesky factor of U and autoregressive coefficient
        # in temporal AR(1) model for noise are fixed.
        # Because the ML estimate of the variance of noise in each voxel
        # (sigma^2) given other parameters has analytic form,
        # we do not need to explicitly parametrize it.
        # Just set it to the ML value.
        #
        # L_l is the lower triangular part of L, a1 is tan(rho1*pi/2),
        # where rho1 is the autoregressive coefficient in each voxel
        # We can optionally include Gaussion Process prior to log(SNR).
        # This term is not included in _fitU, because log(SNR)
        # are fixed in _fitU.
        # GP_space and GP_inten are Boolean, indicating whether we want to
        # include GP kernels either on voxel coordinates or intensity.
        # dist2 and inten_dist2 are the squares of spatial distances and
        # intensity differences ([n_voxel x n_voxel]. space_smooth_range
        # and inten_smooth_range are the range we believe the GP length
        # scale should reside in. They are used in additional half-cauchy
        # prior to constraint these length scales.

        n_l = np.size(l_idx[0])
        # the number of parameters in the index of lower-triangular matrix
        if rank is None:
            rank = int((2 * n_C + 1 -
                        np.sqrt(n_C**2 * 4 + n_C * 4 + 1 - 8 * n_l)) / 2)
        L = np.zeros([n_C, rank])
        L[l_idx] = L_l

        log_SNR2 = np.empty(n_V)
        log_SNR2[0:n_V - 1] = param[idx_param_fitV['log_SNR2']]
        log_SNR2[-1] = -np.sum(log_SNR2[0:n_V - 1])
        # This is following the restriction that SNR's have geometric mean
        # of 1. That is why they are called pseudo-SNR. This restriction
        # is imposed because SNR and L are determined only up to a scale
        # Be cautious that during simulation, when there is absolute
        # no signal in the data, sometimes the fitting diverges,
        # presumably because we have created correlation between logS_NR2
        # due to the constraint. But I have not reproduced this often.
        SNR2 = np.exp(log_SNR2)
        # If requested, a GP prior is imposed on log(SNR).
        rho1 = 2.0 / np.pi * np.arctan(a1)
        # AR(1) coefficient, dimension: space
        LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, sigma2 \
            = self._calc_LL(rho1, LTXTAcorrXL, LTXTAcorrY, YTAcorrY, X0TAX0,
                            SNR2, n_V, n_T, n_run, rank, n_base)
        # Log likelihood of data given parameters, without the GP prior.
        deriv_log_SNR2 = (-rank + np.trace(LAMBDA, axis1=1, axis2=2)) * 0.5\
            + YTAcorrY / (sigma2 * 2.0) - (n_T - n_base) * 0.5 \
            - np.einsum('ij,ijk,ik->i', YTAcorrXL_LAMBDA,
                        LTXTAcorrXL, YTAcorrXL_LAMBDA)\
            / (sigma2 * 2.0) * (SNR2**2)
        # Partial derivative of log likelihood over log(SNR^2)
        # dimension: space,
        # The second term above is due to the equation for calculating
        # sigma2
        if GP_space:
            # Imposing GP prior on log(SNR) at least over
            # spatial coordinates
            c_space = param[idx_param_fitV['c_space']]
            l2_space = np.exp(c_space)
            # The square of the length scale of the GP kernel defined on
            # the spatial coordinates of voxels
            dl2_dc_space = l2_space
            # partial derivative of l^2 over b

            if GP_inten:
                c_inten = param[idx_param_fitV['c_inten']]
                l2_inten = np.exp(c_inten)
                # The square of the length scale of the GP kernel defined
                # on the image intensity of voxels
                dl2_dc_inten = l2_inten
                # partial derivative of l^2 over b
                K_major = np.exp(- (dist2 / l2_space
                                    + inten_dist2 / l2_inten)
                                 / 2.0)
            else:
                K_major = np.exp(- dist2 / l2_space / 2.0)
                # The kernel defined over the spatial coordinates of voxels.
                # This is a template: the diagonal values are all 1, meaning
                # the variance of log(SNR) has not been multiplied
            K_tilde = K_major + np.diag(np.ones(n_V) * self.eta)
            # We add a small number to the diagonal to make sure the matrix
            # is invertible.
            # Note that the K_tilder here is still template:
            # It is the correct K divided by the variance tau^2
            # So it does not depend on the variance of the GP.
            L_K_tilde = np.linalg.cholesky(K_tilde)
            inv_L_K_tilde = np.linalg.solve(L_K_tilde, np.identity(n_V))
            inv_K_tilde = np.dot(inv_L_K_tilde.T, inv_L_K_tilde)
            log_det_K_tilde = np.sum(np.log(np.diag(L_K_tilde)**2))

            invK_tilde_log_SNR = np.dot(inv_K_tilde, log_SNR2) / 2
            log_SNR_invK_tilde_log_SNR = np.dot(log_SNR2,
                                                invK_tilde_log_SNR) / 2

            # ML estimate of the variance of the Gaussian Process given
            # other parameters.
            tau2 = (log_SNR_invK_tilde_log_SNR - n_V * self.tau_range**2
                    + np.sqrt(n_V**2 * self.tau_range**4 + (2 * n_V + 8)
                              * self.tau_range**2
                              * log_SNR_invK_tilde_log_SNR
                              + log_SNR_invK_tilde_log_SNR**2))\
                / 2 / (n_V + 2)
            # Note that this derivation is based on the assumption that
            # half-Cauchy prior on the standard deviation of the GP is
            # imposed. If a different prior is imposed, this term needs
            # to be changed accordingly.

            # GP prior terms added to the log likelihood
            LL = LL - log_det_K_tilde / 2.0 - n_V / 2.0 * np.log(tau2) \
                - np.log(2 * np.pi) * n_V / 2.0 \
                - log_SNR_invK_tilde_log_SNR / tau2 / 2

            deriv_log_SNR2 -= invK_tilde_log_SNR / tau2 / 2.0
            # Note that the derivative to log(SNR) is
            # invK_tilde_log_SNR / tau2, but we are calculating the
            # derivative to log(SNR^2)

            dK_tilde_dl2_space = dist2 * (K_major) / 2.0 \
                / l2_space**2

            deriv_c_space = \
                (np.dot(np.dot(invK_tilde_log_SNR, dK_tilde_dl2_space),
                        invK_tilde_log_SNR) / tau2 / 2.0
                 - np.sum(inv_K_tilde * dK_tilde_dl2_space) / 2.0)\
                * dl2_dc_space

            # Prior on the length scales
            LL += scipy.stats.halfcauchy.logpdf(
                l2_space**0.5, scale=space_smooth_range)
            deriv_c_space -= 1 / (l2_space + space_smooth_range**2)\
                * dl2_dc_space

            # Prior on the standar deviation of GP
            LL += scipy.stats.halfcauchy.logpdf(
                tau2**0.5, scale=self.tau_range)
            # Note that the form of the maximum likelihood estimate
            # of tau2 depends on the form of prior imposed.

            if GP_inten:
                dK_tilde_dl2_inten = inten_dist2 * K_major \
                    / 2.0 / l2_inten**2
                deriv_c_inten = \
                    (np.dot(np.dot(invK_tilde_log_SNR, dK_tilde_dl2_inten),
                            invK_tilde_log_SNR) / tau2 / 2.0
                     - np.sum(inv_K_tilde * dK_tilde_dl2_inten) / 2.0)\
                    * dl2_dc_inten
                # Prior on the length scale
                LL += scipy.stats.halfcauchy.logpdf(
                    l2_inten**0.5, scale=inten_smooth_range)
                deriv_c_inten -= 1 / (l2_inten + inten_smooth_range**2)\
                    * dl2_dc_inten
        else:
            LL += np.sum(scipy.stats.norm.logpdf(log_SNR2 / 2.0,
                                                 scale=self.tau_range))
            # If GP prior is not requested, we still want to regularize on
            # the magnitude of log(SNR).
            deriv_log_SNR2 += - log_SNR2 / self.tau_range**2 / 4.0

        deriv = np.empty(np.size(param))
        deriv[idx_param_fitV['log_SNR2']] = \
            deriv_log_SNR2[0:n_V - 1] - deriv_log_SNR2[n_V - 1]
        if GP_space:
            deriv[idx_param_fitV['c_space']] = deriv_c_space
            if GP_inten:
                deriv[idx_param_fitV['c_inten']] = deriv_c_inten

        return -LL, -deriv

    def _loglike_AR1_singpara(self, param, XTX, XTDX, XTFX, YTY_diag,
                              YTDY_diag, YTFY_diag, XTY, XTDY, XTFY,
                              X0TX0, X0TDX0, X0TFX0,
                              XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                              l_idx, n_C, n_T, n_V, n_run, n_base,
                              idx_param_sing, rank=None):
        # In this version, we assume that beta is independent
        # between voxels and noise is also independent.
        # singpara version uses single parameter of sigma^2 and rho1
        # to all voxels. This serves as the initial fitting to get
        # an estimate of L and sigma^2 and rho1. The SNR is inherently
        # assumed to be 1.

        n_l = np.size(l_idx[0])
        # the number of parameters in the index of lower-triangular matrix

        if rank is None:
            rank = int((2 * n_C + 1
                        - np.sqrt(n_C**2 * 4 + n_C * 4 + 1 - 8 * n_l)) / 2)

        L = np.zeros([n_C, rank])
        L[l_idx] = param[idx_param_sing['Cholesky']]

        a1 = param[idx_param_sing['a1']]
        rho1 = 2.0 / np.pi * np.arctan(a1)

        XTAX = XTX - rho1 * XTDX + rho1**2 * XTFX
        X0TAX0 = X0TX0 - rho1 * X0TDX0 + rho1**2 * X0TFX0
        XTAX0 = XTX0 - rho1 * XTDX0 + rho1**2 * XTFX0
        XTAcorrX = XTAX - np.dot(XTAX0, np.linalg.solve(X0TAX0, XTAX0.T))
        XTAcorrXL = np.dot(XTAcorrX, L)
        LAMBDA_i = np.dot(np.dot(L.T, XTAcorrX), L) + np.eye(rank)

        XTAY = XTY - rho1 * XTDY + rho1**2 * XTFY
        X0TAY = X0TY - rho1 * X0TDY + rho1**2 * X0TFY
        XTAcorrY = XTAY - np.dot(XTAX0, np.linalg.solve(X0TAX0, X0TAY))
        LTXTAcorrY = np.dot(L.T, XTAcorrY)

        YTAY = YTY_diag - rho1 * YTDY_diag + rho1**2 * YTFY_diag
        YTAcorrY = YTAY \
            - np.sum(X0TAY * np.linalg.solve(X0TAX0, X0TAY), axis=0)

        LAMBDA_LTXTAcorrY = np.linalg.solve(LAMBDA_i, LTXTAcorrY)
        L_LAMBDA_LTXTAcorrY = np.dot(L, LAMBDA_LTXTAcorrY)

        sigma2 = np.mean(YTAcorrY -
                         np.sum(LTXTAcorrY * LAMBDA_LTXTAcorrY, axis=0))\
            / (n_T - n_base)
        LL = n_V * (-np.log(sigma2) * (n_T - n_base) * 0.5
                    + np.log(1 - rho1**2) * n_run * 0.5
                    - np.log(np.linalg.det(X0TAX0)) * 0.5
                    - np.log(np.linalg.det(LAMBDA_i)) * 0.5)

        deriv_L = np.dot(XTAcorrY, LAMBDA_LTXTAcorrY.T) / sigma2 \
            - np.dot(np.dot(XTAcorrXL, LAMBDA_LTXTAcorrY),
                     LAMBDA_LTXTAcorrY.T) / sigma2 \
            - np.linalg.solve(LAMBDA_i, XTAcorrXL.T).T * n_V

        # These terms are used to construct derivative to a1.
        dXTAX_drho1 = - XTDX + 2 * rho1 * XTFX
        dX0TAX0_drho1 = - X0TDX0 + 2 * rho1 * X0TFX0
        dXTAX0_drho1 = - XTDX0 + 2 * rho1 * XTFX0
        invX0TAX0_X0TAX = np.linalg.solve(X0TAX0, XTAX0.T)
        dXTAX0_drho1_invX0TAX0_X0TAX = np.dot(dXTAX0_drho1, invX0TAX0_X0TAX)

        dXTAcorrX_drho1 = dXTAX_drho1 - dXTAX0_drho1_invX0TAX0_X0TAX \
            - dXTAX0_drho1_invX0TAX0_X0TAX.T \
            + np.dot(np.dot(invX0TAX0_X0TAX.T, dX0TAX0_drho1),
                     invX0TAX0_X0TAX)
        dLTXTAcorrXL_drho1 = np.dot(np.dot(L.T, dXTAcorrX_drho1), L)

        dYTAY_drho1 = - YTDY_diag + 2 * rho1 * YTFY_diag
        dX0TAY_drho1 = - X0TDY + 2 * rho1 * X0TFY
        invX0TAX0_X0TAY = np.linalg.solve(X0TAX0, X0TAY)
        dYTAX0_drho1_invX0TAX0_X0TAY = np.sum(dX0TAY_drho1
                                              * invX0TAX0_X0TAY, axis=0)

        dYTAcorrY_drho1 = dYTAY_drho1 - dYTAX0_drho1_invX0TAX0_X0TAY * 2\
            + np.sum(invX0TAX0_X0TAY *
                     np.dot(dX0TAX0_drho1, invX0TAX0_X0TAY), axis=0)

        dXTAY_drho1 = - XTDY + 2 * rho1 * XTFY
        dXTAcorrY_drho1 = dXTAY_drho1 \
            - np.dot(dXTAX0_drho1, invX0TAX0_X0TAY) \
            - np.dot(invX0TAX0_X0TAX.T, dX0TAY_drho1) \
            + np.dot(np.dot(invX0TAX0_X0TAX.T, dX0TAX0_drho1),
                     invX0TAX0_X0TAY)

        deriv_a1 = 2.0 / (np.pi * (1 + a1**2)) \
            * (n_V * (- n_run * rho1 / (1 - rho1**2)
                      - 0.5 * np.trace(np.linalg.solve(
                            X0TAX0, dX0TAX0_drho1))
                      - 0.5 * np.trace(np.linalg.solve(
                            LAMBDA_i, dLTXTAcorrXL_drho1)))
               - 0.5 * np.sum(dYTAcorrY_drho1) / sigma2
               + np.sum(dXTAcorrY_drho1 * L_LAMBDA_LTXTAcorrY) / sigma2
               - 0.5 * np.sum(np.dot(dXTAcorrX_drho1, L_LAMBDA_LTXTAcorrY)
                              * L_LAMBDA_LTXTAcorrY) / sigma2)

        deriv = np.empty(np.size(param))
        deriv[idx_param_sing['Cholesky']] = deriv_L[l_idx]
        deriv[idx_param_sing['a1']] = deriv_a1

        return -LL, -deriv
