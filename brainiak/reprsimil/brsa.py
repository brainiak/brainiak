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

    This implementation is based the work in [Cai2016]_.

 .. [Cai2016] "A Bayesian method for reducing bias in neural
    representational similarity analysis",
    M.B. Cai, N. Schuck, J. Pillow, Y. Niv,
    Advances in Neural Information Processing Systems 29, 2016, 4952--4960
    Available at:
    http://papers.nips.cc/paper/6131-a-bayesian-method-for-reducing-bias-in-neural-representational-similarity-analysis.pdf
    Some extensions beyond the paper have been made here.
    More specifically:
    (1) spatial noise correlation (or alternatively
    considered as signals of intrinsic fluctuation not related to tasks);
    (2) new fitting procedure which marginalizes all voxel-specific
    parameters such as pseudo-SNR, noise variance, auto-regressive
    coefficients, in `.GBRSA` class;
    (3) capacity to jointly fit to data of multiple participants,
    in `.GBRSA` class;
    (4) cross-validation score between a full model and a null model
    in `.BRSA.score` and `.GBRSA.score`;
    (5) capability of decoding task-related signals and intrinsic
    fluctuation from new data based on model fitted from training data
    in `.BRSA.transform` and `.GBRSA.transform`.
    `.GBRSA` may perform better than `.BRSA` due to (2). It can be
    use for single participant as well.
"""

# Authors: Mingbo Cai
# Princeton Neuroscience Institute, Princeton University, 2016

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import scipy.special
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite, check_random_state
from sklearn.decomposition import PCA, FactorAnalysis, SparsePCA, FastICA
import logging
import brainiak.utils.utils as utils
import scipy.spatial.distance as spdist
from nitime import algorithms as alg
import copy


logger = logging.getLogger(__name__)

__all__ = [
    "BRSA", "GBRSA", "prior_GP_var_inv_gamma", "prior_GP_var_half_cauchy",
    "Ncomp_SVHT_MG_DLD_approx"
]


def prior_GP_var_inv_gamma(y_invK_y, n_y, tau_range):
    """ Imposing an inverse-Gamma prior onto the variance (tau^2)
        parameter of a Gaussian Process, which is in turn a prior
        imposed over an unknown function y = f(x).
        The inverse-Gamma prior of tau^2, tau^2 ~ invgamma(shape, scale)
        is described by a shape parameter alpha=2 and a scale parameter
        beta=tau_range^2. tau_range describes the reasonable range of
        tau in the inverse-Gamma prior.
        The data y's at locations x's are assumed to follow Gaussian Process:
        f(x, x') ~ N(0, K(x, x') / 2 tau^2), where K is a kernel
        function defined on x. For n observations, K(x1, x2, ..., xn) is
        an n by n positive definite matrix.
        Given the prior parameter tau_range, number of observations
        n_y, and y_invK_y = y * inv(K) * y',
        the function returns the MAP estimate of tau^2 and
        the log posterior probability of tau^2 at the MAP value:
        log(p(tau^2|tau_range)).
        This function is written primarily for BRSA but can also
        be used elsewhere. y in this case corresponds to the log of
        SNR in each voxel. GBRSA does not rely on this function.
        An alternative form of prior is half-Cauchy prior on tau.
        Inverse-Gamma prior penalizes for both very small and very
        large values of tau, while half-Cauchy prior only penalizes
        for very large values of tau.
        For more information on usage, see description in BRSA class:
        `.BRSA`

        See also: `.prior_GP_var_half_cauchy`

    Parameters
    ----------
    y_invK_y: float
        y * inv(K) * y^T, where y=f(x) is a vector of observations
        of unknown function f at different locations x.
        K is correlation matrix of f between different locations, based
        on a Gaussian Process (GP) describing the smoothness property
        of f. K fully incorporates the form of the kernel
        and the length scale of the GP, but not the variance of the GP
        (the purpose of this function is to estimate the variance).
    n_y: int, number of observations
    tau_range: float,
        The reasonable range of tau, the standard deviation of the
        Gaussian Process imposed on y=f(x). tau_range is parameter
        of the inverse-Gamma prior. Say, if you expect the standard
        deviation of the Gaussian process to be around 3, tau_range
        can be set to 3.
        The smaller it is, the more penalization is imposed
        on large variation of y.
    Returns
    -------
    tau2: The MAP estimation of tau^2 based on the prior on tau
        and y_invK_y.
    log_ptau: log(p(tau)) of the returned tau^2 based on the
        inverse-Gamma prior.
    """
    alpha = 2
    tau2 = (y_invK_y + 2 * tau_range**2) / (alpha * 2 + 2 + n_y)
    log_ptau = scipy.stats.invgamma.logpdf(
        tau2, scale=tau_range**2, a=2)
    return tau2, log_ptau


def prior_GP_var_half_cauchy(y_invK_y, n_y, tau_range):
    """ Imposing a half-Cauchy prior onto the standard deviation (tau)
        of the Gaussian Process which is in turn a prior imposed over
        a function y = f(x).
        The scale parameter of the half-Cauchy prior is tau_range.
        The function returns the MAP estimate of tau^2 and
        log(p(tau|tau_range)) for the MAP value of tau^2,
        where tau_range describes the reasonable range of tau
        in the half-Cauchy prior.
        An alternative form of prior is inverse-Gamma prior on tau^2.
        Inverse-Gamma prior penalizes for both very small and very
        large values of tau, while half-Cauchy prior only penalizes
        for very large values of tau.
        For more information on usage, see description in BRSA class:
        `.BRSA`
    """
    tau2 = (y_invK_y - n_y * tau_range**2
            + np.sqrt(n_y**2 * tau_range**4 + (2 * n_y + 8)
                      * tau_range**2 * y_invK_y + y_invK_y**2))\
        / 2 / (n_y + 2)
    log_ptau = scipy.stats.halfcauchy.logpdf(
        tau2**0.5, scale=tau_range)
    return tau2, log_ptau


def Ncomp_SVHT_MG_DLD_approx(X, zscore=True):
    """ This function implements the approximate calculation of the
        optimal hard threshold for singular values, by Matan Gavish
        and David L. Donoho:
        "The optimal hard threshold for singular values is 4 / sqrt(3)"
        http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6846297

    Parameters
    ----------
    X: 2-D numpy array of size [n_T, n_V]
        The data to estimate the optimal rank for selecting principal
        components.
    zscore: Boolean, default: True
        Whether to z-score the data before calculating number of components.

    Returns
    -------
    ncomp: integer
        The optimal number of components determined by the method of MG
        and DLD
    """
    beta = X.shape[0] / X.shape[1]
    if beta > 1:
        beta = 1 / beta
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    if zscore:
        sing = np.linalg.svd(_zscore(X), False, False)
    else:
        sing = np.linalg.svd(X, False, False)
    thresh = omega * np.median(sing)
    ncomp = int(np.sum(np.logical_and(sing > thresh, np.logical_not(
        np.isclose(sing, thresh)))))
    # In the line above, we look for the singular values larger than
    # the threshold but excluding those that happen to be "just" larger
    # than the threshold by an amount close to the numerical precision.
    # This is to prevent close-to-zero singular values to be included if
    # the median of the eigenvalues is close to 0 (which could happen
    # when the input X has lower rank than its minimal size.
    return ncomp


def _zscore(a):
    """ Calculating z-score of data on the first axis.
        If the numbers in any column are all equal, scipy.stats.zscore
        will return NaN for this column. We shall correct them all to
        be zeros.

    Parameters
    ----------
    a: numpy array

    Returns
    -------
    zscore: numpy array
        The z-scores of input "a", with any columns including non-finite
        numbers replaced by all zeros.
    """
    assert a.ndim > 1, 'a must have more than one dimensions'
    zscore = scipy.stats.zscore(a, axis=0)
    zscore[:, np.logical_not(np.all(np.isfinite(zscore), axis=0))] = 0
    return zscore


class BRSA(BaseEstimator, TransformerMixin):
    """Bayesian representational Similarity Analysis (BRSA)

    Given the time series of neural imaging data in a region of interest
    (ROI) and the hypothetical neural response (design matrix) to
    each experimental condition of interest,
    calculate the shared covariance matrix U of
    the voxels(recording unit)' response profiles \\beta_i to each condition,
    and the relative SNR of each voxels.
    The relative SNR could be considered as the degree of contribution
    of each voxel to this shared covariance matrix.
    A correlation matrix converted from the covariance matrix U
    will be provided as a quantification of neural representational similarity.

    .. math::
        Y = X \\cdot \\beta + X_0 \\cdot \\beta_0 + \\epsilon

        \\beta_i \\sim N(0,(s_{i} \\sigma_{i})^2 U)

        \\epsilon_i \\sim AR(1)

    Parameters
    ----------
    n_iter : int. Default: 50
        Number of maximum iterations to run the algorithm.
    rank : int. Default: None
        The rank of the covariance matrix.
        If not provided, the covariance matrix will be assumed
        to be full rank. When you have many conditions
        (e.g., calculating the similarity matrix of responses to each event),
        you might try specifying a lower rank.
    auto_nuisance: boolean. Default: True
        In order to model spatial correlation between voxels that cannot
        be accounted for by common response captured in the design matrix,
        we assume that a set of time courses not related to the task
        conditions are shared across voxels with unknown amplitudes.
        One approach is for users to provide time series which they consider
        as nuisance but exist in the noise (such as head motion).
        The other way is to take the first n_nureg principal components
        in the residual after subtracting the response to the design matrix
        from the data, and use these components as the nuisance regressor.
        This flag is for the second approach. If turned on,
        PCA or factor analysis will be applied to the residuals
        to obtain new nuisance regressors in each round of fitting.
        These two approaches can be combined. If the users provide nuisance
        regressors and set this flag as True, then the first n_nureg
        principal components of the residuals after subtracting
        both the responses to design matrix and the user-supplied nuisance
        regressors will be used in addition to the nuisance regressors
        provided by the users.
        Note that nuisance regressor is not required from user. If it is
        not provided, DC components for each run will be included as nuisance
        regressor regardless of the auto_nuisance parameter.
    n_nureg: Optional[int]. Default: None
        Number of nuisance regressors to use in order to model signals
        shared across voxels not captured by the design matrix.
        This number is in addition to any nuisance regressor that the user
        has already provided.
        If set to None, the number of nuisance regressors will be
        automatically determined based on M Gavish
        and D Donoho's approximate estimation of optimal hard
        threshold for singular values.
        This only takes effect if auto_nuisance is True.
    nureg_zscore: boolean, default: True
        A flag to tell the algorithm whether data is z-scored before
        estimating the number of nuisance regressor components necessary to
        account for spatial noise correlation. It also determinie whether
        the residual noise is z-scored before estimating the nuisance
        regressors from residual.
        This only takes effect if auto_nuisance is True.
    nureg_method: string, naming a method from sklearn.decomposition.
        'PCA', 'ICA', 'FA' or 'SPCA' are currently supported. Default: 'PCA'
        The method to estimate the shared component in noise across voxels.
        This only takes effect if auto_nuisance is True.
    baseline_single: boolean, default: False
        A time course of constant 1 will be included to the nuisance
        regressor regardless of whether the user requests.
        If baseline_single is set to False, one such regressor is included
        for each fMRI run, but a single component in beta0\_ will be
        computed as the average of the weight maps corresponding to
        these regressors. This might cause underestimation of noise variance.
        If baseline_single is True, only one regressor of constant 1 will be
        used for the whole dataset. This might be desirable if you
        believe the average image intensity might not scale with the
        same proportion for different voxels across scan. In other words,
        it is possible that some part of the brain is more vulnerable to
        change in baseline intensity due to facts such as
        field inhomogeneity. Setting baseline_single to True will force the
        nuisance regressors automatically estimated from residuals to
        capture this. However, when each task condition only occurs in one
        run and when the design matrix in each run sums together close to
        a flat line, this option can cause the estimated similarity to be
        extremely high between conditions occuring in the same run.
    GP_space: boolean. Default: False
        Whether to impose a Gaussion Process (GP) prior on the log(pseudo-SNR).
        If true, the GP has a kernel defined over spatial coordinate
        of each voxel. The idea behind this option is that
        adjacent voxels should have similar SNRs.
        This is relatively slow for big ROI. We find that when SNR
        is generally low, smoothness can be overestimated.
        But such regularization may reduce variance in the estimated
        SNR map and similarity matrix.
    GP_inten: boolean. Defualt: False
        Whether to include a kernel defined over the intensity of image.
        GP_space should be True as well if you want to use this,
        because the smoothness should be primarily in space.
        Smoothness in intensity is just complementary. The idea
        behind this option is that voxels should have similar
        SNRs when they are both adjacent (imposed by GP_space)
        and are of the same tissue type (when their image intensities
        are close). If you accept the second assumption, then
        you can set GP_inten as True and provide an array to the `inten`
        variable, expressing the intensities (brightness) for each voxel.
    space_smooth_range: float. Default: None
        The distance (in unit the same as what
        you would use when supplying the spatial coordiates of
        each voxel, typically millimeter) which you believe is
        the maximum range of the length scale parameter of
        Gaussian Process defined over voxel location. This is
        used to impose a half-Cauchy prior on the length scale.
        If not provided, the program will set it to half of the
        maximum distance between all voxels.
    inten_smooth_range: float. Default: None
        The difference in image intensity which
        you believe is the maximum range of plausible length
        scale for the Gaussian Process defined over image
        intensity. Length scales larger than this are allowed,
        but will be penalized. If not supplied, this parameter
        will be set to half of the maximal intensity difference.
    tau_range: float. Default: 5.0
        The reasonable range of the standard deviation
        of log(SNR). This range should not be too
        large. 5 is a loose range.
        When a Gaussian Process is imposed on the log(SNR),
        this parameter is used in a half-Cauchy prior
        on the standard deviation, or an inverse-Gamma prior
        on the variance of the GP.
    tau2_prior: Callable[[float, int, float]], [float, float]],
        Default: prior_GP_var_inv_gamma.
        Can be prior_GP_var_inv_gamma or prior_GP_var_half_cauchy,
        or a custom function.
        The function which impose a prior for tau^2, the variance of the
        GP prior on log(SNR), and returns the MAP estimate of tau^2.
        It can be either prior_GP_var_inv_gamma for inverse-Gamma
        or prior_GP_var_half_cauchy for half-Cauchy.
        half-Cauchy prior is in fact imposed on tau.
        But tau_range describes the range of tau in the prior in both cases.
        Both functions are part of brsa module.
        See also `.prior_GP_var_inv_gamma` and
        `.prior_GP_var_half_cauchy`
        To use the default inverse-Gamma prior, you can ignore this argument::

            from brainiak.reprsimil.brsa import BRSA
            brsa = BRSA()

        If you want to try the alternative half-Cauchy prior,
        then you need to import it in addition to BRSA::

            from brainiak.reprsimil.brsa import BRSA, prior_GP_var_half_cauchy
            brsa = BRSA(tau2_prior=prior_GP_var_half_cauchy)

    eta: float. Default: 0.0001
        A small number added to the diagonal element of the
        covariance matrix in the Gaussian Process prior. This is
        to ensure that the matrix is invertible.
    init_iter: int. Default: 20
        How many initial iterations to fit the model
        without introducing the GP prior before fitting with it,
        if GP_space or GP_inten is requested. This initial
        fitting is to give the parameters a good starting point.
    optimizer: str or callable. Default: 'BFGS'
        The optimizer to use for minimizing cost function which
        scipy.optimize.minimize can accept.
        We use 'L-BFGS-B' as a default. Users can try other strings
        corresponding to optimizer provided by scipy.optimize.minimize,
        or a custom optimizer, such as 'L-BFGS-B' or 'CG'.
        Note that BRSA fits a lot of parameters. So a chosen optimizer
        should accept gradient (Jacobian) of the cost function. Otherwise
        the fitting is likely to be unbarely slow. We do not calculate
        Hessian of the objective function. So an optimizer which requires
        Hessian cannot be used.
    random_state : RandomState or an int seed. Default: None
        A random number generator instance to define the state of
        the random permutations generator whenever the module
        needs to generate random number (e.g., initial parameter
        of the Cholesky factor).
    anneal_speed: float. Default: 20
        Annealing is introduced in fitting of the Cholesky
        decomposition of the shared covariance matrix. The amount
        of perturbation decays exponentially. This parameter sets
        the ratio of the maximum number of iteration to the
        time constant of the exponential.
        anneal_speed=10 means by n_iter/10 iterations,
        the amount of perturbation is reduced by 2.713 times.
    minimize_options: dictionary.
        Default: {'gtol': 1e-4, 'disp': False, 'maxiter': 6}
        This is the dictionary passed as the options argument to
        scipy.optimize.minize which minimizes the cost function during
        fitting. Notice that the minimization is performed for many times,
        alternating between optimizing the covariance matrix U underlying
        the pattern similarity matrix, and SNR. At most n_iter times
        of this alternation is performed. So within each step of fitting,
        the step of iteration performed by scipy.optimize.minize does not
        have to be very large. In other words, scipy.optimize.minize does
        not need to converge within each step of the alternating fitting
        procedure.
    tol: float. Default: 1e-4.
        Tolerance parameter passed to scipy.optimize.minimize. It is also
        used for determining convergence of the alternating fitting
        procedure.

    Attributes
    ----------
    U_ : numpy array, shape=[condition,condition].
        The shared covariance matrix.
    L_ : numpy array, shape=[condition,rank].
        The Cholesky factor of the shared covariance matrix
        (lower-triangular matrix).
    C_: numpy array, shape=[condition,condition].
        The correlation matrix derived from the shared covariance matrix.
        This is the estimated similarity matrix between neural patterns
        to your task conditions. Notice that it is recommended that
        you also check U\_, which is the covariance matrix underlying
        this correlation matrix. In cases there is almost no response
        to your task conditions, the diagonal values of U\_ would become
        very small and C\_ might contain many correlation coefficients
        close to 1 or -1. This might not reflect true strong correlation
        or strong negative correlation, but a result of lack of
        task-related neural activity, design matrix that does not match
        true neural response, or not enough data.
        It is also recommended to check nSNR\_ after mapping it back to
        the brain. A "reasonable" map should at least have higher values
        in gray matter in than white matter.
    nSNR_ : numpy array, shape=[voxels,].
        The normalized pseuso-SNR of all voxels.
        They are normalized such that the geometric mean is 1.
        Note that this attribute can not be interpreted as true SNR,
        but the relative ratios between voxel indicates the contribution
        of each voxel to the representational similarity structure.
    sigma_ : numpy array, shape=[voxels,].
        The estimated standard deviation of the noise in each voxel
        Assuming AR(1) model, this means the standard deviation
        of the innovation noise.
    rho_ : numpy array, shape=[voxels,].
        The estimated autoregressive coefficient of each voxel
    bGP_ : float, only if GP_space or GP_inten is True.
        The standard deviation of the GP prior
    lGPspace_ : float, only if GP_space or GP_inten is True
        The length scale of Gaussian Process prior of log(SNR)
    lGPinten_: float, only if GP_inten is True
        The length scale in fMRI intensity of the GP prior of log(SNR)
    beta_: array, shape=[conditions, voxels]
        The maximum a posterior estimation of the response amplitudes
        of each voxel to each task condition.
    beta0_: numpy array, shape=[n_nureg + n_base, voxels]
        The loading weights of each voxel for the shared time courses
        not captured by the design matrix. This helps capture the
        structure of spatial covariance of task-unrelated signal.
        n_base is the number of columns of the user-supplied nuisance
        regressors plus one for DC component
    X0_: numpy array, shape=[time_points, n_nureg + n_base]
        The estimated time course that is shared across voxels but
        unrelated to the events of interest (design matrix).
    beta0_null_: numpy array, shape=[n_nureg + n_base, voxels]
        The equivalent of beta0\_ in a null model which does not
        include the design matrix and response pattern beta.
    X0_null_: numpy array, shape=[time_points, n_nureg + n_base]
        The equivalent of X0\_ in a null model which does not
        include the design matrix and response pattern beta
    n_nureg_: int
        Number of nuisance regressor in addition to such
        regressors provided by the user (if any), if auto_nuisance
        is set to True. If n_nureg is set to 'opt',
        this will be estimated from data. 'opt' will use M Gavish
        and D Donoho's approximate estimation of optimal hard
        threshold for singular values.
    random_state_: `RandomState`
        Random number generator initialized using random_state.

    """

    def __init__(
            self, n_iter=50, rank=None,
            auto_nuisance=True, n_nureg=None, nureg_zscore=True,
            nureg_method='PCA', baseline_single=False,
            GP_space=False, GP_inten=False,
            space_smooth_range=None, inten_smooth_range=None,
            tau_range=5.0,
            tau2_prior=prior_GP_var_inv_gamma,
            eta=0.0001, init_iter=20, optimizer='BFGS',
            random_state=None, anneal_speed=10, tol=1e-4,
            minimize_options={'gtol': 1e-4, 'disp': False,
                              'maxiter': 6}):
        self.n_iter = n_iter
        self.rank = rank
        self.GP_space = GP_space
        self.GP_inten = GP_inten
        self.tol = tol
        self.auto_nuisance = auto_nuisance
        self.n_nureg = n_nureg
        self.nureg_zscore = nureg_zscore
        if auto_nuisance:
            assert (n_nureg is None) \
                or (isinstance(n_nureg, int) and n_nureg > 0), \
                'n_nureg should be a positive integer or None'\
                ' if auto_nuisance is True.'
        if self.nureg_zscore:
            self.preprocess_residual = lambda x: _zscore(x)
        else:
            self.preprocess_residual = lambda x: x
        if nureg_method == 'FA':
            self.nureg_method = lambda x: FactorAnalysis(n_components=x)
        elif nureg_method == 'PCA':
            self.nureg_method = lambda x: PCA(n_components=x, whiten=True)
        elif nureg_method == 'SPCA':
            self.nureg_method = lambda x: SparsePCA(n_components=x,
                                                    max_iter=20, tol=tol)
        elif nureg_method == 'ICA':
            self.nureg_method = lambda x: FastICA(n_components=x,
                                                  whiten=True)
        else:
            raise ValueError('nureg_method can only be FA, PCA, '
                             'SPCA(for sparse PCA) or ICA')
        self.baseline_single = baseline_single
        self.minimize_options = minimize_options
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
        self.tau2_prior = tau2_prior
        self.init_iter = init_iter
        # When imposing smoothness prior, fit the model without this
        # prior for this number of iterations.
        self.optimizer = optimizer
        self.random_state = random_state
        self.anneal_speed = anneal_speed
        return

    def fit(self, X, design, nuisance=None, scan_onsets=None, coords=None,
            inten=None):
        """Compute the Bayesian RSA

        Parameters
        ----------
        X: numpy array, shape=[time_points, voxels]
            If you have multiple scans of the same participants that you
            want to analyze together, you should concatenate them along
            the time dimension after proper preprocessing (e.g. spatial
            alignment), and specify the onsets of each scan in scan_onsets.
        design: numpy array, shape=[time_points, conditions]
            This is the design matrix. It should only include the hypothetic
            response for task conditions. You should not include
            regressors for a DC component or motion parameters, unless you
            want to estimate their pattern similarity with response patterns
            to your task conditions. If you want to model head motion,
            you should include them in nuisance regressors.
            If you have multiple run, the design matrix
            of all runs should be concatenated along the time dimension,
            with every column for one condition across runs.
            For example, if you have 3 runs of experiment of one participant,
            with each run lasting 200 TR. And you have 4 conditions,
            then design should be a 600 x 4 numpy array.
        nuisance: optional, numpy array, shape=[time_points, nuisance_factors]
            The responses to these regressors will be marginalized out from
            each voxel, which means they are considered, but won't be assumed
            to share the same pseudo-SNR map with the design matrix.
            Therefore, the pseudo-SNR map will only reflect the
            relative contribution of design matrix to each voxel.
            You can provide time courses such as those for head motion
            to this parameter.
            Note that if auto_nuisance is set to True, the first
            n_nureg principal components of residual (excluding the response
            to the design matrix and the user-provided nuisance regressors
            and a constant baseline)
            will be included as additional nuisance regressor after the
            first round of fitting.
            If auto_nuisance is set to False, the nuisance regressors supplied
            by the users together with DC components will be used as
            nuisance time series.
            Please do not include time course of constant baseline in nuisance.
        scan_onsets: optional, numpy array, shape=[runs,]
            This specifies the indices of X which correspond to the onset
            of each scanning run. For example, if you have two experimental
            runs of the same subject, each with 100 TRs, then scan_onsets
            should be [0,100].
            If you do not provide the argument, the program will
            assume all data are from the same run.
            The effect of them is to make the inverse matrix
            of the temporal covariance matrix of noise block-diagonal.
        coords: optional, numpy array, shape=[voxels,3]
            This is the coordinate of each voxel,
            used for implementing Gaussian Process prior.
        inten: optional, numpy array, shape=[voxel,]
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
        self.random_state_ = check_random_state(self.random_state)
        # setting random seed
        logger.debug('RandState set to {}'.format(self.random_state_))

        assert not self.GP_inten or (self.GP_inten and self.GP_space),\
            'You must speficiy GP_space to True'\
            'if you want to use GP_inten'

        # Check input data
        assert_all_finite(X)
        assert X.ndim == 2, 'The data should be 2-dimensional ndarray'

        assert np.all(np.std(X, axis=0) > 0),\
            'The time courses of some voxels do not change at all.'\
            ' Please make sure all voxels are within the brain'

        # check design matrix
        assert_all_finite(design)
        assert design.ndim == 2,\
            'The design matrix should be 2-dimensional ndarray'
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
                'The nuisance regressor should be 2-dimensional ndarray'
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
                '[n_voxel x 3].'
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
        # Estimate the number of necessary nuisance regressors
        if self.auto_nuisance:
            if self.n_nureg is None:
                logger.info('number of nuisance regressors is determined '
                            'automatically.')
                run_TRs, n_runs = self._run_TR_from_scan_onsets(
                    X.shape[0], scan_onsets)
                ts_dc = self._gen_legendre(run_TRs, [0])
                _, ts_base, _ = self._merge_DC_to_base(
                    ts_dc, nuisance, False)
                ts_reg = np.concatenate((ts_base, design), axis=1)
                beta_hat = np.linalg.lstsq(ts_reg, X)[0]
                residuals = X - np.dot(ts_reg, beta_hat)
                self.n_nureg_ = np.max(
                    [1, Ncomp_SVHT_MG_DLD_approx(residuals,
                                                 self.nureg_zscore)])
                logger.info('Use {} nuisance regressors to model the spatial '
                            'correlation in noise.'.format(self.n_nureg_))
                self.n_nureg_ = np.int32(self.n_nureg_)
            else:
                self.n_nureg_ = self.n_nureg
            self.n_nureg_ = np.int32(self.n_nureg_)
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
                self._beta_latent_, self.sigma_, self.rho_, _, _, _,\
                self.X0_ = self._fit_RSA_UV(X=design, Y=X, X_base=nuisance,
                                            scan_onsets=scan_onsets)
        elif not self.GP_inten:
            # If GP_space is requested, but GP_inten is not, a GP prior
            # based on spatial locations of voxels will be imposed.
            self.U_, self.L_, self.nSNR_, self.beta_, self.beta0_,\
                self._beta_latent_, self.sigma_, self.rho_, \
                self.lGPspace_, self.bGP_, _, \
                self.X0_ = self._fit_RSA_UV(
                    X=design, Y=X, X_base=nuisance,
                    scan_onsets=scan_onsets, coords=coords)
        else:
            # If both self.GP_space and self.GP_inten are True,
            # a GP prior based on both location and intensity is imposed.
            self.U_, self.L_, self.nSNR_, self.beta_, self.beta0_,\
                self._beta_latent_, self.sigma_, self.rho_, \
                self.lGPspace_, self.bGP_, self.lGPinten_, self.X0_ = \
                self._fit_RSA_UV(X=design, Y=X, X_base=nuisance,
                                 scan_onsets=scan_onsets,
                                 coords=coords, inten=inten)

        self.C_ = utils.cov2corr(self.U_)
        self.design_ = design.copy()
        self._rho_design_, self._sigma2_design_ = \
            self._est_AR1(self.design_, same_para=True)
        self._rho_X0_, self._sigma2_X0_ = self._est_AR1(self.X0_)
        # AR(1) parameters of the design matrix and nuisance regressors,
        # which will be used in transform or score.

        # Finally, we fit a null model with the same setting except
        # that there is no response to X
        self.beta0_null_, self.sigma_null_, self.rho_null_, \
            self.X0_null_ = self._fit_null(Y=X, X_base=nuisance,
                                           scan_onsets=scan_onsets)
        self._rho_X0_null_, self._sigma2_X0_null_ =\
            self._est_AR1(self.X0_null_)
        return self

    def transform(self, X, y=None, scan_onsets=None):
        """ Use the model to estimate the time course of response to
            each condition (ts), and the time course unrelated to task
            (ts0) which is spread across the brain.
            This is equivalent to "decoding" the design matrix and
            nuisance regressors from a new dataset different from the
            training dataset on which fit() was applied. An AR(1) smooth
            prior is imposed on the decoded ts and ts0 with the AR(1)
            parameters learnt from the corresponding time courses in the
            training data.
            Notice: if you set the rank to be lower than the number of
            experimental conditions (number of columns in the design
            matrix), the recovered task-related activity will have
            collinearity (the recovered time courses of some conditions
            can be linearly explained by the recovered time courses
            of other conditions).

        Parameters
        ----------
        X : numpy arrays, shape=[time_points, voxels]
            fMRI data of new data of the same subject. The voxels should
            match those used in the fit() function. If data are z-scored
            (recommended) when fitting the model, data should be z-scored
            as well when calling transform()
        y : not used (as it is unsupervised learning)
        scan_onsets : numpy array, shape=[number of runs]. Default: None.
            A list of indices corresponding to the onsets of
            scans in the data X. If not provided, data will be assumed
            to be acquired in a continuous scan.
        Returns
        -------
        ts : numpy arrays, shape = [time_points, condition]
            The estimated response to the task conditions which have the
            response amplitudes estimated during the fit step.
        ts0: numpy array, shape = [time_points, n_nureg]
            The estimated time course spread across the brain, with the
            loading weights estimated during the fit step.
        """

        assert X.ndim == 2 and X.shape[1] == self.beta_.shape[1], \
            'The shape of X is not consistent with the shape of data '\
            'used in the fitting step. They should have the same number '\
            'of voxels'
        assert scan_onsets is None or (scan_onsets.ndim == 1 and
                                       0 in scan_onsets), \
            'scan_onsets should either be None or an array of indices '\
            'If it is given, it should include at least 0'

        if scan_onsets is None:
            scan_onsets = np.array([0], dtype=int)
        else:
            scan_onsets = np.int32(scan_onsets)
        ts, ts0, log_p = self._transform(
            Y=X, scan_onsets=scan_onsets, beta=self.beta_,
            beta0=self.beta0_, rho_e=self.rho_, sigma_e=self.sigma_,
            rho_X=self._rho_design_, sigma2_X=self._sigma2_design_,
            rho_X0=self._rho_X0_, sigma2_X0=self._sigma2_X0_)
        return ts, ts0

    def score(self, X, design, scan_onsets=None):
        """ Use the model and parameters estimated by fit function
            from some data of a participant to evaluate the log
            likelihood of some new data of the same participant.
            Design matrix of the same set of experimental
            conditions in the testing data should be provided, with each
            column corresponding to the same condition as that column
            in the design matrix of the training data.
            Unknown nuisance time series will be marginalized, assuming
            they follow the same spatial pattern as in the training
            data. The hypothetical response captured by the design matrix
            will be subtracted from data before the marginalization
            when evaluating the log likelihood. For null model,
            nothing will be subtracted before marginalization.

            There is a difference between the form of likelihood function
            used in fit() and score(). In fit(), the response amplitude
            beta to design matrix X and the modulation beta0 by nuisance
            regressor X0 are both marginalized, with X provided and X0
            estimated from data. In score(), posterior estimation of
            beta and beta0 from the fitting step are assumed unchanged
            to testing data and X0 is marginalized.
            The logic underlying score() is to transfer
            as much as what we can learn from training data when
            calculating a likelihood score for testing data.

            If you z-scored your data during fit step, you should
            z-score them for score function as well. If you did not
            z-score in fitting, you should not z-score here either.

        Parameters
        ----------
        X : numpy arrays, shape=[time_points, voxels]
            fMRI data of new data of the same subject. The voxels should
            match those used in the fit() function. If data are z-scored
            (recommended) when fitting the model, data should be z-scored
            as well when calling transform()
        design : numpy array, shape=[time_points, conditions]
            Design matrix expressing the hypothetical response of
            the task conditions in data X.
        scan_onsets : numpy array, shape=[number of runs]. Default: None.
            A list of indices corresponding to the onsets of
            scans in the data X. If not provided, data will be assumed
            to be acquired in a continuous scan.
        Returns
        -------
        ll: float,
            The log likelihood of the new data based on the model and its
            parameters fit to the training data.
        ll_null: float,
            The log likelihood of the new data based on a null model
            which assumes the same as the full model for everything
            except for that there is no response to any of the
            task conditions.
        """

        assert X.ndim == 2 and X.shape[1] == self.beta_.shape[1], \
            'The shape of X is not consistent with the shape of data '\
            'used in the fitting step. They should have the same number '\
            'of voxels'
        assert scan_onsets is None or (scan_onsets.ndim == 1 and
                                       0 in scan_onsets), \
            'scan_onsets should either be None or an array of indices '\
            'If it is given, it should include at least 0'
        if scan_onsets is None:
            scan_onsets = np.array([0], dtype=int)
        else:
            scan_onsets = np.int32(scan_onsets)
        ll = self._score(Y=X, design=design, beta=self.beta_,
                         scan_onsets=scan_onsets, beta0=self.beta0_,
                         rho_e=self.rho_, sigma_e=self.sigma_,
                         rho_X0=self._rho_X0_, sigma2_X0=self._sigma2_X0_)
        ll_null = self._score(Y=X, design=None, beta=None,
                              scan_onsets=scan_onsets, beta0=self.beta0_,
                              rho_e=self.rho_, sigma_e=self.sigma_,
                              rho_X0=self._rho_X0_,
                              sigma2_X0=self._sigma2_X0_)
        return ll, ll_null

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

    def _run_TR_from_scan_onsets(self, n_T, scan_onsets=None):
        if scan_onsets is None:
            # assume that all data are acquired within the same scan.
            n_run = 1
            run_TRs = np.array([n_T], dtype=int)
        else:
            # Each value in the scan_onsets tells the index at which
            # a new scan starts. For example, if n_T = 500, and
            # scan_onsets = [0,100,200,400], this means that the time points
            # of 0-99 are from the first scan, 100-199 are from the second,
            # 200-399 are from the third and 400-499 are from the fourth
            run_TRs = np.int32(np.diff(np.append(scan_onsets, n_T)))
            run_TRs = np.delete(run_TRs, np.where(run_TRs == 0))
            n_run = run_TRs.size
            # delete run length of 0 in case of duplication in scan_onsets.
            logger.info('I infer that the number of volumes'
                        ' in each scan are: {}'.format(run_TRs))
        return run_TRs, n_run

    def _prepare_DF(self, n_T, scan_onsets=None):
        """ Prepare the essential template matrices D and F for
            pre-calculating some terms to be re-used.
            The inverse covariance matrix of AR(1) noise is
            sigma^-2 * (I - rho1*D + rho1**2 * F).
            And we denote A = I - rho1*D + rho1**2 * F"""
        run_TRs, n_run = self._run_TR_from_scan_onsets(n_T, scan_onsets)
        D_ele = map(self._D_gen, run_TRs)
        F_ele = map(self._F_gen, run_TRs)
        D = scipy.linalg.block_diag(*D_ele)
        F = scipy.linalg.block_diag(*F_ele)
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

    def _gen_X_DC(self, run_TRs):
        if self.baseline_single:
            X_DC = np.ones((np.sum(run_TRs), 1))
        else:
            X_DC = scipy.linalg.block_diag(*map(np.ones, run_TRs)).T
        return X_DC

    def _gen_legendre(self, run_TRs, orders):
        def reg(x):
            return np.concatenate(
                [scipy.special.legendre(o)(np.linspace(-1, 1, x))[None, :]
                 for o in orders], axis=0)
        reg_poly = scipy.linalg.block_diag(
            *map(reg, run_TRs)).T
        return reg_poly

    def _prepare_data_XYX0(self, X, Y, X_base, X_res, D, F, run_TRs,
                           no_DC=False):
        """Prepares different forms of products between design matrix X or
            data Y or nuisance regressors X0.
            These products are re-used a lot during fitting.
            So we pre-calculate them.
            no_DC means not inserting regressors for DC components
            into nuisance regressor.
            It will only take effect if X_base is not None.
        """
        X_DC = self._gen_X_DC(run_TRs)
        reg_sol = np.linalg.lstsq(X_DC, X)
        if np.any(np.isclose(reg_sol[1], 0)):
            raise ValueError('Your design matrix appears to have '
                             'included baseline time series.'
                             'Either remove them, or move them to'
                             ' nuisance regressors.')
        X_DC, X_base, idx_DC = self._merge_DC_to_base(X_DC, X_base,
                                                      no_DC)
        if X_res is None:
            X0 = X_base
        else:
            X0 = np.concatenate((X_base, X_res), axis=1)
        n_X0 = X0.shape[1]
        X0TX0, X0TDX0, X0TFX0 = self._make_templates(D, F, X0, X0)
        XTX0, XTDX0, XTFX0 = self._make_templates(D, F, X, X0)
        X0TY, X0TDY, X0TFY = self._make_templates(D, F, X0, Y)

        return X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, \
            X0TY, X0TDY, X0TFY, X0, X_base, n_X0, idx_DC

    def _merge_DC_to_base(self, X_DC, X_base, no_DC):
        """ Merge DC components X_DC to the baseline time series
            X_base (By baseline, this means any fixed nuisance
            regressors not updated during fitting, including DC
            components and any nuisance regressors provided by
            the user.
            X_DC is always in the first few columns of X_base.
        """
        if X_base is not None:
            reg_sol = np.linalg.lstsq(X_DC, X_base)
            if not no_DC:
                if not np.any(np.isclose(reg_sol[1], 0)):
                    # No columns in X_base can be explained by the
                    # baseline regressors. So we insert them.
                    X_base = np.concatenate((X_DC, X_base), axis=1)
                    idx_DC = np.arange(0, X_DC.shape[1])
                else:
                    logger.warning('Provided regressors for uninteresting '
                                   'time series already include baseline. '
                                   'No additional baseline is inserted.')
                    idx_DC = np.where(np.isclose(reg_sol[1], 0))[0]
            else:
                idx_DC = np.where(np.isclose(reg_sol[1], 0))[0]
        else:
            # If a set of regressors for non-interested signals is not
            # provided, then we simply include one baseline for each run.
            X_base = X_DC
            idx_DC = np.arange(0, X_base.shape[1])
            logger.info('You did not provide time series of no interest '
                        'such as DC component. Trivial regressors of'
                        ' DC component are included for further modeling.'
                        ' The final covariance matrix won''t '
                        'reflect these components.')
        return X_DC, X_base, idx_DC

    def _make_ar1_quad_form(self, XTX, XTDX, XTFX, rho1):
        # Calculate the matrix X'AX = X'X - rho1 * X'DX + rho1^2 * X'FX
        # Here, rho1 is the AR(1) coefficient. X is a matrix of time series
        # with each row corresponding to a vector at one
        # time point. The forms of matrices D and F are defined in _prepare_DF
        # function. sigma^-2 * A would be the inverse of covariance matrix
        # of AR(1) process (precision matrix) with rho1 as the AR coefficient
        # and sigma^2 as the  variance of independent noise at each time point.
        return XTX - rho1 * XTDX + rho1**2 * XTFX

    def _make_ar1_quad_form_grad(self, XTDX, XTFX, rho1):
        # Calculate the derivative of the quadratic form X'AX with respect to
        # AR1 coefficient rho1, given precalculated terms X'DX and X'FX,
        # and rho1.
        return - XTDX + 2 * rho1 * XTFX

    def _make_templates(self, D, F, X, Y):
        XTY = np.dot(X.T, Y)
        XTDY = np.dot(np.dot(X.T, D), Y)
        XTFY = np.dot(np.dot(X.T, F), Y)
        return XTY, XTDY, XTFY

    def _precompute_ar1_quad_forms(self, XTY, XTDY, XTFY, YTY_diag, YTDY_diag,
                                   YTFY_diag, XTX, XTDX, XTFX, X0TX0, X0TDX0,
                                   X0TFX0, XTX0, XTDX0, XTFX0, X0TY, X0TDY,
                                   X0TFY, L, rho1, n_V, n_X0):
        # Calculate the sandwich terms which put A between X, Y and X0
        # These terms are used a lot in the likelihood. But in the _fitV
        # step, they only need to be calculated once, since A is fixed.
        # In _fitU step, they need to be calculated at each iteration,
        # because rho1 changes.
        XTAY = self._make_ar1_quad_form(XTY, XTDY, XTFY, rho1)
        # dimension: feature*space
        YTAY = self._make_ar1_quad_form(YTY_diag, YTDY_diag, YTFY_diag, rho1)
        # dimension: space,
        # A/sigma2 is the inverse of noise covariance matrix in each voxel.
        # YTAY means Y'AY
        XTAX = XTX[None, :, :] - rho1[:, None, None] \
            * XTDX[None, :, :] \
            + rho1[:, None, None]**2 * XTFX[None, :, :]
        # dimension: space*feature*feature
        X0TAX0 = X0TX0[None, :, :] - rho1[:, None, None] \
            * X0TDX0[None, :, :] \
            + rho1[:, None, None]**2 * X0TFX0[None, :, :]
        # dimension: space*#baseline*#baseline
        XTAX0 = XTX0[None, :, :] - rho1[:, None, None] \
            * XTDX0[None, :, :] \
            + rho1[:, None, None]**2 * XTFX0[None, :, :]
        # dimension: space*feature*#baseline
        X0TAY = self._make_ar1_quad_form(X0TY, X0TDY, X0TFY, rho1)
        # dimension: #baseline*space
        X0TAX0_i = np.linalg.solve(X0TAX0, np.identity(n_X0)[None, :, :])
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
                 n_V, n_T, n_run, rank, n_X0):
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
                  * SNR2) / (n_T - n_X0)
        # dimension: space
        LL = - np.sum(np.log(sigma2)) * (n_T - n_X0) * 0.5 \
            + np.sum(np.log(1 - rho1**2)) * n_run * 0.5 \
            - np.sum(self._half_log_det(X0TAX0)) \
            - np.sum(self._half_log_det(LAMBDA_i)) \
            - (n_T - n_X0) * n_V * (1 + np.log(2 * np.pi)) * 0.5
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

    def _half_log_det(self, M):
        """ Return log(|M|)*0.5. For positive definite matrix M
            of more than 2 dimensions, calculate this for the
            last two dimension and return a value corresponding
            to each element in the first few dimensions.
        """
        chol = np.linalg.cholesky(M)
        if M.ndim == 2:
            return np.sum(np.log(np.abs(np.diag(chol))))
        else:
            return np.sum(np.log(np.abs(np.diagonal(
                chol, axis1=-2, axis2=-1))), axis=-1)

    def _chol_idx(self, n_C, rank):
        l_idx = np.tril_indices(n_C)
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
        return l_idx, rank

    def _fit_RSA_UV(self, X, Y, X_base,
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
        l_idx, rank = self._chol_idx(n_C, rank)
        n_l = np.size(l_idx[0])  # the number of parameters for L

        t_start = time.time()

        D, F, run_TRs, n_run = self._prepare_DF(
            n_T, scan_onsets=scan_onsets)
        XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX, \
            XTDX, XTFX = self._prepare_data_XY(X, Y, D, F)

        X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, \
            X0TY, X0TDY, X0TFY, X0, X_base, n_X0, idx_DC = \
            self._prepare_data_XYX0(
                X, Y, X_base, None, D, F, run_TRs, no_DC=False)
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
                l_idx, n_C, n_T, n_V, n_l, n_run, n_X0, rank)

        current_logSNR2 = -current_logSigma2
        norm_factor = np.mean(current_logSNR2)
        current_logSNR2 = current_logSNR2 - norm_factor
        X_res = None

        # Step 2 fitting, which only happens if
        # GP prior is requested
        if GP_space:
            current_vec_U_chlsk_l, current_a1, current_logSNR2, X_res\
                = self._fit_diagV_noGP(
                    XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
                    XTX, XTDX, XTFX, X, Y, X_base, X_res, D, F, run_TRs,
                    current_vec_U_chlsk_l,
                    current_a1, current_logSNR2,
                    idx_param_fitU, idx_param_fitV,
                    l_idx, n_C, n_T, n_V, n_l, n_run, n_X0, rank)

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
                        inten_diff2, k=-1)], 2), 0.5))
                logger.debug(
                    'current GP[1]:{}'.format(current_GP[1]))
                # We start the length scale for intensity with
                # a small value. A heuristic is 2 percentile of
                # all the square differences. But it should not be
                # smaller than 0.5. This limit is set in case
                # many voxels have close to equal intensities,
                # which might render 2 percentile to 0.

        # Step 3 fitting. GP prior is imposed if requested.
        # In this step, unless auto_nuisance is set to False, X_res
        # will be re-estimated from the residuals after each step
        # of fitting. And X0 will be concatenation of X_base and X_res
        logger.debug('indexing:{}'.format(idx_param_fitV))
        logger.debug('initial GP parameters:{}'.format(current_GP))
        current_vec_U_chlsk_l, current_a1, current_logSNR2,\
            current_GP, X_res = self._fit_diagV_GP(
                XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
                XTX, XTDX, XTFX, X, Y, X_base, X_res, D, F, run_TRs,
                current_vec_U_chlsk_l,
                current_a1, current_logSNR2, current_GP, n_smooth,
                idx_param_fitU, idx_param_fitV,
                l_idx, n_C, n_T, n_V, n_l, n_run, n_X0, rank,
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
            X0TY, X0TDY, X0TFY, X0, X_base, n_X0, _ \
            = self._prepare_data_XYX0(
                X, Y, X_base, X_res, D, F, run_TRs, no_DC=True)

        X0TAX0, XTAX0, X0TAY, X0TAX0_i, \
            XTAcorrX, XTAcorrY, YTAcorrY, LTXTAcorrY, XTAcorrXL, LTXTAcorrXL\
            = self._precompute_ar1_quad_forms(XTY, XTDY, XTFY,
                                              YTY_diag, YTDY_diag, YTFY_diag,
                                              XTX, XTDX, XTFX, X0TX0, X0TDX0,
                                              X0TFX0, XTX0, XTDX0, XTFX0, X0TY,
                                              X0TDY, X0TFY,
                                              estU_chlsk_l_AR1_UV,
                                              est_rho1_AR1_UV, n_V, n_X0)
        LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, sigma2 \
            = self._calc_LL(est_rho1_AR1_UV, LTXTAcorrXL, LTXTAcorrY, YTAcorrY,
                            X0TAX0, est_SNR_AR1_UV**2,
                            n_V, n_T, n_run, rank, n_X0)
        est_sigma_AR1_UV = sigma2**0.5
        est_beta_AR1_UV = est_SNR_AR1_UV**2 \
            * np.dot(estU_chlsk_l_AR1_UV, YTAcorrXL_LAMBDA.T)
        est_beta_AR1_UV_latent =  \
            est_SNR_AR1_UV**2 * YTAcorrXL_LAMBDA.T
        # the latent term means that X*L multiplied by this term
        # is the same as X*beta. This will be used for decoding
        # and cross-validating, in case L is low-rank
        est_beta0_AR1_UV = np.einsum(
            'ijk,ki->ji', X0TAX0_i,
            (X0TAY - np.einsum('ikj,ki->ji', XTAX0, est_beta_AR1_UV)))

        # Now we want to collapse all beta0 corresponding to DC components
        # of different runs to a single map, and preserve only one DC component
        # across runs. This is because they should express the same component
        # and the new data to transform do not necessarily have the same
        # numbers of runs as the training data.

        if idx_DC.size > 1:
            collapsed_DC = np.sum(X0[:, idx_DC], axis=1)
            X0 = np.insert(np.delete(X0, idx_DC, axis=1), 0,
                           collapsed_DC, axis=1)
            collapsed_beta0 = np.mean(est_beta0_AR1_UV[idx_DC, :], axis=0)
            est_beta0_AR1_UV = np.insert(
                np.delete(est_beta0_AR1_UV, idx_DC, axis=0),
                0, collapsed_beta0, axis=0)
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
            else:
                est_intensity_kernel_r = None
                K_major = np.exp(- dist2 / est_space_smooth_r**2 / 2.0)
            K = K_major + np.diag(np.ones(n_V) * self.eta)
            invK_tilde_log_SNR = np.linalg.solve(K, current_logSNR2) / 2
            log_SNR_invK_tilde_log_SNR = np.dot(current_logSNR2,
                                                invK_tilde_log_SNR) / 2
            tau2, _ = self.tau2_prior(log_SNR_invK_tilde_log_SNR, n_V,
                                      self.tau_range)
            est_std_log_SNR = tau2 ** 0.5
        else:
            est_space_smooth_r = None
            est_intensity_kernel_r = None
            est_std_log_SNR = None
        return est_cov_AR1_UV, estU_chlsk_l_AR1_UV, est_SNR_AR1_UV, \
            est_beta_AR1_UV, est_beta0_AR1_UV, est_beta_AR1_UV_latent,\
            est_sigma_AR1_UV, est_rho1_AR1_UV, est_space_smooth_r, \
            est_std_log_SNR, est_intensity_kernel_r, X0

    def _transform(self, Y, scan_onsets, beta, beta0,
                   rho_e, sigma_e, rho_X, sigma2_X, rho_X0, sigma2_X0):
        """ Given the data Y and the response amplitudes beta and beta0
            estimated in the fit step, estimate the corresponding X and X0.
            It is done by a forward-backward algorithm.
            We assume X and X0 both are vector autoregressive (VAR)
            processes, to capture temporal smoothness. Their VAR
            parameters are estimated from training data at the fit stage.
        """
        logger.info('Transforming new data.')
        # Constructing the transition matrix and the variance of
        # innovation noise as prior for the latent variable X and X0
        # in new data.

        n_C = beta.shape[0]
        n_T = Y.shape[0]
        weight = np.concatenate((beta, beta0), axis=0)
        T_X = np.diag(np.concatenate((rho_X, rho_X0)))
        Var_X = np.concatenate((sigma2_X / (1 - rho_X**2),
                                sigma2_X0 / (1 - rho_X0**2)))
        Var_dX = np.concatenate((sigma2_X, sigma2_X0))
        sigma2_e = sigma_e ** 2
        scan_onsets = np.setdiff1d(scan_onsets, n_T)
        n_scan = scan_onsets.size
        X = [None] * scan_onsets.size
        X0 = [None] * scan_onsets.size
        total_log_p = 0
        for scan, onset in enumerate(scan_onsets):
            # Forward step
            if scan == n_scan - 1:
                offset = n_T
            else:
                offset = scan_onsets[scan + 1]
            mu, mu_Gamma_inv, Gamma_inv, log_p_data, Lambda_0, \
                Lambda_1, H, deltaY, deltaY_sigma2inv_rho_weightT = \
                self._forward_step(Y[onset:offset, :],
                                   T_X, Var_X, Var_dX, rho_e, sigma2_e,
                                   weight)
            total_log_p += log_p_data
            # Backward step
            mu_hat, mu_Gamma_inv_hat, Gamma_inv_hat \
                = self._backward_step(
                    deltaY, deltaY_sigma2inv_rho_weightT, sigma2_e,
                    weight, mu, mu_Gamma_inv, Gamma_inv,
                    Lambda_0, Lambda_1, H)
            X[scan] = np.concatenate(
                [mu_t[None, :n_C] for mu_t in mu_hat])
            X0[scan] = np.concatenate(
                [mu_t[None, n_C:] for mu_t in mu_hat])
        X = np.concatenate(X)
        X0 = np.concatenate(X0)
        return X, X0, total_log_p

    def _score(self, Y, design, beta, scan_onsets, beta0, rho_e, sigma_e,
               rho_X0, sigma2_X0):
        """ Given the data Y, and the spatial pattern beta0
            of nuisance time series, return the cross-validated score
            of the data Y given all parameters of the subject estimated
            during the first step.
            It is assumed that the user has design matrix built for the
            data Y. Both beta and beta0 are posterior expectation estimated
            from training data with the estimated covariance matrix U and
            SNR serving as prior. We marginalize X0 instead of fitting
            it in this function because this function is for the purpose
            of evaluating model no new data. We should avoid doing any
            additional fitting when performing cross-validation.
            The hypothetic response to the task will be subtracted, and
            the unknown nuisance activity which contributes to the data
            through beta0 will be marginalized.
        """
        logger.info('Estimating cross-validated score for new data.')
        n_T = Y.shape[0]
        if design is not None:
            Y = Y - np.dot(design, beta)
        # The function works for both full model and null model.
        # If design matrix is not provided, the whole data is
        # used as input for _forward_step. If design matrix is provided,
        # residual after subtracting design * beta is fed to _forward_step
        T_X = np.diag(rho_X0)
        Var_X = sigma2_X0 / (1 - rho_X0**2)
        Var_dX = sigma2_X0
        # Prior parmeters for X0: T_X is transitioning matrix, Var_X
        # is the marginal variance of the first time point. Var_dX is the
        # variance of the updating noise.
        sigma2_e = sigma_e ** 2
        # variance of voxel-specific updating noise component
        scan_onsets = np.setdiff1d(scan_onsets, n_T).astype(int)
        n_scan = scan_onsets.size
        total_log_p = 0
        for scan, onset in enumerate(scan_onsets):
            # Forward step
            if scan == n_scan - 1:
                offset = n_T
            else:
                offset = scan_onsets[scan + 1]
            _, _, _, log_p_data, _, _, _, _, _ = \
                self._forward_step(
                    Y[onset:offset, :], T_X, Var_X, Var_dX, rho_e, sigma2_e,
                    beta0)
            total_log_p += log_p_data
        return total_log_p

    def _est_AR1(self, x, same_para=False):
        """ Estimate the AR(1) parameters of input x.
            Each column of x is assumed as independent from other columns,
            and each column is treated as an AR(1) process.
            If same_para is set as True, then all columns of x
            are concatenated and a single set of AR(1) parameters
            is estimated. Strictly speaking the breaking point
            between each concatenated column should be considered.
            But for long time series, this is ignored.
        """
        if same_para:
            n_c = x.shape[1]
            x = np.reshape(x, x.size, order='F')
            rho, sigma2 = alg.AR_est_YW(x, 1)
            # We concatenate all the design matrix to estimate common AR(1)
            # parameters. This creates some bias because the end of one column
            # and the beginning of the next column of the design matrix are
            # treated as consecutive samples.
            rho = np.ones(n_c) * rho
            sigma2 = np.ones(n_c) * sigma2
        else:
            rho = np.zeros(np.shape(x)[1])
            sigma2 = np.zeros(np.shape(x)[1])
            for c in np.arange(np.shape(x)[1]):
                rho[c], sigma2[c] = alg.AR_est_YW(x[:, c], 1)
        return rho, sigma2

    def _forward_step(self, Y, T_X, Var_X, Var_dX, rho_e, sigma2_e, weight):
        """ forward step for HMM, assuming both the hidden state and noise
            have 1-step dependence on the previous value.
        """
        # We currently only implement diagonal form
        # of covariance matrix for Var_X, Var_dX and T_X, which means
        # each dimension of X is independent and their innovation noise
        # are also independent. Note that log_p_data takes this assumption.
        if Var_X.ndim == 1:
            inv_Var_X = np.diag(1 / Var_X)
            half_log_det_Var_X = np.sum(np.log(Var_X)) / 2.0
            Var_X = np.diag(Var_X)
            # the marginal variance of X
        else:
            half_log_det_Var_X = self._half_log_det(Var_X)
            inv_Var_X = np.linalg.inv(Var_X)
        if Var_dX.ndim == 1:
            inv_Var_dX = np.diag(1 / Var_dX)
            half_log_det_Var_dX = np.sum(np.log(Var_dX)) / 2.0
            Var_dX = np.diag(Var_dX)
            # the marginal variance of Delta X (the change of X from
            # previous time point)
        else:
            inv_Var_dX = np.linalg.inv(Var_dX)
            half_log_det_Var_dX = self._half_log_det(Var_dX)
        if T_X.ndim == 1:
            T_X = np.diag(T_X)
            # Transfer function of X: the expected mean of X at t+1
            # time point is T_x * X
        [n_T, n_V] = np.shape(Y)
        # numbers of time points and voxels
        mu = [None] * n_T
        # posterior mean of X, conditioned on all data up till the current
        # time point
        Gamma_inv = [None] * n_T
        # inverse of poterior Gamma.
        mu_Gamma_inv = [None] * n_T
        # mu * inv(Gamma)
        log_p_data = - np.log(np.pi * 2) * (n_T * n_V) / 2 \
            - half_log_det_Var_X - np.sum(np.log(sigma2_e)) * n_T / 2.0\
            + np.sum(np.log(1 - rho_e**2)) / 2.0 - half_log_det_Var_dX \
            * (n_T - 1)
        # This is the term to be incremented by c_n at each time step.
        # We first add all the fixed terms to it.

        # The following are a few fixed terms.
        Lambda_0 = np.dot(T_X, np.dot(inv_Var_dX, T_X.T)) \
            + np.dot(weight * rho_e**2 / sigma2_e, weight.T)
        H = np.dot(inv_Var_dX, T_X.T) + np.dot(weight * rho_e / sigma2_e,
                                               weight.T)
        Lambda_1 = inv_Var_dX + np.dot(weight / sigma2_e, weight.T)

        Gamma_inv[0] = inv_Var_X + np.dot(
            weight * (1 - rho_e**2) / sigma2_e, weight.T)
        # We might not need this and only use linalg.solve for related terms.
        mu_Gamma_inv[0] = np.dot(
            Y[0, :] * (1 - rho_e**2) / sigma2_e, weight.T)
        mu[0] = np.linalg.solve(Gamma_inv[0], mu_Gamma_inv[0])
        log_p_data -= 0.5 * np.sum(Y[0, :]**2 * (1 - rho_e**2) / sigma2_e)
        # This is the term added for the first time point.

        deltaY = Y[1:, :] - rho_e * Y[:-1, :]
        deltaY_sigma2inv_rho_weightT = np.dot(
            deltaY / sigma2_e * rho_e, weight.T)
        for t in np.arange(1, n_T):
            Gamma_tilde_inv = Lambda_0 + Gamma_inv[t - 1]
            tmp = np.linalg.solve(Gamma_tilde_inv, H.T)
            Gamma_inv[t] = Lambda_1 - np.dot(H, tmp)
            mu_Gamma_inv[t] = np.dot(deltaY[t - 1, :] / sigma2_e, weight.T) \
                + np.dot(mu_Gamma_inv[t - 1]
                         - deltaY_sigma2inv_rho_weightT[t - 1, :], tmp)
            mu[t] = np.linalg.solve(Gamma_inv[t], mu_Gamma_inv[t])
            tmp2 = mu_Gamma_inv[t - 1] - deltaY_sigma2inv_rho_weightT[t - 1, :]
            log_p_data += -self._half_log_det(Gamma_tilde_inv) \
                + np.dot(tmp2, np.linalg.solve(Gamma_tilde_inv, tmp2)) / 2.0
        log_p_data += -self._half_log_det(Gamma_inv[-1]) \
            + np.dot(mu_Gamma_inv[-1], mu[-1]) / 2.0 \
            - np.sum(deltaY**2 / sigma2_e) / 2.0
        return mu, mu_Gamma_inv, Gamma_inv, log_p_data, Lambda_0, \
            Lambda_1, H, deltaY, deltaY_sigma2inv_rho_weightT

    def _backward_step(self, deltaY, deltaY_sigma2inv_rho_weightT,
                       sigma2_e, weight, mu, mu_Gamma_inv, Gamma_inv,
                       Lambda_0, Lambda_1, H):
        """ backward step for HMM, assuming both the hidden state and noise
            have 1-step dependence on the previous value.
        """
        n_T = len(Gamma_inv)
        # All the terms with hat before are parameters of posterior
        # distributions of X conditioned on data from all time points,
        # whereas the ones without hat calculated by _forward_step
        # are mean and covariance of posterior of X conditioned on
        # data up to the time point.
        Gamma_inv_hat = [None] * n_T
        mu_Gamma_inv_hat = [None] * n_T
        mu_hat = [None] * n_T

        mu_hat[-1] = mu[-1].copy()
        mu_Gamma_inv_hat[-1] = mu_Gamma_inv[-1].copy()
        Gamma_inv_hat[-1] = Gamma_inv[-1].copy()

        for t in np.arange(n_T - 2, -1, -1):
            tmp = np.linalg.solve(Gamma_inv_hat[t + 1] - Gamma_inv[t + 1]
                                  + Lambda_1, H)
            Gamma_inv_hat[t] = Gamma_inv[t] + Lambda_0 - np.dot(H.T, tmp)
            mu_Gamma_inv_hat[t] = mu_Gamma_inv[t] \
                - deltaY_sigma2inv_rho_weightT[t, :] + np.dot(
                    mu_Gamma_inv_hat[t + 1] - mu_Gamma_inv[t + 1]
                    + np.dot(deltaY[t, :] / sigma2_e, weight.T), tmp)
            mu_hat[t] = np.linalg.solve(Gamma_inv_hat[t],
                                        mu_Gamma_inv_hat[t])
        return mu_hat, mu_Gamma_inv_hat, Gamma_inv_hat

    def _initial_fit_singpara(self, XTX, XTDX, XTFX,
                              YTY_diag, YTDY_diag, YTFY_diag,
                              XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                              XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                              X, Y, X0, idx_param_sing, l_idx,
                              n_C, n_T, n_V, n_l, n_run, n_X0, rank):
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

        cov_point_est = np.cov(beta_hat[n_X0:, :]) / np.var(residual)
        current_vec_U_chlsk_l = \
            np.linalg.cholesky((cov_point_est + np.eye(n_C)) / 2)[l_idx]

        # We use the average of covariance of point estimation and an identity
        # matrix as the initial value of the covariance matrix, just in case
        # the user provides data in which n_V is smaller than n_C.

        # (2) start from identity matrix

        # current_vec_U_chlsk_l = np.eye(n_C)[l_idx]

        # (3) random initialization

        # current_vec_U_chlsk_l = self.random_state_.randn(n_l)

        # vectorized version of L, Cholesky factor of U, the shared
        # covariance matrix of betas across voxels.

        rho1 = np.sum(
            residual[0:-1, :] * residual[1:, :], axis=0) / \
            np.sum(residual[0:-1, :] * residual[0:-1, :], axis=0)
        # Estimate of auto correlation assuming data includes pure noise.
        log_sigma2 = np.log(np.var(
            residual[1:, :] - residual[0:-1, :] * rho1, axis=0))
        # log of estimates of the variance of the "innovation" noise
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
                  l_idx, n_C, n_T, n_V, n_run, n_X0,
                  idx_param_sing, rank),
            method=self.optimizer, jac=True, tol=self.tol,
            options={'disp': self.minimize_options['disp'],
                     'maxiter': 100})
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
            XTX, XTDX, XTFX, X, Y, X_base, X_res, D, F, run_TRs,
            current_vec_U_chlsk_l,
            current_a1, current_logSNR2,
            idx_param_fitU, idx_param_fitV,
            l_idx, n_C, n_T, n_V, n_l, n_run, n_X0, rank):
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
                X0TY, X0TDY, X0TFY, X0, X_base, n_X0, _ \
                = self._prepare_data_XYX0(
                    X, Y, X_base, X_res, D, F, run_TRs, no_DC=True)

            # fit U, the covariance matrix, together with AR(1) param
            param0_fitU[idx_param_fitU['Cholesky']] = \
                current_vec_U_chlsk_l \
                + self.random_state_.randn(n_l) \
                * np.linalg.norm(current_vec_U_chlsk_l) \
                / n_l**0.5 * np.exp(-it / init_iter * self.anneal_speed - 1)
            param0_fitU[idx_param_fitU['a1']] = current_a1
            res_fitU = scipy.optimize.minimize(
                self._loglike_AR1_diagV_fitU, param0_fitU,
                args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                      XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                      XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                      current_logSNR2, l_idx, n_C,
                      n_T, n_V, n_run, n_X0, idx_param_fitU, rank),
                method=self.optimizer, jac=True, tol=tol,
                options=self.minimize_options)
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
                self._precompute_ar1_quad_forms(XTY, XTDY, XTFY,
                                                YTY_diag, YTDY_diag, YTFY_diag,
                                                XTX, XTDX, XTFX,
                                                X0TX0, X0TDX0, X0TFX0,
                                                XTX0, XTDX0, XTFX0,
                                                X0TY, X0TDY, X0TFY,
                                                L, rho1, n_V, n_X0)
            res_fitV = scipy.optimize.minimize(
                self._loglike_AR1_diagV_fitV, param0_fitV,
                args=(X0TAX0, XTAX0, X0TAY,
                      X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY,
                      LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                      current_vec_U_chlsk_l,
                      current_a1, l_idx, n_C, n_T, n_V, n_run,
                      n_X0, idx_param_fitV, rank,
                      False, False),
                method=self.optimizer, jac=True, tol=tol,
                options=self.minimize_options)

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

            # Re-estimating X_res from residuals
            current_SNR2 = np.exp(current_logSNR2)
            if self.auto_nuisance:
                LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, current_sigma2 \
                    = self._calc_LL(rho1, LTXTAcorrXL, LTXTAcorrY, YTAcorrY,
                                    X0TAX0, current_SNR2,
                                    n_V, n_T, n_run, rank, n_X0)
                betas = current_SNR2 * np.dot(L, YTAcorrXL_LAMBDA.T)
                beta0s = np.einsum(
                    'ijk,ki->ji', X0TAX0_i,
                    (X0TAY - np.einsum('ikj,ki->ji', XTAX0, betas)))
                residuals = Y - np.dot(X, betas) - np.dot(
                    X_base, beta0s[:np.shape(X_base)[1], :])
                X_res = self.nureg_method(
                    self.n_nureg_).fit_transform(
                    self.preprocess_residual(residuals))
            if norm_fitVchange / np.sqrt(param0_fitV.size) < tol \
                    and norm_fitUchange / np.sqrt(param0_fitU.size) \
                    < tol:
                break
        return current_vec_U_chlsk_l, current_a1, current_logSNR2, X_res

    def _fit_diagV_GP(
            self, XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
            XTX, XTDX, XTFX, X, Y, X_base, X_res, D, F, run_TRs,
            current_vec_U_chlsk_l,
            current_a1, current_logSNR2, current_GP, n_smooth,
            idx_param_fitU, idx_param_fitV, l_idx,
            n_C, n_T, n_V, n_l, n_run, n_X0, rank, GP_space, GP_inten,
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
                X0TY, X0TDY, X0TFY, X0, X_base, n_X0, _ = \
                self._prepare_data_XYX0(
                    X, Y, X_base, X_res, D, F, run_TRs, no_DC=True)

            # fit U

            param0_fitU[idx_param_fitU['Cholesky']] = \
                current_vec_U_chlsk_l \
                + self.random_state_.randn(n_l) \
                * np.linalg.norm(current_vec_U_chlsk_l) \
                / n_l**0.5 * np.exp(-it / n_iter * self.anneal_speed - 1)
            param0_fitU[idx_param_fitU['a1']] = current_a1

            res_fitU = scipy.optimize.minimize(
                self._loglike_AR1_diagV_fitU, param0_fitU,
                args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                      XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                      XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                      current_logSNR2, l_idx, n_C, n_T, n_V,
                      n_run, n_X0, idx_param_fitU, rank),
                method=self.optimizer, jac=True,
                tol=tol,
                options=self.minimize_options)
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
                self._precompute_ar1_quad_forms(XTY, XTDY, XTFY,
                                                YTY_diag, YTDY_diag, YTFY_diag,
                                                XTX, XTDX, XTFX,
                                                X0TX0, X0TDX0, X0TFX0,
                                                XTX0, XTDX0, XTFX0,
                                                X0TY, X0TDY, X0TFY,
                                                L, rho1, n_V, n_X0)
            res_fitV = scipy.optimize.minimize(
                self._loglike_AR1_diagV_fitV, param0_fitV, args=(
                    X0TAX0, XTAX0, X0TAY, X0TAX0_i,
                    XTAcorrX, XTAcorrY, YTAcorrY,
                    LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                    current_vec_U_chlsk_l, current_a1,
                    l_idx, n_C, n_T, n_V, n_run, n_X0,
                    idx_param_fitV, rank,
                    GP_space, GP_inten, dist2, inten_diff2,
                    space_smooth_range, inten_smooth_range),
                method=self.optimizer, jac=True,
                tol=tol,
                options=self.minimize_options)

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

            # Re-estimating X_res from residuals
            current_SNR2 = np.exp(current_logSNR2)
            if self.auto_nuisance:
                LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, current_sigma2 \
                    = self._calc_LL(rho1, LTXTAcorrXL, LTXTAcorrY, YTAcorrY,
                                    X0TAX0, current_SNR2,
                                    n_V, n_T, n_run, rank, n_X0)
                betas = current_SNR2 \
                    * np.dot(L, YTAcorrXL_LAMBDA.T)
                beta0s = np.einsum(
                    'ijk,ki->ji', X0TAX0_i,
                    (X0TAY - np.einsum('ikj,ki->ji', XTAX0, betas)))
                residuals = Y - np.dot(X, betas) - np.dot(
                    X_base, beta0s[:np.shape(X_base)[1], :])
                X_res = self.nureg_method(self.n_nureg_).fit_transform(
                    self.preprocess_residual(residuals))
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
            current_GP, X_res

    def _fit_null(self, Y, X_base, scan_onsets=None):
        """ Fit a null model.
        """
        n_V = np.size(Y, axis=1)
        n_T = np.size(Y, axis=0)
        t_start = time.time()

        D, F, run_TRs, n_run = self._prepare_DF(
            n_T, scan_onsets=scan_onsets)
        YTY_diag = np.sum(Y * Y, axis=0)
        YTDY_diag = np.sum(Y * np.dot(D, Y), axis=0)
        YTFY_diag = np.sum(Y * np.dot(F, Y), axis=0)

        tol = self.tol
        n_iter = self.n_iter
        logger.info('Fitting null model'
                    ' for maximum {} times'.format(n_iter))

        # Add DC components capturing run-specific baselines.
        X_DC = self._gen_X_DC(run_TRs)
        X_DC, X_base, idx_DC = self._merge_DC_to_base(
            X_DC, X_base, no_DC=False)
        X_res = None
        param0 = np.zeros(n_V)
        for it in range(0, n_iter):
            if X_res is None:
                X0 = X_base
            else:
                X0 = np.concatenate((X_base, X_res), axis=1)
            n_X0 = X0.shape[1]
            X0TX0, X0TDX0, X0TFX0 = self._make_templates(D, F, X0, X0)
            X0TY, X0TDY, X0TFY = self._make_templates(D, F, X0, Y)
            res_null = scipy.optimize.minimize(
                self._loglike_AR1_null, param0, args=(
                    YTY_diag, YTDY_diag, YTFY_diag,
                    X0TX0, X0TDX0, X0TFX0, X0TY, X0TDY, X0TFY,
                    n_T, n_V, n_run, n_X0),
                method=self.optimizer, jac=True, tol=tol,
                options=self.minimize_options)
            param_change = res_null.x - param0
            param0 = res_null.x.copy()
            est_rho1_AR1_null = 2.0 / np.pi * np.arctan(param0)
            if self.auto_nuisance:
                X0TAX0 = X0TX0[None, :, :] \
                    - est_rho1_AR1_null[:, None, None] \
                    * X0TDX0[None, :, :] \
                    + est_rho1_AR1_null[:, None, None]**2 \
                    * X0TFX0[None, :, :]
                # dimension: space*#baseline*#baseline
                X0TAY = self._make_ar1_quad_form(X0TY, X0TDY, X0TFY,
                                                 est_rho1_AR1_null)
                # dimension: #baseline*space
                beta0s = np.linalg.solve(X0TAX0, X0TAY.T).T
                residuals = Y - np.dot(X_base, beta0s[:np.shape(X_base)[1], :])
                X_res = self.nureg_method(self.n_nureg_).fit_transform(
                    self.preprocess_residual(residuals))
            if np.max(np.abs(param_change)) < self.tol:
                logger.info('The change of parameters is smaller than '
                            'the tolerance value {}. Fitting is finished '
                            'after {} iterations'.format(self.tol, it + 1))
                break
        X0TAX0 = X0TX0[None, :, :] \
            - est_rho1_AR1_null[:, None, None] \
            * X0TDX0[None, :, :] \
            + est_rho1_AR1_null[:, None, None]**2 \
            * X0TFX0[None, :, :]
        # dimension: space*#baseline*#baseline
        X0TAY = self._make_ar1_quad_form(X0TY, X0TDY, X0TFY,
                                         est_rho1_AR1_null)
        # dimension: #baseline*space
        est_beta0_AR1_null = np.linalg.solve(X0TAX0, X0TAY.T).T

        YTAY = self._make_ar1_quad_form(YTY_diag, YTDY_diag, YTFY_diag,
                                        est_rho1_AR1_null)
        # dimension: space,
        YTAcorrY = YTAY - np.sum(X0TAY * est_beta0_AR1_null, axis=0)
        # dimension: space,
        est_sigma_AR1_null = (YTAcorrY / (n_T - n_X0)) ** 0.5
        if idx_DC.size > 1:
            collapsed_DC = np.sum(X0[:, idx_DC], axis=1)
            X0 = np.insert(np.delete(X0, idx_DC, axis=1), 0,
                           collapsed_DC, axis=1)
            collapsed_beta0 = np.mean(est_beta0_AR1_null[idx_DC, :], axis=0)
            est_beta0_AR1_null = np.insert(
                np.delete(est_beta0_AR1_null, idx_DC, axis=0),
                0, collapsed_beta0, axis=0)
        t_finish = time.time()
        logger.info(
            'total time of fitting: {} seconds'.format(t_finish - t_start))
        return est_beta0_AR1_null, est_sigma_AR1_null, est_rho1_AR1_null, X0

    # We fit two parts of the parameters iteratively.
    # The following are the corresponding negative log likelihood functions.

    def _loglike_AR1_diagV_fitU(self, param, XTX, XTDX, XTFX, YTY_diag,
                                YTDY_diag, YTFY_diag, XTY, XTDY, XTFY,
                                X0TX0, X0TDX0, X0TFX0,
                                XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                log_SNR2, l_idx, n_C, n_T, n_V, n_run, n_X0,
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
            self._precompute_ar1_quad_forms(XTY, XTDY, XTFY,
                                            YTY_diag, YTDY_diag, YTFY_diag,
                                            XTX, XTDX, XTFX, X0TX0, X0TDX0,
                                            X0TFX0, XTX0, XTDX0, XTFX0,
                                            X0TY, X0TDY, X0TFY,
                                            L, rho1, n_V, n_X0)

        # Only starting from this point, SNR2 is involved

        LL, LAMBDA_i, LAMBDA, YTAcorrXL_LAMBDA, sigma2 \
            = self._calc_LL(rho1, LTXTAcorrXL, LTXTAcorrY, YTAcorrY,
                            X0TAX0, SNR2, n_V, n_T, n_run, rank, n_X0)
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
        dXTAX_drho1 = -XTDX + 2 * rho1[:, None, None] * XTFX
        # dimension: space*feature*feature
        dXTAY_drho1 = self._make_ar1_quad_form_grad(XTDY, XTFY, rho1)
        # dimension: feature*space
        dYTAY_drho1 = self._make_ar1_quad_form_grad(YTDY_diag, YTFY_diag, rho1)
        # dimension: space,

        dX0TAX0_drho1 = - X0TDX0 \
            + 2 * rho1[:, None, None] * X0TFX0
        # dimension: space*rank*rank
        dXTAX0_drho1 = - XTDX0 \
            + 2 * rho1[:, None, None] * XTFX0
        # dimension: space*feature*rank
        dX0TAY_drho1 = self._make_ar1_quad_form_grad(X0TDY, X0TFY, rho1)
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
                                n_X0, idx_param_fitV, rank=None,
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
                            SNR2, n_V, n_T, n_run, rank, n_X0)
        # Log likelihood of data given parameters, without the GP prior.
        deriv_log_SNR2 = (-rank + np.trace(LAMBDA, axis1=1, axis2=2)) * 0.5\
            + np.sum(YTAcorrXL_LAMBDA**2, axis=1) * SNR2 / sigma2 / 2
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

            # MAP estimate of the variance of the Gaussian Process given
            # other parameters.
            tau2, log_ptau = self.tau2_prior(log_SNR_invK_tilde_log_SNR, n_V,
                                             self.tau_range)
            # log_ptau is log(p(tau)) given the form of prior for tau
            LL += log_ptau

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
                              l_idx, n_C, n_T, n_V, n_run, n_X0,
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
            / (n_T - n_X0)
        LL = n_V * (-np.log(sigma2) * (n_T - n_X0) * 0.5
                    + np.log(1 - rho1**2) * n_run * 0.5
                    - self._half_log_det(X0TAX0)
                    - self._half_log_det(LAMBDA_i))

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

    def _loglike_AR1_null(self, param, YTY_diag, YTDY_diag, YTFY_diag,
                          X0TX0, X0TDX0, X0TFX0, X0TY, X0TDY, X0TFY,
                          n_T, n_V, n_run, n_X0):
        # This function calculates the log likelihood of data given AR(1)
        # parameters of noise as free parameters.
        # Free parameters are in param.
        # It serves as a null model which assumes no response to design
        # matrix.
        a1 = param
        rho1 = 2.0 / np.pi * np.arctan(a1)  # auto-regressive coefficients

        YTAY = self._make_ar1_quad_form(YTY_diag, YTDY_diag, YTFY_diag, rho1)
        # dimension: space,
        # A/sigma2 is the inverse of noise covariance matrix in each voxel.
        # YTAY means Y'AY
        X0TAX0 = X0TX0[None, :, :] - rho1[:, None, None] \
            * X0TDX0[None, :, :] \
            + rho1[:, None, None]**2 * X0TFX0[None, :, :]
        # dimension: space*#baseline*#baseline
        X0TAY = self._make_ar1_quad_form(X0TY, X0TDY, X0TFY, rho1)
        # dimension: #baseline*space
        # X0TAX0_i = np.linalg.solve(X0TAX0, np.identity(n_X0)[None, :, :])
        X0TAX0_i = np.linalg.inv(X0TAX0)
        # dimension: space*#baseline*#baseline
        YTAcorrY = YTAY - np.sum(X0TAY * np.einsum('ijk,ki->ji',
                                                   X0TAX0_i, X0TAY), axis=0)
        # dimension: space,
        sigma2 = YTAcorrY / (n_T - n_X0)
        # dimension: space,
        LL = - np.sum(np.log(sigma2)) * (n_T - n_X0) * 0.5 \
            + np.sum(np.log(1 - rho1**2)) * n_run * 0.5 \
            - np.sum(self._half_log_det(X0TAX0)) \
            - (n_T - n_X0) * n_V * (1 + np.log(2 * np.pi)) * 0.5
        # The following are for calculating the derivative to a1
        deriv_a1 = np.empty(n_V)
        dYTAY_drho1 = self._make_ar1_quad_form_grad(YTDY_diag, YTFY_diag, rho1)
        # dimension: space,
        dX0TAX0_drho1 = - X0TDX0 \
            + 2 * rho1[:, None, None] * X0TFX0
        # dimension: space*rank*rank
        dX0TAY_drho1 = self._make_ar1_quad_form_grad(X0TDY, X0TFY, rho1)
        # dimension: rank*space

        # The following are executed for each voxel.
        for i_v in range(n_V):
            # All variables with _ele as suffix are for data of just one voxel
            invX0TAX0_X0TAY_ele = np.dot(X0TAX0_i[i_v, :, :], X0TAY[:, i_v])
            # preparation for the variable below
            dYTAcorrY_drho1_ele = dYTAY_drho1[i_v] \
                - np.dot(dX0TAY_drho1[:, i_v], invX0TAX0_X0TAY_ele) * 2\
                + np.dot(np.dot(invX0TAX0_X0TAY_ele, dX0TAX0_drho1[i_v, :, :]),
                         invX0TAX0_X0TAY_ele)
            deriv_a1[i_v] = 2 / np.pi / (1 + a1[i_v]**2) \
                * (- n_run * rho1[i_v] / (1 - rho1[i_v]**2)
                   - np.einsum('ij,ij', X0TAX0_i[i_v, :, :],
                               dX0TAX0_drho1[i_v, :, :]) * 0.5
                   - dYTAcorrY_drho1_ele * 0.5 / sigma2[i_v])
        deriv = deriv_a1

        return -LL, -deriv


class GBRSA(BRSA):
    """Group Bayesian representational Similarity Analysis (GBRSA)

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
    Both tools provide estimation of SNR and noise parameters at the end,
    and both tools provide empirical Bayesian estimates of activity patterns
    beta, together with weight map of nuisance signals beta0.

    The differences of this tool from BRSA are:
    (1) It allows fitting a shared covariance matrix (which can be converted
    to similarity matrix) across multiple subjects.
    This is analogous to SRM under funcalign submodule. Because of using
    multiple subjects, the result is less noisy.
    (2) In the fitting process, the SNR and noise parameters are marginalized
    for each voxel. Therefore, this tool should be faster than BRSA
    when analyzing an ROI of hundreds to thousands voxels. It does not
    provide a spatial smoothness prior on SNR though.
    (3) The voxel-wise pseudo-SNR and noise parameters estimated are
    posterior mean estimates, while those estimated by BRSA are
    maximum-a-posterior estimates.
    If your goal is to perform searchlight RSA with relatively fewer voxels
    on single subject, BRSA should be faster. However, GBRSA can in principle
    be used together with searchlight in a template space such as MNI.

    .. math::
        Y = X \\cdot \\beta + X_0 \\cdot \\beta_0 + \\epsilon

        \\beta_i \\sim N(0,(s_{i} \\sigma_{i})^2 U)

    See also `.BRSA`.

    Parameters
    ----------
    n_iter : int. Default: 50
        Number of maximum iterations to run the algorithm.
    rank : int. Default: None
        The rank of the covariance matrix.
        If not provided, the covariance matrix will be assumed
        to be full rank. When you have many conditions
        (e.g., calculating the similarity matrix of responses to each event),
        you might want to start with specifying a lower rank and use metrics
        such as AIC or BIC to decide the optimal rank. The log likelihood
        for the fitted data can be retrieved through private attributes
        _LL_train\_. Note that this log likelihood score is only used
        here for selecting hyperparameters such as rank. For any formal
        model comparison, we recommend using score() function on left-out
        data.
    auto_nuisance: boolean. Default: True
        In order to model spatial correlation between voxels that cannot
        be accounted for by common response captured in the design matrix,
        we assume that a set of time courses not related to the task
        conditions are shared across voxels with unknown amplitudes.
        One approach is for users to provide time series which they consider
        as nuisance but exist in the noise (such as head motion).
        The other way is to take the first n_nureg principal components
        in the residual after subtracting the response to the design matrix
        from the data, and use these components as the nuisance regressor.
        This flag is for the second approach. If turned on,
        PCA or factor analysis will be applied to the residuals
        to obtain new nuisance regressors in each round of fitting.
        These two approaches can be combined. If the users provide nuisance
        regressors and set this flag as True, then the first n_nureg
        principal components of the residuals after subtracting
        both the responses to design matrix and the user-supplied nuisance
        regressors will be used in addition to the nuisance regressors
        provided by the users.
        Note that nuisance regressor is not required from user. If it is
        not provided, DC components for each run will be included as nuisance
        regressor regardless of the auto_nuisance parameter.
    n_nureg: Optional[int]. Default: None
        Number of nuisance regressors to use in order to model signals
        shared across voxels not captured by the design matrix.
        This number is in addition to any nuisance regressor that the user
        has already provided.
        If set to None, the number of nuisance regressors will be
        automatically determined based on M Gavish
        and D Donoho's approximate estimation of optimal hard
        threshold for singular values. (Gavish & Donoho,
        IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.)
        This only takes effect if auto_nuisance is True.
    nureg_zscore: boolean, default: True
        A flag to tell the algorithm whether data is z-scored before
        estimating the number of nuisance regressor components necessary to
        account for spatial noise correlation. It also determinie whether
        the residual noise is z-scored before estimating the nuisance
        regressors from residual.
        This only takes effect if auto_nuisance is True.
    nureg_method: string, naming a method from sklearn.decomposition.
        'PCA', 'ICA', 'FA' or 'SPCA' are currently supported. Default: 'PCA'
        The method to estimate the shared component in noise across voxels.
        This only takes effect if auto_nuisance is True.
    baseline_single: boolean. Default: False
        A time course of constant 1 will be included to the nuisance
        regressor for each participant. If baseline_single is set to False,
        one such regressor is included for each fMRI run, but at the end of
        fitting, a single component in beta0\_ will be computed as the average
        of the weight maps corresponding to these regressors. This might
        cause underestimation of noise variance.
        If baseline_single is True, only one regressor of constant 1 will be
        used for the whole dataset. This might be desirable if you
        believe the average image intensity might not scale with the
        same proportion for different voxels across scan. In other words,
        it is possible that some part of the brain is more vulnerable to
        change in baseline intensity due to facts such as
        field inhomogeneity. Setting baseline_single to True will force the
        nuisance regressors automatically estimated from residuals to
        capture this. However, when each task condition only occurs in one
        run and when the design matrix in each run sums together close to
        a flat line, this option can cause the estimated similarity to be
        extremely high between conditions occuring in the same run.
    SNR_prior: string. Default: 'exp'
        The type of prior for pseudo-SNR.
        If set to 'exp', truncated exponential distribution with scale
        parameter of 1 is imposed on pseudo-SNR.
        If set to 'lognorm', a truncated log normal prior is imposed.
        In this case, the standard deviation of log(SNR) is set
        by the parameter logS_range.
        If set to 'unif', a uniform prior in [0,1] is imposed.
        In all these cases, SNR is numerically
        marginalized on a grid of parameters. So the parameter SNR_bins
        determines how accurate the numerical integration is. The more
        number of bins are used, the more accurate the numerical
        integration becomes.
        In all the cases, the grids used for pseudo-SNR do not really
        set an upper bound for SNR, because the real SNR is determined
        by both pseudo-SNR and U, the shared covariance structure.
    logS_range: float. Default: 1.0
        The reasonable range of the spread of SNR in log scale.
        This parameter only takes effect if SNR_prior is set to 'lognorm'.
        It is effectively the `s` parameter of `scipy.stats.lognorm`,
        or the standard deviation of the distribution in log scale.
        logS_range specifies how variable you believe the SNRs
        to vary across voxels in log scale.
        This range should not be set too large, otherwise the fitting
        may encounter numerical issue.
        If it is set too small, the estimated SNR will turn to be too
        close to each other and the estimated similarity matrix might
        overfit to voxels of low SNR.
        If you increase logS_range, it is recommended to increase
        SNR_bins accordingly, otherwise the pseudo-SNR values evaluated might
        be too sparse, causing the posterior pseudo-SNR estimations
        to be clustered around the bins.
    SNR_bins: integer. Default: 21
        The number of bins used to numerically marginalize the pseudo-SNR
        parameter. In general, you should try to choose a large number
        to the degree that decreasing SNR_bins does not change the result
        of fitting result. However, very large number of bins also causes
        slower computation and larger memory consumption.
        For SNR_prior='lognorm', the default value 21 is based on
        the default value of logS_range=1.0 and bin width of 0.3 on log scale.
        But it is also a reasonable choice for the other two options
        for SNR_prior.
    rho_bins: integer. Default: 20
        The number of bins to divide the region of (-1, 1) for rho.
        This only takes effect for fitting the marginalized version.
        If set to 20, discrete numbers of {-0.95, -0.85, ..., 0.95} will
        be used to numerically integrate rho from -1 to 1.
    optimizer: str or callable. Default: 'BFGS'
        The optimizer to use for minimizing cost function which
        scipy.optimize.minimize can accept.
        We use 'L-BFGS-B' as a default. Users can try other strings
        corresponding to optimizer provided by scipy.optimize.minimize,
        or a custom optimizer, such as 'L-BFGS-B' or 'CG'.
        Note that BRSA fits a lot of parameters. So a chosen optimizer
        should accept gradient (Jacobian) of the cost function. Otherwise
        the fitting is likely to be unbarely slow. We do not calculate
        Hessian of the objective function. So an optimizer which requires
        Hessian cannot be used.
    minimize_options: dictionary.
        Default: {'gtol': 1e-4, 'disp': False,
        'maxiter': 20}
        This is the dictionary passed as the options argument to
        scipy.optimize.minize which minimizes the cost function during
        fitting. Notice that the minimization is performed for up to
        n_iter times, with the nuisance regressor re-estimated each time.
        So within each of the n_iter steps of fitting,
        scipy.optimize.minize does not need to fully converge. The key
        'maxiter' in this dictionary determines the maximum number of
        iteration done by scipy.optimize.minimize within each of the n_iter
        steps of fitting.
    tol: float. Default: 1e-4.
        Tolerance parameter passed to scipy.optimize.minimize. It is also
        used for determining convergence of the alternating fitting
        procedure.
    random_state : RandomState or an int seed. Default: None
        A random number generator instance to define the state of
        the random permutations generator whenever the module
        needs to generate random number (e.g., initial parameter
        of the Cholesky factor).
    anneal_speed: float. Default: 10
        Annealing is introduced in fitting of the Cholesky
        decomposition of the shared covariance matrix. The amount
        of perturbation decays exponentially. This parameter sets
        the ratio of the maximum number of iteration to the
        time constant of the exponential.
        anneal_speed=10 means by n_iter/10 iterations,
        the amount of perturbation is reduced by 2.713 times.

    Attributes
    ----------
    U_ : numpy array, shape=[condition,condition].
        The shared covariance matrix
    L_ : numpy array, shape=[condition,condition].
        The Cholesky factor of the shared covariance matrix
        (lower-triangular matrix).
    C_: numpy array, shape=[condition,condition].
        The correlation matrix derived from the shared covariance matrix.
        This is the estimated similarity matrix between neural patterns
        to your task conditions. Notice that it is recommended that
        you also check U\_, which is the covariance matrix underlying
        this correlation matrix. In cases there is almost no response
        to your task conditions, the diagonal values of U\_ would become
        very small and C\_ might contain many correlation coefficients
        close to 1 or -1. This might not reflect true strong correlation
        or strong negative correlation, but a result of lack of
        task-related neural activity, design matrix that does not match
        true neural response, or not enough data.
        It is also recommended to check nSNR\_ after mapping it back to
        the brain. A "reasonable" map should at least have higher values
        in gray matter in than white matter.
    nSNR_ : list of numpy arrays, shape=[voxels,] for each subject in the list.
        The pseuso-SNR of all voxels. If SNR_prior='lognormal',
        the geometric mean of nSNR\_ would be approximately 1.
        If SNR_prior='unif', all nSNR\_ would be in the range of (0,1).
        If SNR_prior='exp' (default), the range of values would vary
        depending on the data and SNR_bins, but many should have low
        values with few voxels with high values.
        Note that this attribute can not be interpreted as true SNR,
        but the relative ratios between voxels indicate the contribution
        of each voxel to the representational similarity structure.
    sigma_ : list of numpy arrays, shape=[voxels,] for each subject.
        The estimated standard deviation of the noise in each voxel
        Assuming AR(1) model, this means the standard deviation
        of the innovation noise.
    rho_ : list of numpy arrays, shape=[voxels,] for each subject.
        The estimated autoregressive coefficient of each voxel
    beta_: list of numpy arrays, shape=[conditions, voxels] for each subject.
        The posterior mean estimation of the response amplitudes
        of each voxel to each task condition.
    beta0_: list of numpy arrays, shape=[n_nureg + n_base, voxels]
        for each subject.
        The loading weights of each voxel for the shared time courses
        not captured by the design matrix.
        n_base is the number of columns of the user-supplied nuisance
        regressors plus one for DC component.
    X0_: list of numpy arrays, shape=[time_points, n_nureg + n_base]
        for each subject.
        The estimated time course that is shared across voxels but
        unrelated to the events of interest (design matrix).
    beta0_null_: list of numpy arrays, shape=[n_nureg + n_base, voxels]
        for each subject.
        The equivalent of beta0\_ in a null model which does not
        include the design matrix and response pattern beta
    X0_null_: list of numpy arrays, shape=[time_points, n_nureg + n_base]
        for each subject.
        The equivalent of X0\_ in a null model which does not
        include the design matrix and response pattern beta
    n_nureg_: 1-d numpy array
        Number of nuisance regressor used to model the spatial noise
        correlation of each participant.
    random_state_: `RandomState`
        Random number generator initialized using random_state.
    """

    def __init__(
            self, n_iter=50, rank=None,
            auto_nuisance=True, n_nureg=None, nureg_zscore=True,
            nureg_method='PCA',
            baseline_single=False, logS_range=1.0, SNR_prior='exp',
            SNR_bins=21, rho_bins=20, tol=1e-4, optimizer='BFGS',
            minimize_options={'gtol': 1e-4, 'disp': False,
                              'maxiter': 20}, random_state=None,
            anneal_speed=10):

        self.n_iter = n_iter
        self.rank = rank
        self.auto_nuisance = auto_nuisance
        self.n_nureg = n_nureg
        self.nureg_zscore = nureg_zscore
        if auto_nuisance:
            assert (n_nureg is None) \
                or (isinstance(n_nureg, int) and n_nureg > 0), \
                'n_nureg should be a positive integer or None'\
                ' if auto_nuisance is True.'
        if self.nureg_zscore:
            self.preprocess_residual = lambda x: _zscore(x)
        else:
            self.preprocess_residual = lambda x: x
        if nureg_method == 'FA':
            self.nureg_method = lambda x: FactorAnalysis(n_components=x)
        elif nureg_method == 'PCA':
            self.nureg_method = lambda x: PCA(n_components=x, whiten=True)
        elif nureg_method == 'SPCA':
            self.nureg_method = lambda x: SparsePCA(n_components=x,
                                                    max_iter=20, tol=tol)
        elif nureg_method == 'ICA':
            self.nureg_method = lambda x: FastICA(n_components=x,
                                                  whiten=True)
        else:
            raise ValueError('nureg_method can only be FA, PCA, '
                             'SPCA(for sparse PCA) or ICA')
        self.baseline_single = baseline_single
        if type(logS_range) is int:
            logS_range = float(logS_range)
        self.logS_range = logS_range
        assert SNR_prior in ['unif', 'lognorm', 'exp'], \
            'SNR_prior can only be chosen from ''unif'', ''lognorm''' \
            ' and ''exp'''
        self.SNR_prior = SNR_prior
        self.SNR_bins = SNR_bins
        self.rho_bins = rho_bins
        self.tol = tol
        self.optimizer = optimizer
        self.minimize_options = minimize_options
        self.random_state = random_state
        self.anneal_speed = anneal_speed
        return

    def fit(self, X, design, nuisance=None, scan_onsets=None):
        """ Fit the model to data of all participants jointly.

        Parameters
        ----------
        X: list of numpy arrays, shape=[time_points, voxels] for each entry.
            Data to be fitted. Each participant corresponds to one item in
            the list. If you have multiple scans of the same participants
            that you want to analyze together, you should concatenate them
            along the time dimension after proper preprocessing (e.g. spatial
            alignment), and specify the onsets of each scan in scan_onsets.
        design: list of numpy arrays, shape=[time_points, conditions] for each.
            This is the design matrix of each participant.
            It should only include the hypothetic response for task conditions.
            You should not include regressors for a DC component or
            motion parameters, unless with a strong reason.
            If you want to model head motion, you should include them
            in nuisance regressors.
            If you have multiple run, the design matrix
            of all runs should be concatenated along the time dimension for
            each participant, with every column for one condition across runs.
            If the design matrix is the same for all subjects,
            either provide a list as required, or provide single numpy array.
        nuisance: optional, list of numpy arrays,
            shape=[time_points, nuisance_factors] for each subject in the list.
            Nuisance regressors of each participant.
            The responses to these regressors will be marginalized out from
            each voxel, which means they are considered, but won't be assumed
            to share the same pseudo-SNR map with the design matrix.
            Therefore, the pseudo-SNR map will only reflect the
            relative contribution of design matrix to each voxel.
            You can provide time courses such as those for head motion
            to this parameter.
            Note that if auto_nuisance is set to True, the first
            n_nureg principal components of residual (excluding the response
            to the design matrix and the user-provided nuisance regressors)
            will be included as additional nuisance regressor after the
            first round of fitting.
            If auto_nuisance is set to False, the nuisance regressors supplied
            by the users together with DC components will be used as
            nuisance time series.
        scan_onsets: optional, list numpy arrays, shape=[runs,] for each.
            Each item in the list specifies the indices of X which correspond
            to the onset of each scanning run for one participant.
            For example, if you have two experimental runs of
            the first participant, each with 100 TRs, and one run of the
            second participant, with 150 TR, then scan_onsets should be
            [ndarry([0, 100]), ndarry([150])].
            The effect of this argument is to make the inverse matrix
            of the temporal covariance matrix of noise block-diagonal.
            If you do not provide the argument, the program will
            assume all data are from the same run for each participant.
        """

        logger.info('Running Group Bayesian RSA (which can also analyze'
                    ' data of a single participant). Voxel-specific parameters'
                    'are all marginalized.')
        self.random_state_ = check_random_state(self.random_state)
        # setting random seed
        logger.debug('RandState set to {}'.format(self.random_state_))
        # Checking all inputs.
        X = self._check_data_GBRSA(X)
        design = self._check_design_GBRSA(design, X)
        nuisance = self._check_nuisance_GBRSA(
            copy.deepcopy(nuisance), X)
        # The reason that we use copy of nuisance is because they
        # may be modified inside our code.

        scan_onsets = self._check_scan_onsets_GBRSA(scan_onsets, X)
        # Run Marginalized Bayesian RSA
        # Note that we have a change of notation here.
        # Within _fit_RSA_marginalized, design matrix is named X
        # and data is named Y, to reflect the
        # generative model that data Y is generated by mixing the response
        # X to experiment conditions and other neural activity.
        # However, in fit(), we keep the scikit-learn API that
        # X is the input data to fit and y, a reserved name not used, is
        # the label to map to from X.
        assert self.SNR_bins >= 10 and self.rho_bins >= 10, \
            'At least 10 bins are required to perform the numerical'\
            ' integration over SNR and rho'
        assert self.logS_range * 6 / self.SNR_bins < 0.5 \
            or self.SNR_prior != 'lognorm', \
            'The minimum grid of log(SNR) should not be larger than 0.5 '\
            'if log normal prior is chosen for SNR.' \
            ' Please consider increasing SNR_bins or reducing logS_range'
        self.n_subj_ = len(X)
        self.n_V_ = [None] * self.n_subj_
        for subj, x in enumerate(X):
            self.n_V_[subj] = x.shape[1]
        if self.auto_nuisance:
            if self.n_nureg is None:
                logger.info('numbers of nuisance regressors are determined '
                            'automatically.')
                n_runs = np.zeros(self.n_subj_)
                n_comps = np.ones(self.n_subj_)
                for s_id in np.arange(self.n_subj_):
                    # For each subject, determine the number of nuisance
                    # regressors needed to account for the covariance
                    # in residuals.
                    # Residual is calculated by regrssing
                    # out the design matrix and DC component and linear trend
                    # from data of each run.
                    run_TRs, n_runs[s_id] = self._run_TR_from_scan_onsets(
                        X[s_id].shape[0], scan_onsets[s_id])
                    ts_dc = self._gen_legendre(run_TRs, [0])
                    _, ts_base, _ = self._merge_DC_to_base(
                        ts_dc, nuisance[s_id], False)
                    ts_reg = np.concatenate((ts_base, design[s_id]), axis=1)
                    beta_hat = np.linalg.lstsq(ts_reg, X[s_id])[0]
                    residuals = X[s_id] - np.dot(ts_reg, beta_hat)

                    n_comps[s_id] = np.min(
                        [np.max([Ncomp_SVHT_MG_DLD_approx(
                            residuals, self.nureg_zscore), 1]),
                         np.linalg.matrix_rank(residuals) - 1])
                    # n_nureg_ should not exceed the rank of
                    # residual minus 1.
                self.n_nureg_ = n_comps
                logger.info('Use {} nuisance regressors to model the spatial '
                            'correlation in noise.'.format(self.n_nureg_))
            else:
                self.n_nureg_ = self.n_nureg * np.ones(self.n_subj_)
            self.n_nureg_ = np.int32(self.n_nureg_)

        self.beta0_null_, self.sigma_null_, self.rho_null_, self.X0_null_,\
            self._LL_null_train_ = self._fit_RSA_marginalized_null(
                Y=X, X_base=nuisance, scan_onsets=scan_onsets)
        self.U_, self.L_, self.nSNR_, self.beta_, self.beta0_,\
            self.sigma_, self.rho_, self.X0_, self._LL_train_ = \
            self._fit_RSA_marginalized(
                X=design, Y=X, X_base=nuisance,
                scan_onsets=scan_onsets)

        self.C_ = utils.cov2corr(self.U_)
        self.design_ = design.copy()

        self._rho_design_ = [None] * self.n_subj_
        self._sigma2_design_ = [None] * self.n_subj_
        self._rho_X0_ = [None] * self.n_subj_
        self._sigma2_X0_ = [None] * self.n_subj_
        self._rho_X0_null_ = [None] * self.n_subj_
        self._sigma2_X0_null_ = [None] * self.n_subj_
        for subj in np.arange(self.n_subj_):
            self._rho_design_[subj], self._sigma2_design_[subj] = \
                self._est_AR1(self.design_[subj], same_para=True)
            self._rho_X0_[subj], self._sigma2_X0_[subj] = \
                self._est_AR1(self.X0_[subj])
            self._rho_X0_null_[subj], self._sigma2_X0_null_[subj] =\
                self._est_AR1(self.X0_null_[subj])
        # AR(1) parameters of the design matrix and nuisance regressors,
        # which will be used in transform or score.

        return self

    def transform(self, X, y=None, scan_onsets=None):
        """ Use the model to estimate the time course of response to
            each condition (ts), and the time course unrelated to task
            (ts0) which is spread across the brain.
            This is equivalent to "decoding" the design matrix and
            nuisance regressors from a new dataset different from the
            training dataset on which fit() was applied. An AR(1) smooth
            prior is imposed on the decoded ts and ts0 with the AR(1)
            parameters learnt from the corresponding time courses in the
            training data.

        Parameters
        ----------
        X : list of 2-D arrays. For each item, shape=[time_points, voxels]
            New fMRI data of the same subjects. The voxels should
            match those used in the fit() function.
            The size of the list should match the size of the list X fed
            to fit(), with each item in the list corresponding to data
            from the same subject in the X fed to fit(). If you do not
            need to transform some subjects' data, leave the entry
            corresponding to that subject as None.
            If data are z-scored when fitting the model,
            data should be z-scored as well when calling transform()
        y : not used (as it is unsupervised learning)
        scan_onsets : list of 1-D numpy arrays,
            Each array corresponds to the onsets of
            scans in the data X for the particular subject.
            If not provided, data will be assumed
            to be acquired in a continuous scan.

        Returns
        -------
        ts : list of 2-D arrays. For each, shape = [time_points, condition]
            The estimated response to the cognitive dimensions
            （task dimensions) whose response amplitudes were estimated
            during the fit step.
            One item for each subject. If some subjects' data are
            not provided, None will be returned.
        ts0: list of 2-D array. For each, shape = [time_points, n_nureg]
            The estimated time courses spread across the brain, with the
            loading weights estimated during the fit step.
            One item for each subject. If some subjects' data are
            not provided, None will be returned.
        """
        X = self._check_data_GBRSA(X, for_fit=False)
        scan_onsets = self._check_scan_onsets_GBRSA(scan_onsets, X)
        assert len(X) == self.n_subj_
        ts = [None] * self.n_subj_
        ts0 = [None] * self.n_subj_
        log_p = [None] * self.n_subj_
        for i, x in enumerate(X):
            if x is not None:
                s = scan_onsets[i]
                ts[i], ts0[i], log_p[i] = self._transform(
                    Y=x, scan_onsets=s, beta=self.beta_[i],
                    beta0=self.beta0_[i], rho_e=self.rho_[i],
                    sigma_e=self.sigma_[i], rho_X=self._rho_design_[i],
                    sigma2_X=self._sigma2_design_[i],
                    rho_X0=self._rho_X0_[i], sigma2_X0=self._sigma2_X0_[i])
        return ts, ts0

    def score(self, X, design, scan_onsets=None):
        """ After fit() is applied to the data of a group of participants,
            use the parameters estimated by fit() function to evaluate
            from some data of a set of participants to evaluate
            the log likelihood of some new data of the same participants
            given these estimated parameters.
            Design matrices of the same set of experimental
            conditions in the testing data should be provided, with each
            column corresponding to the same condition as that column
            in the design matrix of the training data.
            Unknown nuisance time series will be marginalized, assuming
            they follow the same spatial pattern as in the training
            data. The hypothetical response captured by the design matrix
            will be subtracted from data before the marginalization
            when evaluating the log likelihood. For null model,
            nothing will be subtracted before marginalization.

            There is a difference between the form of likelihood function
            used in fit() and score(). In fit(), the response amplitude
            beta to design matrix X and the modulation beta0 by nuisance
            regressor X0 are both marginalized, with X provided and X0
            estimated from data. In score(), posterior estimation of
            beta and beta0 from the fitting step are assumed unchanged
            in testing data; X is assumed given by the user,
            and X0 is marginalized.
            The logic underlying score() is to transfer
            as much as what we can learn from training data when
            calculating a likelihood score for testing data. This is done
            at the cost of using point estimation for beta and beta0.

            If you z-scored your data during fit step, you should
            z-score them for score function as well. If you did not
            z-score in fitting, you should not z-score here either.

        Parameters
        ----------
        X : List of 2-D arrays. For each item, shape=[time_points, voxels]
            fMRI data of new data of the same participants.
            The voxels of each participants should
            match those used in the fit() function. If data are z-scored
            (recommended) when fitting the model, data should be z-scored
            as well when calling transform()
        design : List of 2-D arrays. shape=[time_points, conditions] for each
            Each corresponds to one participant.
            Design matrices expressing the hypothetical response of
            the task conditions in data X.
        scan_onsets : List of 2-D arrays, shape=[#fMRI runs] for each
            Each array corresponds to one participant.
            Lists of indices corresponding to the onsets of
            scans in the data X.
            If not provided, data will be assumed
            to be acquired in a continuous scan.

        Returns
        -------
        ll: list, shape=[number of participants]
            The log likelihoods of the new data based on the model and its
            parameters fit to the training data.
            If data of some participants are not provided, the corresponding
            entry will be None.
        ll_null: list, shape=[number of participants]
            The log likelihood of the new data based on a null model
            which assumes the same as the full model for everything
            except for that there is no response to any of the
            task conditions.
        """
        X = self._check_data_GBRSA(X, for_fit=False)
        scan_onsets = self._check_scan_onsets_GBRSA(scan_onsets, X)
        design = self._check_design_GBRSA(design, X)
        assert len(X) == self.n_subj_
        ll = [None] * self.n_subj_
        ll_null = [None] * self.n_subj_
        for subj in np.arange(self.n_subj_):
            if X[subj] is not None:
                ll[subj] = self._score(
                    Y=X[subj], design=design[subj], beta=self.beta_[subj],
                    scan_onsets=scan_onsets[subj], beta0=self.beta0_[subj],
                    rho_e=self.rho_[subj], sigma_e=self.sigma_[subj],
                    rho_X0=self._rho_X0_[subj],
                    sigma2_X0=self._sigma2_X0_[subj])
                ll_null[subj] = self._score(
                    Y=X[subj], design=None, beta=None,
                    scan_onsets=scan_onsets[subj], beta0=self.beta0_[subj],
                    rho_e=self.rho_[subj], sigma_e=self.sigma_[subj],
                    rho_X0=self._rho_X0_[subj],
                    sigma2_X0=self._sigma2_X0_[subj])
        return ll, ll_null

    def _precompute_ar1_quad_forms_marginalized(
            self, XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag,
            XTX, XTDX, XTFX, X0TX0, X0TDX0, X0TFX0,
            XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
            rho1, n_V, n_X0):
        # Calculate the sandwich terms which put Acorr between X, Y and X0
        # These terms are used a lot in the likelihood. This function
        # is used for the marginalized version.
        XTAY = XTY - rho1[:, None, None] * XTDY \
            + rho1[:, None, None]**2 * XTFY
        # dimension: #rho*feature*space
        YTAY_diag = YTY_diag - rho1[:, None] * YTDY_diag \
            + rho1[:, None]**2 * YTFY_diag
        # dimension: #rho*space,
        # A/sigma2 is the inverse of noise covariance matrix in each voxel.
        # YTAY means Y'AY
        XTAX = XTX - rho1[:, None, None] * XTDX \
            + rho1[:, None, None]**2 * XTFX
        # dimension: n_rho*feature*feature
        X0TAX0 = X0TX0[None, :, :] - rho1[:, None, None] \
            * X0TDX0[None, :, :] \
            + rho1[:, None, None]**2 * X0TFX0[None, :, :]
        # dimension: #rho*#baseline*#baseline
        XTAX0 = XTX0[None, :, :] - rho1[:, None, None] \
            * XTDX0[None, :, :] \
            + rho1[:, None, None]**2 * XTFX0[None, :, :]
        # dimension: n_rho*feature*#baseline
        X0TAY = X0TY - rho1[:, None, None] * X0TDY \
            + rho1[:, None, None]**2 * X0TFY
        # dimension: #rho*#baseline*space
        X0TAX0_i = np.linalg.solve(X0TAX0, np.identity(n_X0)[None, :, :])
        # dimension: #rho*#baseline*#baseline
        XTAcorrX = XTAX
        # dimension: #rho*feature*feature
        XTAcorrY = XTAY
        # dimension: #rho*feature*space
        YTAcorrY_diag = YTAY_diag
        for i_r in range(np.size(rho1)):
            XTAcorrX[i_r, :, :] -= \
                np.dot(np.dot(XTAX0[i_r, :, :], X0TAX0_i[i_r, :, :]),
                       XTAX0[i_r, :, :].T)
            XTAcorrY[i_r, :, :] -= np.dot(np.dot(XTAX0[i_r, :, :],
                                                 X0TAX0_i[i_r, :, :]),
                                          X0TAY[i_r, :, :])
            YTAcorrY_diag[i_r, :] -= np.sum(
                X0TAY[i_r, :, :] * np.dot(X0TAX0_i[i_r, :, :],
                                          X0TAY[i_r, :, :]), axis=0)

        return X0TAX0, X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY_diag, \
            X0TAY, XTAX0

    def _fit_RSA_marginalized(self, X, Y, X_base,
                              scan_onsets=None):
        """ The major utility of fitting Bayesian RSA
            (marginalized version).
            Note that there is a naming change of variable. X in fit()
            is changed to Y here, and design in fit() is changed to X here.
            This is because we follow the tradition that X expresses the
            variable defined (controlled) by the experimenter, i.e., the
            time course of experimental conditions convolved by an HRF,
            and Y expresses data.
            However, in wrapper function fit(), we follow the naming
            routine of scikit-learn.
        """
        rank = self.rank
        n_subj = len(Y)
        n_V = [np.size(y, axis=1) for y in Y]
        n_T = [np.size(y, axis=0) for y in Y]
        n_C = np.size(X[0], axis=1)
        l_idx, rank = self._chol_idx(n_C, rank)
        n_l = np.size(l_idx[0])  # the number of parameters for L

        t_start = time.time()

        logger.info('Starting to fit the model. Maximum iteration: '
                    '{}.'.format(self.n_iter))

        # log_SNR_grids, SNR_weights \
        #     = np.polynomial.hermite.hermgauss(SNR_bins)
        # SNR_weights = SNR_weights / np.pi**0.5
        # SNR_grids = np.exp(log_SNR_grids * self.logS_range * 2**.5)

        SNR_grids, SNR_weights = self._set_SNR_grids()
        logger.info('The grids of pseudo-SNR used for numerical integration '
                    'is {}.'.format(SNR_grids))
        assert np.max(SNR_grids) < 1e10, \
            'ATTENTION!! The range of grids of pseudo-SNR' \
            ' to be marginalized is too large. Please ' \
            'consider reducing logS_range to 1 or 2'
        rho_grids, rho_weights = self._set_rho_grids()
        logger.info('The grids of rho used to do numerical integration '
                    'is {}.'.format(rho_grids))
        n_grid = self.SNR_bins * self.rho_bins
        log_weights = np.reshape(
            np.log(SNR_weights[:, None]) + np.log(rho_weights), n_grid)
        all_rho_grids = np.reshape(np.repeat(
            rho_grids[None, :], self.SNR_bins, axis=0), n_grid)
        all_SNR_grids = np.reshape(np.repeat(
            SNR_grids[:, None], self.rho_bins, axis=1), n_grid)
        # Prepare the data for fitting. These pre-calculated matrices
        # will be re-used a lot in evaluating likelihood function and
        # gradient.
        D = [None] * n_subj
        F = [None] * n_subj
        run_TRs = [None] * n_subj
        n_run = [None] * n_subj
        XTY = [None] * n_subj
        XTDY = [None] * n_subj
        XTFY = [None] * n_subj
        YTY_diag = [None] * n_subj
        YTDY_diag = [None] * n_subj
        YTFY_diag = [None] * n_subj
        XTX = [None] * n_subj
        XTDX = [None] * n_subj
        XTFX = [None] * n_subj

        X0TX0 = [None] * n_subj
        X0TDX0 = [None] * n_subj
        X0TFX0 = [None] * n_subj
        XTX0 = [None] * n_subj
        XTDX0 = [None] * n_subj
        XTFX0 = [None] * n_subj
        X0TY = [None] * n_subj
        X0TDY = [None] * n_subj
        X0TFY = [None] * n_subj
        X0 = [None] * n_subj
        X_res = [None] * n_subj
        n_X0 = [None] * n_subj
        idx_DC = [None] * n_subj
        log_fixed_terms = [None] * n_subj

        # Initialization for L.
        # There are several possible ways of initializing the covariance.
        # (1) start from the point estimation of covariance

        cov_point_est = np.zeros((n_C, n_C))

        for subj in range(n_subj):
            D[subj], F[subj], run_TRs[subj], n_run[subj] = self._prepare_DF(
                n_T[subj], scan_onsets=scan_onsets[subj])
            XTY[subj], XTDY[subj], XTFY[subj], YTY_diag[subj], \
                YTDY_diag[subj], YTFY_diag[subj], XTX[subj], XTDX[subj], \
                XTFX[subj] = self._prepare_data_XY(
                    X[subj], Y[subj], D[subj], F[subj])
            # The contents above stay fixed during fitting.

            # Initializing X0 as DC baseline
            # DC component will be added to the nuisance regressors.
            # In later steps, we do not need to add DC components again
            X0TX0[subj], X0TDX0[subj], X0TFX0[subj], XTX0[subj], XTDX0[subj], \
                XTFX0[subj], X0TY[subj], X0TDY[subj], X0TFY[subj], X0[subj], \
                X_base[subj], n_X0[subj], idx_DC[subj] = \
                self._prepare_data_XYX0(
                    X[subj], Y[subj], X_base[subj], None, D[subj], F[subj],
                    run_TRs[subj], no_DC=False)

            X_joint = np.concatenate((X0[subj], X[subj]), axis=1)
            beta_hat = np.linalg.lstsq(X_joint, Y[subj])[0]
            residual = Y[subj] - np.dot(X_joint, beta_hat)
            # point estimates of betas and fitting residuals without assuming
            # the Bayesian model underlying RSA.

            cov_point_est += np.cov(beta_hat[n_X0[subj]:, :]
                                    / np.std(residual, axis=0))
            log_fixed_terms[subj] = - (n_T[subj] - n_X0[subj]) \
                / 2 * np.log(2 * np.pi) + n_run[subj] \
                / 2 * np.log(1 - all_rho_grids**2) \
                + scipy.special.gammaln(
                    (n_T[subj] - n_X0[subj] - 2) / 2) \
                + (n_T[subj] - n_X0[subj] - 2) / 2 * np.log(2)
            # These are terms in the log likelihood that do not
            # depend on L. Notice that the last term comes from
            # ther term of marginalizing sigma. We take the 2 in
            # the denominator out. Accordingly, the "denominator"
            # variable in the _raw_loglike_grids() function is not
            # divided by 2
        cov_point_est = cov_point_est / n_subj
        current_vec_U_chlsk_l = np.linalg.cholesky(
            (cov_point_est + np.eye(n_C)) / 2)[l_idx]
        # We use the average of covariance of point estimation and an identity
        # matrix as the initial value of the covariance matrix, just in case
        # the user provides data in which n_V is smaller than n_C.

        # (2) start from identity matrix

        # current_vec_U_chlsk_l = np.eye(n_C)[l_idx]

        # (3) random initialization

        # current_vec_U_chlsk_l = self.random_state_.randn(n_l)
        # vectorized version of L, Cholesky factor of U, the shared
        # covariance matrix of betas across voxels.
        L = np.zeros((n_C, rank))
        L[l_idx] = current_vec_U_chlsk_l

        X0TAX0 = [None] * n_subj
        X0TAX0_i = [None] * n_subj
        XTAcorrX = [None] * n_subj
        s2XTAcorrX = [None] * n_subj
        YTAcorrY_diag = [None] * n_subj
        XTAcorrY = [None] * n_subj
        sXTAcorrY = [None] * n_subj
        X0TAY = [None] * n_subj
        XTAX0 = [None] * n_subj
        half_log_det_X0TAX0 = [None] * n_subj
        s_post = [None] * n_subj
        rho_post = [None] * n_subj
        sigma_post = [None] * n_subj
        beta_post = [None] * n_subj
        beta0_post = [None] * n_subj
        # The contents below can be updated during fitting.
        # e.g., X0 will be re-estimated
        logger.info('start real fitting')
        LL = np.zeros(n_subj)
        for it in range(self.n_iter):
            logger.info('Iteration {}'.format(it))
            # Re-estimate part of X0: X_res
            for subj in range(n_subj):
                if self.auto_nuisance and it > 0:
                    residuals = Y[subj] - np.dot(X[subj], beta_post[subj]) \
                        - np.dot(
                            X_base[subj],
                            beta0_post[subj][:np.shape(X_base[subj])[1], :])
                    X_res[subj] = self.nureg_method(
                        self.n_nureg_[subj]).fit_transform(
                        self.preprocess_residual(residuals))
                    X0TX0[subj], X0TDX0[subj], X0TFX0[subj], XTX0[subj],\
                        XTDX0[subj], XTFX0[subj], X0TY[subj], X0TDY[subj], \
                        X0TFY[subj], X0[subj], X_base[subj], n_X0[subj], _ = \
                        self._prepare_data_XYX0(
                            X[subj], Y[subj], X_base[subj], X_res[subj],
                            D[subj], F[subj], run_TRs[subj], no_DC=True)
                X0TAX0[subj], X0TAX0_i[subj], XTAcorrX[subj], XTAcorrY[subj],\
                    YTAcorrY_diag[subj], X0TAY[subj], XTAX0[subj] \
                    = self._precompute_ar1_quad_forms_marginalized(
                        XTY[subj], XTDY[subj], XTFY[subj], YTY_diag[subj],
                        YTDY_diag[subj], YTFY_diag[subj], XTX[subj],
                        XTDX[subj], XTFX[subj], X0TX0[subj], X0TDX0[subj],
                        X0TFX0[subj], XTX0[subj], XTDX0[subj], XTFX0[subj],
                        X0TY[subj], X0TDY[subj], X0TFY[subj], rho_grids,
                        n_V[subj], n_X0[subj])

                # Now we expand to another dimension including SNR
                # and collapse the dimension again.
                half_log_det_X0TAX0[subj], X0TAX0[subj], X0TAX0_i[subj], \
                    s2XTAcorrX[subj], YTAcorrY_diag[subj], sXTAcorrY[subj], \
                    X0TAY[subj], XTAX0[subj] = self._matrix_flattened_grid(
                        X0TAX0[subj], X0TAX0_i[subj], SNR_grids,
                        XTAcorrX[subj], YTAcorrY_diag[subj], XTAcorrY[subj],
                        X0TAY[subj], XTAX0[subj], n_C, n_V[subj], n_X0[subj],
                        n_grid)

            res = scipy.optimize.minimize(
                self._sum_loglike_marginalized, current_vec_U_chlsk_l
                + self.random_state_.randn(n_l) *
                np.linalg.norm(current_vec_U_chlsk_l)
                / n_l**0.5 * np.exp(-it / self.n_iter
                                    * self.anneal_speed - 1),
                args=(s2XTAcorrX, YTAcorrY_diag, sXTAcorrY,
                      half_log_det_X0TAX0,
                      log_weights, log_fixed_terms,
                      l_idx, n_C, n_T, n_V, n_X0,
                      n_grid, rank),
                method=self.optimizer, jac=True, tol=self.tol,
                options=self.minimize_options)
            param_change = res.x - current_vec_U_chlsk_l
            current_vec_U_chlsk_l = res.x.copy()

            # Estimating a few parameters.
            L[l_idx] = current_vec_U_chlsk_l
            for subj in range(n_subj):
                LL_raw, denominator, L_LAMBDA, L_LAMBDA_LT = \
                    self._raw_loglike_grids(
                        L, s2XTAcorrX[subj], YTAcorrY_diag[subj],
                        sXTAcorrY[subj], half_log_det_X0TAX0[subj],
                        log_weights, log_fixed_terms[subj], n_C, n_T[subj],
                        n_V[subj], n_X0[subj], n_grid, rank)
                result_sum, max_value, result_exp = utils.sumexp_stable(LL_raw)
                LL[subj] = np.sum(np.log(result_sum) + max_value)
                weight_post = result_exp / result_sum
                s_post[subj] = np.sum(all_SNR_grids[:, None] * weight_post,
                                      axis=0)
                # Mean-posterior estimate of SNR.
                rho_post[subj] = np.sum(all_rho_grids[:, None] * weight_post,
                                        axis=0)
                # Mean-posterior estimate of rho.
                sigma_means = denominator ** 0.5 \
                    * (np.exp(scipy.special.gammaln(
                        (n_T[subj] - n_X0[subj] - 3) / 2)
                        - scipy.special.gammaln(
                        (n_T[subj] - n_X0[subj] - 2) / 2)) / 2**0.5)
                sigma_post[subj] = np.sum(sigma_means * weight_post, axis=0)
                # The mean of inverse-Gamma distribution is beta/(alpha-1)
                # The mode is beta/(alpha+1). Notice that beta here does not
                # refer to the brain activation, but the scale parameter of
                # inverse-Gamma distribution. In the _UV version, we use the
                # maximum likelihood estimate of sigma^2. So we divide by
                # (alpha+1), which is (n_T - n_X0).
                beta_post[subj] = np.zeros((n_C, n_V[subj]))
                beta0_post[subj] = np.zeros((n_X0[subj], n_V[subj]))
                for grid in range(n_grid):
                    beta_post[subj] += np.dot(L_LAMBDA_LT[grid, :, :],
                                              sXTAcorrY[subj][grid, :, :])\
                        * all_SNR_grids[grid] \
                        * weight_post[grid, :]
                    beta0_post[subj] += weight_post[grid, :] * np.dot(
                        X0TAX0_i[subj][grid, :, :],
                        (X0TAY[subj][grid, :, :]
                         - np.dot(np.dot(XTAX0[subj][grid, :, :].T,
                                         L_LAMBDA_LT[grid, :, :]),
                                  sXTAcorrY[subj][grid, :, :])
                         * all_SNR_grids[grid]))
            if np.max(np.abs(param_change)) < self.tol:
                logger.info('The change of parameters is smaller than '
                            'the tolerance value {}. Fitting is finished '
                            'after {} iterations'.format(self.tol, it + 1))
                break
        for subj in range(n_subj):
            if idx_DC[subj].size > 1:
                collapsed_DC = np.sum(X0[subj][:, idx_DC[subj]], axis=1)
                X0[subj] = np.insert(np.delete(X0[subj], idx_DC[subj], axis=1),
                                     0, collapsed_DC, axis=1)
                collapsed_beta0 = np.mean(beta0_post[subj][idx_DC[subj], :],
                                          axis=0)
                beta0_post[subj] = np.insert(
                    np.delete(beta0_post[subj], idx_DC[subj], axis=0),
                    0, collapsed_beta0, axis=0)
        t_finish = time.time()
        logger.info(
            'total time of fitting: {} seconds'.format(t_finish - t_start))
        return np.dot(L, L.T), L, s_post, \
            beta_post, beta0_post, sigma_post, \
            rho_post, X0, LL

    def _fit_RSA_marginalized_null(self, Y, X_base,
                                   scan_onsets):
        """ The marginalized version of the null model for Bayesian RSA.
            The null model assumes no task-related response to the
            design matrix.
            Note that there is a naming change of variable. X in fit()
            is changed to Y here.
            This is because we follow the tradition that Y corresponds
            to data.
            However, in wrapper function fit(), we follow the naming
            routine of scikit-learn.
        """
        # Because there is nothing to learn that is shared across
        # participants, we can run each subject in serial.
        # The only fitting required is to re-estimate X0 after
        # each iteration
        n_subj = len(Y)
        t_start = time.time()
        logger.info('Starting to fit the model. Maximum iteration: '
                    '{}.'.format(self.n_iter))

        rho_grids, rho_weights = self._set_rho_grids()
        logger.info('The grids of rho used to do numerical integration '
                    'is {}.'.format(rho_grids))
        n_grid = self.rho_bins
        log_weights = np.log(rho_weights)
        rho_post = [None] * n_subj
        sigma_post = [None] * n_subj
        beta0_post = [None] * n_subj
        X0 = [None] * n_subj
        LL_null = np.zeros(n_subj)
        for subj in range(n_subj):
            logger.debug('Running on subject {}.'.format(subj))
            [n_T, n_V] = np.shape(Y[subj])
            D, F, run_TRs, n_run = self._prepare_DF(
                n_T, scan_onsets=scan_onsets[subj])
            YTY_diag = np.sum(Y[subj] * Y[subj], axis=0)
            YTDY_diag = np.sum(Y[subj] * np.dot(D, Y[subj]), axis=0)
            YTFY_diag = np.sum(Y[subj] * np.dot(F, Y[subj]), axis=0)

            # Add DC components capturing run-specific baselines.
            X_DC = self._gen_X_DC(run_TRs)
            X_DC, X_base[subj], idx_DC = self._merge_DC_to_base(
                X_DC, X_base[subj], no_DC=False)
            X_res = np.empty((n_T, 0))
            for it in range(0, self.n_iter):
                X0[subj] = np.concatenate(
                    (X_base[subj], X_res), axis=1)
                n_X0 = X0[subj].shape[1]
                X0TX0, X0TDX0, X0TFX0 = self._make_templates(
                    D, F, X0[subj], X0[subj])
                X0TY, X0TDY, X0TFY = self._make_templates(
                    D, F, X0[subj], Y[subj])

                YTAY_diag = YTY_diag - rho_grids[:, None] * YTDY_diag \
                    + rho_grids[:, None]**2 * YTFY_diag
                # dimension: #rho*space,
                # A/sigma2 is the inverse of noise covariance matrix.
                # YTAY means Y'AY

                X0TAX0 = X0TX0[None, :, :] \
                    - rho_grids[:, None, None] \
                    * X0TDX0[None, :, :] \
                    + rho_grids[:, None, None]**2 \
                    * X0TFX0[None, :, :]
                # dimension: #rho*#baseline*#baseline
                X0TAY = X0TY - rho_grids[:, None, None] * X0TDY \
                    + rho_grids[:, None, None]**2 * X0TFY
                # dimension: #rho*#baseline*space
                X0TAX0_i = np.linalg.solve(
                    X0TAX0, np.identity(n_X0)[None, :, :])
                # dimension: #rho*#baseline*#baseline
                YTAcorrY_diag = np.empty(np.shape(YTAY_diag))
                for i_r in range(np.size(rho_grids)):
                    YTAcorrY_diag[i_r, :] = YTAY_diag[i_r, :] \
                        - np.sum(X0TAY[i_r, :, :] * np.dot(
                            X0TAX0_i[i_r, :, :], X0TAY[i_r, :, :]),
                        axis=0)

                log_fixed_terms = - (n_T - n_X0) / 2 * np.log(2 * np.pi)\
                    + n_run / 2 * np.log(1 - rho_grids**2) \
                    + scipy.special.gammaln((n_T - n_X0 - 2) / 2) \
                    + (n_T - n_X0 - 2) / 2 * np.log(2)
                # These are terms in the log likelihood that do not
                # depend on L. Notice that the last term comes from
                # ther term of marginalizing sigma. We take the 2 in
                # the denominator out. Accordingly, the "denominator"
                # variable in the _raw_loglike_grids() function is not
                # divided by 2
                half_log_det_X0TAX0 = self._half_log_det(X0TAX0)

                LL_raw = -half_log_det_X0TAX0[:, None] \
                    - (n_T - n_X0 - 2) / 2 * np.log(YTAcorrY_diag) \
                    + log_weights[:, None] + log_fixed_terms[:, None]
                # dimension: n_grid * space
                # The log likelihood at each pair of values of rho1.
                # half_log_det_X0TAX0 is 0.5*log(det(X0TAX0)) with the size of
                # number of parameter grids. So is the size of log_weights

                result_sum, max_value, result_exp = utils.sumexp_stable(LL_raw)
                weight_post = result_exp / result_sum
                rho_post[subj] = np.sum(rho_grids[:, None] * weight_post,
                                        axis=0)
                # Mean-posterior estimate of rho.
                sigma_means = YTAcorrY_diag ** 0.5 \
                    * (np.exp(scipy.special.gammaln((n_T - n_X0 - 3) / 2)
                              - scipy.special.gammaln((n_T - n_X0 - 2) / 2))
                       / 2**0.5)
                sigma_post[subj] = np.sum(sigma_means * weight_post, axis=0)
                beta0_post[subj] = np.zeros((n_X0, n_V))
                for grid in range(n_grid):
                    beta0_post[subj] += weight_post[grid, :] * np.dot(
                        X0TAX0_i[grid, :, :], X0TAY[grid, :, :])
                if self.auto_nuisance:
                    residuals = Y[subj] - np.dot(
                        X_base[subj],
                        beta0_post[subj][:np.size(X_base[subj], 1), :])
                    X_res_new = self.nureg_method(
                        self.n_nureg_[subj]).fit_transform(
                        self.preprocess_residual(residuals))
                    if it >= 1:
                        if np.max(np.abs(X_res_new - X_res)) <= self.tol:
                            logger.info('The change of X_res is '
                                        'smaller than the tolerance value {}.'
                                        'Fitting is finished after {} '
                                        'iterations'.format(self.tol, it + 1))
                            break
                    X_res = X_res_new
            if idx_DC.size > 1:
                collapsed_DC = np.sum(X0[subj][:, idx_DC], axis=1)
                X0[subj] = np.insert(np.delete(X0[subj], idx_DC, axis=1), 0,
                                     collapsed_DC, axis=1)
                collapsed_beta0 = np.mean(beta0_post[subj][idx_DC, :], axis=0)
                beta0_post[subj] = np.insert(
                    np.delete(beta0_post[subj], idx_DC, axis=0),
                    0, collapsed_beta0, axis=0)
            LL_null[subj] = np.sum(np.log(result_sum) + max_value)
        t_finish = time.time()
        logger.info(
            'total time of fitting: {} seconds'.format(t_finish - t_start))
        return beta0_post, sigma_post, rho_post, X0, LL_null

    def _raw_loglike_grids(self, L, s2XTAcorrX, YTAcorrY_diag,
                           sXTAcorrY, half_log_det_X0TAX0,
                           log_weights, log_fixed_terms,
                           n_C, n_T, n_V, n_X0,
                           n_grid, rank):
        # LAMBDA_i = np.dot(np.einsum('ijk,jl->ilk', s2XTAcorrX, L), L) \
        #     + np.identity(rank)
        LAMBDA_i = np.empty((n_grid, rank, rank))
        for grid in np.arange(n_grid):
            LAMBDA_i[grid, :, :] = np.dot(np.dot(L.T,
                                                 s2XTAcorrX[grid, :, :]), L)
        LAMBDA_i += np.identity(rank)
        # dimension: n_grid * rank * rank
        Chol_LAMBDA_i = np.linalg.cholesky(LAMBDA_i)
        # dimension: n_grid * rank * rank
        half_log_det_LAMBDA_i = np.sum(
            np.log(np.abs(np.diagonal(Chol_LAMBDA_i, axis1=1, axis2=2))),
            axis=1)
        # dimension: n_grid
        L_LAMBDA = np.empty((n_grid, n_C, rank))
        L_LAMBDA_LT = np.empty((n_grid, n_C, n_C))
        s2YTAcorrXL_LAMBDA_LTXTAcorrY = np.empty((n_grid, n_V))
        # dimension: space * n_grid

        for grid in np.arange(n_grid):
            L_LAMBDA[grid, :, :] = scipy.linalg.cho_solve(
                (Chol_LAMBDA_i[grid, :, :], True), L.T).T
            L_LAMBDA_LT[grid, :, :] = np.dot(L_LAMBDA[grid, :, :], L.T)
            s2YTAcorrXL_LAMBDA_LTXTAcorrY[grid, :] = np.sum(
                sXTAcorrY[grid, :, :] * np.dot(L_LAMBDA_LT[grid, :, :],
                                               sXTAcorrY[grid, :, :]),
                axis=0)
        denominator = (YTAcorrY_diag - s2YTAcorrXL_LAMBDA_LTXTAcorrY)
        # dimension: n_grid * space
        # Not necessary the best name for it. But this term appears
        # as the denominator within the gradient wrt L
        # In the equation of the log likelihood, this "denominator"
        # term is in fact divided by 2. But we absorb that into the
        # log fixted term.
        LL_raw = -half_log_det_X0TAX0[:, None] \
            - half_log_det_LAMBDA_i[:, None] \
            - (n_T - n_X0 - 2) / 2 * np.log(denominator) \
            + log_weights[:, None] + log_fixed_terms[:, None]
        # dimension: n_grid * space
        # The log likelihood at each pair of values of SNR and rho1.
        # half_log_det_X0TAX0 is 0.5*log(det(X0TAX0)) with the size of
        # number of parameter grids. So is the size of log_weights
        return LL_raw, denominator, L_LAMBDA, L_LAMBDA_LT

    def _sum_loglike_marginalized(self, L_vec, s2XTAcorrX, YTAcorrY_diag,
                                  sXTAcorrY, half_log_det_X0TAX0,
                                  log_weights, log_fixed_terms,
                                  l_idx, n_C, n_T, n_V, n_X0,
                                  n_grid, rank=None):
        sum_LL_total = 0
        sum_grad_L = np.zeros(np.size(l_idx[0]))
        for subj in range(len(YTAcorrY_diag)):
            LL_total, grad_L = self._loglike_marginalized(
                L_vec, s2XTAcorrX[subj], YTAcorrY_diag[subj],
                sXTAcorrY[subj], half_log_det_X0TAX0[subj], log_weights,
                log_fixed_terms[subj], l_idx, n_C, n_T[subj],
                n_V[subj], n_X0[subj], n_grid, rank)
            sum_LL_total += LL_total
            sum_grad_L += grad_L
        return sum_LL_total, sum_grad_L

    def _loglike_marginalized(self, L_vec, s2XTAcorrX, YTAcorrY_diag,
                              sXTAcorrY, half_log_det_X0TAX0,
                              log_weights, log_fixed_terms,
                              l_idx, n_C, n_T, n_V, n_X0,
                              n_grid, rank=None):
        # In this version, we assume that beta is independent
        # between voxels and noise is also independent. X0 captures the
        # co-flucturation between voxels that is
        # not captured by design matrix X.
        # marginalized version marginalize sigma^2, s and rho1
        # for all voxels. n_grid is the number of grid on which the numeric
        # integration is performed to marginalize s and rho1 for each voxel.
        # The log likelihood is an inverse-Gamma distribution sigma^2,
        # so we can analytically marginalize it assuming uniform prior.
        # n_grid is the number of grid in the parameter space of (s, rho1)
        # that is used for numerical integration over (s, rho1).

        n_l = np.size(l_idx[0])
        # the number of parameters in the index of lower-triangular matrix

        if rank is None:
            rank = int((2 * n_C + 1
                        - np.sqrt(n_C**2 * 4 + n_C * 4 + 1 - 8 * n_l)) / 2)

        L = np.zeros([n_C, rank])
        L[l_idx] = L_vec

        LL_raw, denominator, L_LAMBDA, _ = self._raw_loglike_grids(
            L, s2XTAcorrX, YTAcorrY_diag, sXTAcorrY, half_log_det_X0TAX0,
            log_weights, log_fixed_terms, n_C, n_T, n_V, n_X0, n_grid, rank)

        result_sum, max_value, result_exp = utils.sumexp_stable(LL_raw)
        LL_total = np.sum(np.log(result_sum) + max_value)

        # Now we start the gradient with respect to L
        # s2XTAcorrXL_LAMBDA = np.einsum('ijk,ikl->ijl',
        #                                s2XTAcorrX, L_LAMBDA)
        s2XTAcorrXL_LAMBDA = np.empty((n_grid, n_C, rank))
        for grid in range(n_grid):
            s2XTAcorrXL_LAMBDA[grid, :, :] = np.dot(s2XTAcorrX[grid, :, :],
                                                    L_LAMBDA[grid, :, :])
        # dimension: n_grid * condition * rank
        I_minus_s2XTAcorrXL_LAMBDA_LT = np.identity(n_C) \
            - np.dot(s2XTAcorrXL_LAMBDA, L.T)
        # dimension: n_grid * condition * condition
        # The step above may be calculated by einsum. Not sure
        # which is faster.
        weight_grad = result_exp / result_sum
        weight_grad_over_denominator = weight_grad / denominator
        # dimension: n_grid * space
        weighted_sXTAcorrY = sXTAcorrY \
            * weight_grad_over_denominator[:, None, :]
        # dimension: n_grid * condition * space
        # sYTAcorrXL_LAMBDA = np.einsum('ijk,ijl->ikl', sXTAcorrY, L_LAMBDA)
        # dimension: n_grid * space * rank
        grad_L = np.zeros([n_C, rank])
        for grid in range(n_grid):
            grad_L += np.dot(
                np.dot(I_minus_s2XTAcorrXL_LAMBDA_LT[grid, :, :],
                       sXTAcorrY[grid, :, :]),
                np.dot(weighted_sXTAcorrY[grid, :, :].T,
                       L_LAMBDA[grid, :, :])) * (n_T - n_X0 - 2)
        grad_L -= np.sum(s2XTAcorrXL_LAMBDA
                         * np.sum(weight_grad, axis=1)[:, None, None],
                         axis=0)
        # dimension: condition * rank

        return -LL_total, -grad_L[l_idx]

    def _check_data_GBRSA(self, X, for_fit=True):
        # Check input data
        if type(X) is np.ndarray:
            X = [X]
        assert type(X) is list, 'Input data X must be either a list '\
            'with each entry for one participant, or a numpy arrary '\
            'for single participant.'
        if for_fit:
            for i, x in enumerate(X):
                assert_all_finite(x)
                assert x.ndim == 2, 'Each participants'' data should be ' \
                    '2 dimension ndarray'
                assert np.all(np.std(x, axis=0) > 0),\
                    'The time courses of some voxels in participant {} '\
                    'do not change at all. Please make sure all voxels '\
                    'are within the brain'.format(i)
        else:
            for i, x in enumerate(X):
                if x is not None:
                    assert x.ndim == 2, 'Each participants'' data should be ' \
                        '2 dimension ndarray'
                    assert x.shape[1] == self.n_V_[i], 'Number of voxels '\
                        'does not match that in the data used for fitting: '\
                        'subject {}'.format(i)
        # This program allows to fit a single subject. But to have a consistent
        # data structure, we make sure X and design are both lists.
        return X

    def _check_design_GBRSA(self, design, X):
        # check design matrix
        if type(design) is np.ndarray:
            design = [design] * len(X)
            if len(X) > 1:
                logger.warning('There are multiple subjects while '
                               'there is only one design matrix. '
                               'I assume that the design matrix '
                               'is shared across all subjects.')
        assert type(design) is list, 'design matrix must be either a list '\
            'with each entry for one participant, or an numpy arrary '\
            'for single participant.'

        for i, d in enumerate(design):
            if X[i] is not None:
                assert_all_finite(d)
                assert d.ndim == 2,\
                    'The design matrix should be 2 dimension ndarray'
                assert np.linalg.matrix_rank(d) == d.shape[1], \
                    'Your design matrix of subject {} has rank ' \
                    'smaller than the number of columns. Some columns '\
                    'can be explained by linear combination of other columns.'\
                    'Please check your design matrix.'.format(i)
                assert np.size(d, axis=0) == np.size(X[i], axis=0),\
                    'Design matrix and data of subject {} do not '\
                    'have the same number of time points.'.format(i)
                assert self.rank is None or self.rank <= d.shape[1],\
                    'Your design matrix of subject {} '\
                    'has fewer columns than the rank you set'.format(i)
                if i == 0:
                    n_C = np.shape(d)[1]
                else:
                    assert n_C == np.shape(d)[1], \
                        'In Group Bayesian RSA, all subjects should have'\
                        ' the same set of experiment conditions, t'\
                        'hus the same number of columns in design matrix'
                if X[i].shape[1] <= d.shape[1]:
                    logger.warning('Your data have fewer voxels than the '
                                   'number of task conditions. This might '
                                   'cause problem in fitting. Please consider '
                                   'increasing the size of your ROI, or set '
                                   'the rank parameter to a lower number to '
                                   'estimate a low-rank representational '
                                   'structure.')
        return design

    def _check_nuisance_GBRSA(sef, nuisance, X):
        # Check the nuisance regressors.
        if nuisance is not None:
            if type(nuisance) is np.ndarray:
                nuisance = [nuisance] * len(X)
                if len(X) > 1:
                    logger.warning('ATTENTION! There are multiple subjects '
                                   'while there is only one nuisance matrix. '
                                   'I assume that the nuisance matrix '
                                   'is shared across all subjects. '
                                   'Please double check.')
            assert type(nuisance) is list, \
                'nuisance matrix must be either a list '\
                'with each entry for one participant, or an numpy arrary '\
                'for single participant.'
            for i, n in enumerate(nuisance):
                assert_all_finite(n)
                if n is not None:
                    assert n.ndim == 2,\
                        'The nuisance regressor should be '\
                        '2 dimension ndarray or None'
                    assert np.linalg.matrix_rank(n) == n.shape[1], \
                        'The nuisance regressor of subject {} has rank '\
                        'smaller than the number of columns.'\
                        'Some columns can be explained by linear '\
                        'combination of other columns. Please check your' \
                        ' nuisance regressors.'.format(i)
                    assert np.size(n, axis=0) == np.size(X[i], axis=0), \
                        'Nuisance regressor and data do not have the same '\
                        'number of time points.'
        else:
            nuisance = [None] * len(X)
            logger.info('None was provided for nuisance matrix. Replicating '
                        'it for all subjects.')
        return nuisance

    def _check_scan_onsets_GBRSA(self, scan_onsets, X):
        # check scan_onsets validity
        if scan_onsets is None or type(scan_onsets) is np.ndarray:
            if scan_onsets is None:
                scan_onsets = np.array([0], dtype=int)
            scan_onsets = [scan_onsets] * len(X)
            if len(X) > 1:
                logger.warning('There are multiple subjects while '
                               'there is only one set of scan_onsets. '
                               'I assume that it is the same for all'
                               ' subjects. Please double check')
        for i in np.arange(len(scan_onsets)):
            if X[i] is not None:
                if scan_onsets[i] is None:
                    scan_onsets[i] = np.array([0], dtype=int)
                    logger.warning('No scan onsets were provided for subject'
                                   ' {}. Treating all data of this subject as'
                                   ' coming from the same run.')
                else:
                    scan_onsets[i] = np.int32(scan_onsets[i])
                assert (np.max(scan_onsets[i]) <= X[i].shape[0]
                        and np.min(scan_onsets[i]) >= 0
                        and 0 in scan_onsets[i]
                        and scan_onsets[i].ndim == 1), \
                    'Scan onsets of subject {} has formatting ' \
                    'issues: {}'.format(i, scan_onsets[i])
        return scan_onsets

    def _bin_exp(self, n_bin, scale=1.0):
        """ Calculate the bin locations to approximate exponential distribution.
            It breaks the cumulative probability of exponential distribution
            into n_bin equal bins, each covering 1 / n_bin probability. Then it
            calculates the center of mass in each bins and returns the
            centers of mass. So, it approximates the exponential distribution
            with n_bin of Delta function weighted by 1 / n_bin, at the
            locations of these centers of mass.
        Parameters:
        -----------
        n_bin: int
            The number of bins to approximate the exponential distribution
        scale: float, default: 1.0
            The scale parameter of the exponential distribution, defined in
            the same way as scipy.stats. It does not influence the ratios
            between the bins, but just controls the spacing between the bins.
            So generally users should not change its default.
        Returns:
        --------
        bins: numpy array of size [n_bin,]
            The centers of mass for each segment of the
            exponential distribution.
        """
        boundaries = np.flip(scipy.stats.expon.isf(
            np.linspace(0, 1, n_bin + 1),
            scale=scale), axis=0)
        bins = np.empty(n_bin)
        for i in np.arange(n_bin):
            bins[i] = utils.center_mass_exp(
                (boundaries[i], boundaries[i + 1]), scale=scale)
        return bins

    def _set_SNR_grids(self):
        """ Set the grids and weights for SNR used in numerical integration
            of SNR parameters.
        """
        if self.SNR_prior == 'unif':
            SNR_grids = np.linspace(0, 1, self.SNR_bins)
            SNR_weights = np.ones(self.SNR_bins) / (self.SNR_bins - 1)
            SNR_weights[0] = SNR_weights[0] / 2.0
            SNR_weights[-1] = SNR_weights[-1] / 2.0
        elif self.SNR_prior == 'lognorm':
            dist = scipy.stats.lognorm
            alphas = np.arange(np.mod(self.SNR_bins, 2),
                               self.SNR_bins + 2, 2) / self.SNR_bins
            # The goal here is to divide the area under the pdf curve
            # to segments representing equal probabilities.
            bounds = dist.interval(alphas, (self.logS_range,))
            bounds = np.unique(bounds)
            # bounds contain the boundaries which equally separate
            # the probability mass of the distribution
            SNR_grids = np.zeros(self.SNR_bins)
            for i in np.arange(self.SNR_bins):
                SNR_grids[i] = dist.expect(
                    lambda x: x, args=(self.logS_range,),
                    lb=bounds[i], ub=bounds[i + 1]) * self.SNR_bins
            # Center of mass of each segment between consecutive
            # bounds are set as the grids for SNR.
            SNR_weights = np.ones(self.SNR_bins) / self.SNR_bins
        else:  # SNR_prior == 'exp'
            SNR_grids = self._bin_exp(self.SNR_bins)
            SNR_weights = np.ones(self.SNR_bins) / self.SNR_bins
        SNR_weights = SNR_weights / np.sum(SNR_weights)
        return SNR_grids, SNR_weights

    def _set_rho_grids(self):
        """ Set the grids and weights for rho used in numerical integration
            of AR(1) parameters.
        """
        rho_grids = np.arange(self.rho_bins) * 2 / self.rho_bins - 1 \
            + 1 / self.rho_bins
        rho_weights = np.ones(self.rho_bins) / self.rho_bins
        return rho_grids, rho_weights

    def _matrix_flattened_grid(self, X0TAX0, X0TAX0_i, SNR_grids, XTAcorrX,
                               YTAcorrY_diag, XTAcorrY, X0TAY, XTAX0,
                               n_C, n_V, n_X0, n_grid):
        """ We need to integrate parameters SNR and rho on 2-d discrete grids.
            This function generates matrices which have only one dimension for
            these two parameters, with each slice in that dimension
            corresponding to each combination of the discrete grids of SNR
            and discrete grids of rho.
        """
        half_log_det_X0TAX0 = np.reshape(
            np.repeat(self._half_log_det(X0TAX0)[None, :],
                      self.SNR_bins, axis=0), n_grid)
        X0TAX0 = np.reshape(
            np.repeat(X0TAX0[None, :, :, :],
                      self.SNR_bins, axis=0),
            (n_grid, n_X0, n_X0))
        X0TAX0_i = np.reshape(np.repeat(
            X0TAX0_i[None, :, :, :],
            self.SNR_bins, axis=0),
            (n_grid, n_X0, n_X0))
        s2XTAcorrX = np.reshape(
            SNR_grids[:, None, None, None]**2 * XTAcorrX,
            (n_grid, n_C, n_C))
        YTAcorrY_diag = np.reshape(np.repeat(
            YTAcorrY_diag[None, :, :],
            self.SNR_bins, axis=0), (n_grid, n_V))
        sXTAcorrY = np.reshape(SNR_grids[:, None, None, None]
                               * XTAcorrY, (n_grid, n_C, n_V))
        X0TAY = np.reshape(np.repeat(X0TAY[None, :, :, :],
                                     self.SNR_bins, axis=0),
                           (n_grid, n_X0, n_V))
        XTAX0 = np.reshape(np.repeat(XTAX0[None, :, :, :],
                                     self.SNR_bins, axis=0),
                           (n_grid, n_C, n_X0))
        return half_log_det_X0TAX0, X0TAX0, X0TAX0_i, s2XTAcorrX, \
            YTAcorrY_diag, sXTAcorrY, X0TAY, XTAX0
