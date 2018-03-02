import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from .covs import CovIdentity
from brainiak.utils.utils import cov2corr
import numpy as np
from brainiak.matnormal.matnormal_likelihoods import matnorm_logp_marginal_row
from tensorflow.contrib.opt import ScipyOptimizerInterface
import logging


class MNRSA(BaseEstimator):
    """ Matrix normal version of RSA.

    The goal of this analysis is to find the covariance of the mapping from
    some design matrixX to the fMRI signal Y. It does so by marginalizing over
    the actual mapping (i.e. averaging over the uncertainty in it), which
    happens to correct a bias imposed by structure in the design matrix on the
    RSA estimate (see Cai et al., NIPS 2016).

    This implementation makes different choices about two things relative to
    `brainiak.reprsimil.BRSA`:

    1. The noise covariance is assumed to be kronecker-separable. Informally,
    this means that all voxels has the same temporal covariance, and all time
    points have the same spatialcovariance. This is in contrast to BRSA, which
    allows different temporal covariance for each voxel. On the other hand,
    computational efficiencies enabled by this choice allow MNRSA to
    support a richer class of space and time covariances (anything in
    `brainiak.matnormal.covs`).

    2. MNRSA does not estimate the nuisance timecourse X_0. Instead,
    we expect the temporal noise covariance to capture the same property
    (because when marginalizing over B_0 gives a low-rank component to the noise
    covariance, something we hope to have available soon.

    For users: in general, if you are worried about voxels each having
    different temporal noise structure,you should use `brainiak.reprsimil.BRSA`.
    If you are worried about between-voxel correlations or temporal covaraince
    structures that BRSA does not support, you should use MNRSA.

    .. math::
        Y \\sim \\mathcal{MN}(0, \\Sigma_t + XLL^{\\top}X^{\\top}+ X_0X_0^{\\top}, \\Sigma_s)
        U = LL^{\\top}

    Parameters
    ----------
    time_cov : subclass of CovBase
        Temporal noise covariance class following CovBase interface.
    space_cov : subclass of CovBase
        Spatial noise covariance class following CovBase interface.
    optimizer : string, Default :'L-BFGS'
        Name of scipy optimizer to use.
    optCtrl :  dict, default: None
        Dict of options for optimizer (e.g. {'maxiter': 100})

    """

    def __init__(self, time_cov, space_cov, n_nureg=5,
                 optimizer='L-BFGS-B', optCtrl=None):

        self.n_T = time_cov.size
        self.n_V = space_cov.size
        self.n_nureg = n_nureg

        self.optCtrl, self.optMethod = optCtrl, optimizer

        # placeholders for inputs
        self.X = tf.placeholder(tf.float64, [self.n_T, None], name="Design")
        self.Y = tf.placeholder(tf.float64, [self.n_T, self.n_V], name="Brain")

        self.X_0 = tf.Variable(tf.random_normal([self.n_T, n_nureg],
                                                dtype=tf.float64), name="X_0")

        self.train_variables = [self.X_0]

        self.time_cov = time_cov
        self.space_cov = space_cov

        self.train_variables.extend(self.time_cov.get_optimize_vars())
        self.train_variables.extend(self.space_cov.get_optimize_vars())

        # create a tf session we reuse for this object
        self.sess = tf.Session()

    def fit(self, X, y, structured_RSA_cov=None):
        """ Estimate dimension reduction and cognitive model parameters

        Parameters
        ----------
        X: 2d array
            Brain data matrix (voxels by TRs). Y in the math
        y: 2d array or vector
            Behavior data matrix (behavioral obsevations by TRs). X in the math
        max_iter: int, default=1000
            Maximum number of iterations to run
        step: int, default=100
            Number of steps between optimizer output
        restart: bool, default=True
            If this is true, optimizer is restarted (e.g. for a new dataset).
            Otherwise optimizer will continue from where it is now (for example
            for running more iterations if the initial number was not enough).

        """

        # self.sess.run(tf.global_variables_initializer())

        feed_dict = {self.X: y, self.Y: X}

        self.n_c = y.shape[1]

        # initialize from naive RSA
        m = LinearRegression(fit_intercept=False)
        # counterintuitive given sklearn interface above:
        # brain is passed in as X and design is passed in as y
        m.fit(X=y, y=X)
        self.naive_U_ = np.cov(m.coef_.T)
        naiveRSA_L = np.linalg.cholesky(self.naive_U_)
        self.naive_C_ = cov2corr(self.naive_U_)
        self.L_full = tf.Variable(naiveRSA_L, name="L_full", dtype="float64")

        L_indeterminate = tf.matrix_band_part(self.L_full, -1, 0)
        self.L = tf.matrix_set_diag(L_indeterminate,
                                    tf.exp(tf.matrix_diag_part(L_indeterminate)))

        self.train_variables.extend([self.L_full])

        self.x_stack = tf.concat([tf.matmul(self.X, self.L), self.X_0], 1)
        self.sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        optimizer = ScipyOptimizerInterface(-self.logp(),
                                            var_list=self.train_variables,
                                            method=self.optMethod,
                                            options=self.optCtrl)

        if logging.getLogger().isEnabledFor(logging.INFO):
            optimizer._packed_loss_grad = tf.Print(optimizer._packed_loss_grad,
                                                   [tf.reduce_min(optimizer._packed_loss_grad)],
                                                   'mingrad')
            optimizer._packed_loss_grad = tf.Print(optimizer._packed_loss_grad,
                                                   [tf.reduce_max(optimizer._packed_loss_grad)],
                                                   'maxgrad')
            optimizer._packed_loss_grad = tf.Print(optimizer._packed_loss_grad,
                                                   [self.logp()], 'logp')

        optimizer.minimize(session=self.sess, feed_dict=feed_dict)

        self.L_ = self.L.eval(session=self.sess)
        self.X_0_ = self.X_0.eval(session=self.sess)
        self.U_ = self.L_.dot(self.L_.T)
        self.C_ = cov2corr(self.U_)

    def logp(self):
        """ MNRSA Log-likelihood"""

        rsa_cov = CovIdentity(size=self.n_c + self.n_nureg)

        return self.time_cov.logp + \
            self.space_cov.logp + \
            rsa_cov.logp + \
            matnorm_logp_marginal_row(self.Y, row_cov=self.time_cov,
                                      col_cov=self.space_cov,
                                      marg=self.x_stack, marg_cov=rsa_cov)
