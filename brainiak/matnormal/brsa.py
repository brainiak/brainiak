import tensorflow as tf
from .helpers import define_scope
from .base import MatnormModelBase
from .covs import NoisePrecFullRank


class MatnormBRSA(MatnormModelBase):
    """ Alternate matrix normal version of BRSA.

    The goal of this analysis is to find the covariance of the mapping from
    some design matrixX to the fMRI signal Y. It does so by marginalizing over
    the actual mapping (i.e. averaging over the uncertainty in it), which
    happens to correct a bias imposed by structure in the design matrix on the
    RSA estimate (see Cai et al., NIPS 2016).

    This implementation makes different choices about two things relative to
    `brainiak.reprsimil.BRSA`:

    1. Estimation of both the cholesky of the RSA matrix L and the nuisance
    timecourse X_0 occurs simultaneously by gradient descent, in contrast to
    the alternating optimization method in BRSA.

    2. The noise covariance is assumed to be kronecker-separable. Informally,
    this means that all voxels has the same temporal covariance, and all time
    points have the same spatialcovariance. This is in contrast to BRSA, which
    allows different temporal covariance for each voxel. On the other hand,
    computational efficiencies enabled by this choice allow MatnormBRSA to
    support a richer class of space and time covariances (anything in
    `brainiak.matnormal.covs`).

    For users: in general, if you are worried about voxels each having
    different temporal noise structure,you should use `brainiak.reprsimil.BRSA`.
    If you are worried about between-voxel correlations or temporal covaraince
    structures that BRSA does not support, you should use MatnormBRSA.

    .. math::
        Y \\sim \\mathcal{MN}(0, \\Sigma_t + XLL^{\\top}X^{\\top}+ X_0X_0^{\\top}, \\Sigma_s)
        U = LL^{\\top}

    Parameters
    ----------
    n_TRs : int
        Number of TRs
    n_V : int
        number of voxels
    n_C : int
        number of conditions
    time_noise_cov : subclass of CovBase
        Temporal noise covariance class following CovBase interface.
    space_noise_cov : subclass of CovBase
        Spatial noise covariance class following CovBase interface.
    n_nureg : int
        Number of nuisance regressors
    learnRate : real, default=0.01
        Step size for the Adam optimizer

    """

    def __init__(self, n_TRs, n_V, n_C, time_noise_cov, space_noise_cov,
                 structured_RSA_cov=None, n_nureg=5, learnRate=0.01):

        if structured_RSA_cov is None:
            self.rsa_cov = NoisePrecFullRank(size=n_C+n_nureg)
        else:
            self.rsa_cov = structured_RSA_cov

        self.n_T = n_TRs
        self.n_V = n_V
        self.n_C = n_C

        # placeholders for param estimates
        self.X_0 = tf.Variable(tf.random_normal([self.n_T, n_nureg],
                                                dtype=tf.float64), name="X_0")

        # placeholders for inputs
        self.X = tf.placeholder(tf.float64, [self.n_T, n_C], name="Design")
        self.Y = tf.placeholder(tf.float64, [self.n_T, self.n_V], name="Brain")

        self.time_noise_cov = time_noise_cov
        self.space_noise_cov = space_noise_cov

        # register what we are optimizing over for TF's gradient descent
        self.train_variables = [self.X_0]
        self.train_variables.extend(time_noise_cov.get_optimize_vars())
        self.train_variables.extend(space_noise_cov.get_optimize_vars())
        self.train_variables.extend(self.rsa_cov.get_optimize_vars())

        # create optimizer node
        self.optimizer = tf.train.AdamOptimizer(learnRate)

        # create a tf session we reuse for this object
        self.sess = tf.Session()

        # create training nodes
        self.logp
        self.optimize

        # initialize
        self.sess.run(tf.global_variables_initializer())

    @define_scope
    def x_stack(self):
        """ Stack of X and X0, so we can just do:

        .. math::
            \\begin{bmatrix}X_0L_0\\\\ XL
            \\end{bmatrix}
            \\begin{bmatrix}X_0L_0 &  XL
            \\end{bmatrix}

            Note that doing this like this forces us to estimate the covariance
            of X0 -- which is probably fine for small n_nureg, but I'm not sure
            if estimating arbitrary correlation between X and X0 is problematic.
            In general what we should do here is define a low rank Cov
            (which is what X0 is) and make code for composing Covs so that
            Sigma_t can be the sum of a low rank and a toeplitz matrix, which is
            what this model basically is. Another alternative is define a
            Cov that is a direct sum (block diagonal) of other Covs.
            (Internal use.)
        """
        return tf.concat(1, [self.X, self.X_0])

    @define_scope
    def optimize(self):
        """tensorflow op for optimization
        """
        return self.optimizer.minimize(-self.logp,
                                       var_list=self.train_variables)

    def fit(self, X, y, max_iter=1000, step=100, loss_tol=1e-8, grad_tol=1e-8,
            restart=True):
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
        if restart:
            self.sess.run(tf.global_variables_initializer())

        feed_dict = {self.X: y, self.Y: X}

        self._optimize_impl(self.optimize, -self.logp, self.train_variables,
                            feed_dict, max_iter, step, loss_tol, grad_tol)

        self.U_ = self.rsa_cov.Sigma.eval(session=self.sess)[0: self.n_C,
                                                             0: self.n_C]

        self.X_0_ = self.X_0.eval(self.sess)

    @define_scope
    def logp(self):
        """ MatnormBRSA Log-likelihood"""

        return self.matnorm_logp_marginal_row(self.Y, row_cov=self.time_noise_cov,
                                              col_cov=self.space_noise_cov,
                                              marg=self.x_stack, marg_cov=self.rsa_cov)
