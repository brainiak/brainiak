import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from .covs import CovIdentity
from brainiak.utils.utils import cov2corr
import numpy as np
from brainiak.matnormal.matnormal_likelihoods import matnorm_logp_marginal_row
from brainiak.matnormal.utils import (
    pack_trainable_vars,
    unpack_trainable_vars,
    make_val_and_grad,
    unflatten_cholesky_unique,
    flatten_cholesky_unique,
)

from scipy.optimize import minimize

__all__ = ["MNRSA"]


class MNRSA(BaseEstimator):
    """ Matrix normal version of RSA.

    The goal of this analysis is to find the covariance of the mapping from
    some design matrix X to the fMRI signal Y. It does so by marginalizing over
    the actual mapping (i.e. averaging over the uncertainty in it), which
    happens to correct a bias imposed by structure in the design matrix on the
    RSA estimate (see Cai et al., NIPS 2016).

    This implementation makes different choices about residual covariance
    relative to `brainiak.reprsimil.BRSA`: Here, the noise covariance is
    assumed to be kronecker-separable. Informally, this means that all voxels
    have the same temporal covariance, and all time points have the same
    spatial covariance. This is in contrast to BRSA, which allows different
    temporal covariance for each voxel. On the other hand, computational
    efficiencies enabled by this choice allow MNRSA to support a richer class
    of space and time covariances (anything in `brainiak.matnormal.covs`).

    For users: in general, if you are worried about voxels each having
    different temporal noise structure,you should use
    `brainiak.reprsimil.BRSA`. If you are worried about between-voxel
    correlations or temporal covaraince structures that BRSA does not
    support, you should use MNRSA.

    .. math::
        Y &\\sim \\mathcal{MN}(0, \\Sigma_t + XLL^TX^T+
        X_0X_0^T, \\Sigma_s)\\

        U &= LL^T

    Parameters
    ----------
    time_cov : subclass of CovBase
        Temporal noise covariance class following CovBase interface.
    space_cov : subclass of CovBase
        Spatial noise covariance class following CovBase interface.
    optimizer : string, Default :'L-BFGS'
        Name of scipy optimizer to use.
    optCtrl :  dict, default: None
        Additional arguments to pass to scipy.optimize.minimize.

    """

    def __init__(
        self, time_cov, space_cov, n_nureg=5, optimizer="L-BFGS-B",
        optCtrl=None
    ):

        self.n_T = time_cov.size
        self.n_V = space_cov.size
        self.n_nureg = n_nureg

        self.optMethod = optimizer
        if optCtrl is None:
            self.optCtrl = {}

        self.X_0 = tf.Variable(
            tf.random.normal([self.n_T, n_nureg], dtype=tf.float64), name="X_0"
        )

        self.train_variables = [self.X_0]

        self.time_cov = time_cov
        self.space_cov = space_cov

        self.train_variables.extend(self.time_cov.get_optimize_vars())
        self.train_variables.extend(self.space_cov.get_optimize_vars())

    @property
    def L(self):
        """
        Cholesky factor of the RSA matrix.
        """
        return unflatten_cholesky_unique(self.L_flat)

    def fit(self, X, y, naive_init=True):
        """ Estimate dimension reduction and cognitive model parameters

        Parameters
        ----------
        X: 2d array
            Brain data matrix (TRs by voxels). Y in the math
        y: 2d array or vector
            Behavior data matrix (TRs by behavioral obsevations). X in the math
        max_iter: int, default=1000
            Maximum number of iterations to run
        step: int, default=100
            Number of steps between optimizer output
        restart: bool, default=True
            If this is true, optimizer is restarted (e.g. for a new dataset).
            Otherwise optimizer will continue from where it is now (for example
            for running more iterations if the initial number was not enough).

        """

        # In the method signature we follow sklearn discriminative API
        # where brain is X and behavior is y. Internally we are
        # generative so we flip this here
        X, Y = y, X

        self.n_c = X.shape[1]

        if naive_init:
            # initialize from naive RSA
            m = LinearRegression(fit_intercept=False)
            m.fit(X=X, y=Y)
            self.naive_U_ = np.cov(m.coef_.T)
            naiveRSA_L = np.linalg.cholesky(self.naive_U_)
            self.L_flat = tf.Variable(
                flatten_cholesky_unique(naiveRSA_L), name="L_flat",
                dtype="float64"
            )
        else:
            chol_flat_size = (self.n_c * (self.n_c + 1)) // 2
            self.L_flat = tf.Variable(
                tf.random.normal([chol_flat_size], dtype="float64"),
                name="L_flat",
                dtype="float64",
            )

        self.train_variables.extend([self.L_flat])

        def lossfn(theta): return -self.logp(X, Y)
        val_and_grad = make_val_and_grad(lossfn, self.train_variables)

        x0 = pack_trainable_vars(self.train_variables)

        opt_results = minimize(fun=val_and_grad, x0=x0,
                               jac=True, method=self.optMethod, **self.optCtrl)

        unpacked_theta = unpack_trainable_vars(
            opt_results.x, self.train_variables)
        for var, val in zip(self.train_variables, unpacked_theta):
            var.assign(val)

        self.U_ = self.L.numpy().dot(self.L.numpy().T)
        self.C_ = cov2corr(self.U_)

    def logp(self, X, Y):
        """ MNRSA Log-likelihood"""

        rsa_cov = CovIdentity(size=self.n_c + self.n_nureg)
        x_stack = tf.concat([tf.matmul(X, self.L), self.X_0], 1)
        return (
            self.time_cov.logp
            + self.space_cov.logp
            + rsa_cov.logp
            + matnorm_logp_marginal_row(
                Y,
                row_cov=self.time_cov,
                col_cov=self.space_cov,
                marg=x_stack,
                marg_cov=rsa_cov,
            )
        )
