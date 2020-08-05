import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.base import BaseEstimator
from brainiak.matnormal.matnormal_likelihoods import matnorm_logp
from brainiak.matnormal.utils import (
    pack_trainable_vars,
    unpack_trainable_vars,
    make_val_and_grad,
)
from scipy.optimize import minimize

__all__ = ["MatnormalRegression"]


class MatnormalRegression(BaseEstimator):
    """ This analysis allows maximum likelihood estimation of regression models
    in the presence of both spatial and temporal covariance.

    ..math::
    Y \\sim \\mathcal{MN}(X\beta, time_cov, space_cov)

    Parameters
    ----------
    time_cov : subclass of CovBase
        TR noise covariance class following CovBase interface.
    space_cov : subclass of CovBase
        Voxel noise covariance class following CovBase interface.
    optimizer : string, default="L-BFGS-B"
        Scipy optimizer to use. For other options, see "method" argument
        of scipy.optimize.minimize
    optCtrl: dict, default=None
        Additional arguments to pass to scipy.optimize.minimize.

    """

    def __init__(self, time_cov, space_cov, optimizer="L-BFGS-B", optCtrl=None):

        self.optCtrl, self.optMethod = optCtrl, optimizer
        self.time_cov = time_cov
        self.space_cov = space_cov

        self.n_t = time_cov.size
        self.n_v = space_cov.size

    def logp(self, X, Y):
        """ Log likelihood of model (internal)
        """
        y_hat = tf.matmul(X, self.beta)
        resid = Y - y_hat
        return matnorm_logp(resid, self.time_cov, self.space_cov)

    def fit(self, X, y):
        """ Compute the regression fit.

        Parameters
        ----------
        X : np.array, TRs by conditions.
            Design matrix
        Y : np.array, TRs by voxels.
            fMRI data
        """

        self.n_c = X.shape[1]

        # initialize to the least squares solution (basically all
        # we need now is the cov)
        sigma_inv_x = self.time_cov.solve(X)
        sigma_inv_y = self.time_cov.solve(y)

        beta_init = np.linalg.solve((X.T).dot(sigma_inv_x), (X.T).dot(sigma_inv_y))

        self.beta = tf.Variable(beta_init, name="beta")

        self.train_variables = [self.beta]
        self.train_variables.extend(self.time_cov.get_optimize_vars())
        self.train_variables.extend(self.space_cov.get_optimize_vars())

        val_and_grad = make_val_and_grad(self, extra_args=(X, y))
        x0 = pack_trainable_vars(self.train_variables)

        opt_results = minimize(
            fun=val_and_grad, x0=x0, args=(X, y), jac=True, method="L-BFGS-B"
        )

        unpacked_theta = unpack_trainable_vars(opt_results.x, self.train_variables)
        for var, val in zip(self.train_variables, unpacked_theta):
            var = val

    def predict(self, X):
        """ Predict fMRI signal from design matrix.

        Parameters
        ----------
        X : np.array, TRs by conditions.
            Design matrix

        """

        return X.dot(self.beta_)

    def calibrate(self, Y):
        """ Decode design matrix from fMRI dataset, based on a previously
        trained mapping. This method just does naive MLE:

        .. math::
            X = Y \\Sigma_s^{-1}B'(B \\Sigma_s^{-1} B')^{-1}

        Parameters
        ----------
        Y : np.array, TRs by voxels.
            fMRI dataset
        """

        if Y.shape[1] <= self.n_c:
            raise RuntimeError(
                "More conditions than voxels! System is singular,\
                                cannot decode."
            )

        # Sigma_s^{-1} B'
        Sigma_s_btrp = self.space_cov.solve(tf.transpose(a=self.beta))
        # Y Sigma_s^{-1} B'
        Y_Sigma_Btrp = tf.matmul(Y, Sigma_s_btrp).eval(session=self.sess)
        # (B Sigma_s^{-1} B')^{-1}
        B_Sigma_Btrp = tf.matmul(self.beta, Sigma_s_btrp).eval(session=self.sess)

        X_test = np.linalg.solve(B_Sigma_Btrp.T, Y_Sigma_Btrp.T).T

        return X_test
