import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator
from brainiak.matnormal.matnormal_likelihoods import matnorm_logp
from tensorflow.contrib.opt import ScipyOptimizerInterface

__all__ = ['MatnormRegression']


class MatnormRegression(BaseEstimator):
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
    learnRate : real, default=0.01
        Step size for the Adam optimizer

    """
    def __init__(self, time_cov, space_cov,
                 optimizer='L-BFGS-B', optCtrl=None):

        self.optCtrl, self.optMethod = optCtrl, optimizer
        self.time_cov = time_cov
        self.space_cov = space_cov

        self.n_t = time_cov.size
        self.n_v = space_cov.size

        self.Y = tf.placeholder(tf.float64, [self.n_t, self.n_v], name="Y")

        self.X = tf.placeholder(tf.float64, [self.n_t, None], name="X")

        # create a tf session we reuse for this object
        self.sess = tf.Session()

    # @define_scope
    def logp(self):
        """ Log likelihood of model (internal)
        """
        y_hat = tf.matmul(self.X, self.beta)
        resid = self.Y - y_hat
        return matnorm_logp(resid, self.time_cov, self.space_cov)

    def fit(self, X, y):
        """ Compute the regression fit.

        Parameters
        ----------
        X : np.array, TRs by conditions.
            Design matrix
        Y : np.array, TRs by voxels.
            fMRI data
        voxel_pos: np.array, n_voxels by 3, default: None
            Spatial positions of voxels (optional).
            If provided, and if space_cov is a CovGP, the positions
            for computing the GP covaraince matrix. Otherwise CovGP
            defaults to distances of 1 unit between all voxels.
            Ignored by non-GP noise covariances.
        times : np.array, TRs by 1, default:None
            Timestamps of observations (optional).
            If provided, and if time_cov is a CovGP, the the times
            for computing the GP covaraince matrix. Otherwise CovGP
            defaults to distances of 1 unit between all times.
            Ignored by non-GP noise covariances.
        max_iter: int, default=1000
            Maximum number of iterations to run
        step: int, default=100
            Number of steps between optimizer status outputs.
        restart: bool, default=True
            If this is true, optimizer is restarted (e.g. for a new dataset).
            Otherwise optimizer will continue from where it is now (for example
            for running more iterations if the initial number was not enough).
        """

        self.n_c = X.shape[1]

        feed_dict = {self.X: X, self.Y: y}
        self.sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        # initialize to the least squares solution (basically all
        # we need now is the cov)
        sigma_inv_x = self.time_cov.solve(self.X)\
            .eval(session=self.sess, feed_dict=feed_dict)
        sigma_inv_y = self.time_cov.solve(self.Y)\
            .eval(session=self.sess, feed_dict=feed_dict)

        beta_init = np.linalg.solve((X.T).dot(sigma_inv_x),
                                    (X.T).dot(sigma_inv_y))

        self.beta = tf.Variable(beta_init, name="beta")

        self.train_variables = [self.beta]
        self.train_variables.extend(self.time_cov.get_optimize_vars())
        self.train_variables.extend(self.space_cov.get_optimize_vars())

        self.sess.run(tf.variables_initializer([self.beta]))

        optimizer = ScipyOptimizerInterface(-self.logp(),
                                            var_list=self.train_variables,
                                            method=self.optMethod,
                                            options=self.optCtrl)

        optimizer.minimize(session=self.sess, feed_dict=feed_dict)

        self.beta_ = self.beta.eval(session=self.sess)

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
            X = Y \Sigma_s^{-1}B'(B \Sigma_s^{-1} B')^{-1}

        Parameters
        ----------
        Y : np.array, TRs by voxels.
            fMRI dataset
        """

        if (Y.shape[1] <= self.n_c):
            raise RuntimeError("More conditions than voxels! System is singular,\
                                cannot decode.")

        # Sigma_s^{-1} B'
        Sigma_s_btrp = self.space_cov.solve(tf.transpose(
                                                        self.beta))
        # Y Sigma_s^{-1} B'
        Y_Sigma_Btrp = tf.matmul(Y, Sigma_s_btrp).eval(session=self.sess)
        # (B Sigma_s^{-1} B')^{-1}
        B_Sigma_Btrp = tf.matmul(self.beta, Sigma_s_btrp)\
            .eval(session=self.sess)

        X_test = np.linalg.solve(B_Sigma_Btrp.T, Y_Sigma_Btrp.T).T

        return X_test
