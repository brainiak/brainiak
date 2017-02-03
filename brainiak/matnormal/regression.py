import tensorflow as tf
import numpy as np
from .helpers import define_scope
from .base import MatnormModelBase


class MatnormRegression(MatnormModelBase):
    """ This analysis allows maximum likelihood estimation of regression models
    in the presence of both spatial and temporal covariance.

    ..math::
    Y \\sim \\mathcal{MN}(X\beta, time_noise_cov, space_noise_cov)

    Parameters
    ----------
    n_c : int
        Number of columns in design matrix
    n_v : int
        number of voxels
    time_noise_cov : subclass of CovBase
        TR noise covariance class following CovBase interface.
    space_noise_cov : subclass of CovBase
        Voxel noise covariance class following CovBase interface.
    learnRate : real, default=0.01
        Step size for the Adam optimizer

    """
    def __init__(self, n_c, n_v, time_noise_cov, space_noise_cov,
                 learnRate=0.01):

        self.n_c, self.n_v, self.time_noise_cov, self.space_noise_cov =\
            n_c, n_v, time_noise_cov, space_noise_cov

        self.Y = tf.placeholder(tf.float64, [None, n_v], name="Y")

        self.X = tf.placeholder(tf.float64, [None, n_c], name="X")

        self.beta = tf.Variable(tf.random_normal([n_c, n_v], dtype=tf.float64),
                                name="beta")

        self.train_variables = [self.beta]

        self.train_variables.extend(time_noise_cov.get_optimize_vars())
        self.train_variables.extend(space_noise_cov.get_optimize_vars())

        # create optimizer node
        self.optimizer = tf.train.AdamOptimizer(learnRate)

        # create a tf session we reuse for this object
        self.sess = tf.Session()

        # create training nodes
        self.train_logp
        self.train_optimize

        self.sess.run(tf.global_variables_initializer())

    @define_scope
    def train_optimize(self):
        """ Optimizer op (internal)
        """
        return self.optimizer.minimize(-self.train_logp,
                                       var_list=self.train_variables)

    @define_scope
    def train_logp(self):
        """ Log likelihood of model (internal)
        """
        y_hat = tf.matmul(self.X, self.beta)
        resid = self.Y - y_hat
        return self.matnorm_logp(resid, self.time_noise_cov, self.space_noise_cov)

    def fit(self, X, y, voxel_pos=None, times=None, max_iter=1000, step=100,
            loss_tol=1e-8, grad_tol=1e-8, restart=True):
        """ Compute the regression fit.

        Parameters
        ----------
        X : np.array, TRs by conditions.
            Design matrix
        Y : np.array, TRs by voxels.
            fMRI data
        voxel_pos: np.array, n_voxels by 3, default: None
            Spatial positions of voxels (optional).
            If provided, and if space_noise_cov is a CovGP, the positions
            for computing the GP covaraince matrix. Otherwise CovGP
            defaults to distances of 1 unit between all voxels.
            Ignored by non-GP noise covariances.
        times : np.array, TRs by 1, default:None
            Timestamps of observations (optional).
            If provided, and if time_noise_cov is a CovGP, the the times
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

        if restart:
            self.sess.run(tf.global_variables_initializer())

        feed_dict = {self.X: X, self.Y: y}

        if voxel_pos is not None:
            feed_dict[self.space_noise_cov.loc] = voxel_pos

        if times is not None:
            feed_dict[self.time_noise_cov.loc] = times

        self._optimize_impl(self.train_optimize, -self.train_logp,
                            self.train_variables, feed_dict, max_iter,
                            step, loss_tol, grad_tol)

        self.beta_ = self.beta.eval(session=self.sess)

    def predict(self, X):
        """ Predict fMRI signal from design matrix.

        Parameters
        ----------
        X : np.array, TRs by conditions.
            Design matrix

        """

        y_predicted = tf.matmul(self.X, self.beta)

        self.y_predicted_ = y_predicted.eval(session=self.sess, 
                                             feed_dict={self.X: X})

        return self.y_predicted_

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
        Sigma_s_btrp = self.space_noise_cov.Sigma_inv_x(tf.transpose(self.beta))
        # Y Sigma_s^{-1} B'
        Y_Sigma_Btrp = tf.matmul(Y, Sigma_s_btrp)
        # (B Sigma_s^{-1} B')^{-1}
        B_Sigma_Btrp = tf.matrix_inverse(tf.matmul(self.beta, Sigma_s_btrp))

        X_test = tf.matmul(Y_Sigma_Btrp, B_Sigma_Btrp)

        return X_test.eval(session=self.sess)
