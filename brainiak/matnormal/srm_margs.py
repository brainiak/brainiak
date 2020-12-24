import inspect
import logging
import warnings

from brainiak.matnormal.covs import CovIdentity
from brainiak.matnormal.utils import make_val_and_grad, pack_trainable_vars

import numpy as np

from pymanopt import Problem
from pymanopt.function import TensorFlow
from pymanopt.manifolds import Product, Stiefel
from pymanopt.solvers import TrustRegions

from scipy.optimize import minimize

from sklearn.base import BaseEstimator

import tensorflow as tf


logger = logging.getLogger(__name__)


class MNSRM(BaseEstimator):
    """Probabilistic SRM, aka SRM with marginalization over S (and 
    orthonormal W). This generalizes SRM (Chen et al. 2015) by allowing
    arbitrary kronecker-structured residual covariance. Inference is
    performed by ECM algorithm. 
    """

    def __init__(self, n_features=5, time_noise_cov=CovIdentity,
                 space_noise_cov=CovIdentity, s_cov=CovIdentity,
                 optMethod="L-BFGS-B", optCtrl={}):

        self.k = n_features

        self.time_noise_cov_class = time_noise_cov
        self.space_noise_cov_class = space_noise_cov
        self.marg_cov_class = s_cov

        self.optCtrl, self.optMethod = optCtrl, optMethod

    def Q_fun(self, W, X):
        """
        Q function for ECM algorithm
        """

        mean = X - self.b - \
            tf.matmul(W, tf.tile(tf.expand_dims(
                self.s_prime, 0), [self.n, 1, 1]))

        det_terms = -(self.v*self.n + self.k)*self.time_cov.logdet -\
                     (self.t*self.n)*self.space_cov.logdet -\
            self.t*self.marg_cov.logdet +\
                     (self.t*self.v) * \
            tf.reduce_sum(input_tensor=tf.math.log(self.rhoprec))

        # used twice below
        trace_t_t = tf.linalg.trace(self.time_cov.solve(self.tcov_prime))

        # covs don't support batch ops (yet!) (TODO):
        x_quad_form = -tf.linalg.trace(tf.reduce_sum(input_tensor=[tf.matmul(self.time_cov.solve(tf.transpose(a=mean[j])),
                                                                             self.space_cov.solve(mean[j]))*self.rhoprec[j]
                                                                   for j in range(self.n)], axis=0))

        w_quad_form = -tf.linalg.trace(tf.reduce_sum(input_tensor=[tf.matmul(tf.matmul(self.scov_prime, tf.transpose(a=W[j])),
                                                                             self.space_cov.solve(W[j]))*self.rhoprec[j]
                                                                   for j in range(self.n)], axis=0)) * trace_t_t

        s_quad_form = -tf.linalg.trace(tf.matmul(self.time_cov.solve(tf.transpose(a=self.s_prime)),
                                                 self.marg_cov.solve(self.s_prime)))

        sig_trace_prod = -trace_t_t * \
            tf.linalg.trace(self.marg_cov.solve(self.scov_prime))

        return 0.5 * (det_terms + x_quad_form + s_quad_form + w_quad_form + sig_trace_prod)

    def estep(self, X):
        """
        Compute expectation of the log posterior density (aka complete data log-likelihood)
        for ECM.
        """

        tcov_prime = self.time_cov
        Xmb = X - self.b

        sprec_chol = tf.linalg.cholesky(self.marg_cov._prec + tf.reduce_sum(input_tensor=[tf.matmul(tf.transpose(
            a=self.W[j]), self.space_cov.solve(self.W[j]))*self.rhoprec[j] for j in range(self.n)], axis=0))

        wsig_x = tf.reduce_sum(input_tensor=[tf.matmul(tf.transpose(
            a=self.W[j]), self.space_cov.solve(Xmb[j]))*self.rhoprec[j] for j in range(self.n)], axis=0)

        scov_prime = tf.linalg.cholesky_solve(
            sprec_chol, tf.eye(self.k, dtype=tf.float64))

        s_prime = tf.linalg.cholesky_solve(sprec_chol, wsig_x)

        return s_prime, scov_prime, tcov_prime._cov

    def mstep_b(self, X):
        """
        Update b (intercept term) as part of M-step.
        """

        return tf.expand_dims(tf.reduce_sum(input_tensor=[self.time_cov.solve(tf.transpose(a=X[j] - tf.matmul(self.W[j], self.s_prime)))
                                                          for j in range(self.n)], axis=1) /
                              tf.reduce_sum(input_tensor=self.time_cov._prec), -1)

    def mstep_rhoprec(self, X):
        """
        Update rho^-1 (subject-wise precision scalers) as part of M-step. 
        """

        mean = X - self.b - \
            tf.matmul(self.W, tf.tile(tf.expand_dims(
                self.s_prime, 0), [self.n, 1, 1]))

        mean_trace = tf.stack([tf.linalg.trace(tf.matmul(self.time_cov.solve(tf.transpose(a=mean[j])),
                                                         self.space_cov.solve(mean[j]))) for j in range(self.n)])

        trace_t_t = tf.linalg.trace(self.time_cov.solve(self.tcov_prime))

        w_trace = trace_t_t * tf.stack([tf.linalg.trace(tf.matmul(tf.matmul(self.scov_prime, tf.transpose(a=self.W[j])),
                                                                  self.space_cov.solve(self.W[j]))) for j in range(self.n)])

        rho_hat_unscaled = mean_trace + w_trace

        return (self.v*self.t) / rho_hat_unscaled

    def fit(self, X, n_iter=10, y=None, w_cov=None, svd_init=True):
        """
        Fit SRM by ECM marginalizing over S. 

        Parameters
        ----------
        X: 2d array
            Brain data matrix (voxels by TRs). Y in the math
        n_iter: int, default=10
            Number of ECM iterations to run
        y: None
            Ignored (just here for sklearn API compatibility)
        w_cov : CovBase, default = CovIdentity
            Prior covariance of the columns of W.
        svd_init : bool, default=True
            If true, initialize to the W_i left singular vectors of
            X_i and S to the average of the right singular vectors
            over all subjects. If false, initialize to random orthonormal
            matrices. 
        """

        self.n = len(X)

        self.v, self.t = X[0].shape

        X = tf.stack(X, name="X")

        if svd_init:
            xsvd = [np.linalg.svd(x) for x in X]
        else:
            xsvd = [np.linalg.svd(np.random.normal(
                size=(self.v, self.t))) for i in range(self.n)]

        w_init = [sv[0][:, :self.k] for sv in xsvd]
        s_init = np.average([sv[2][:self.k, :] for sv in xsvd], 0)

        # parameters
        self.b = tf.Variable(np.random.normal(
            size=(self.n, self.v, 1)), name="b")
        self.rhoprec = tf.Variable(np.ones(self.n), name="rhoprec")
        self.W = tf.Variable(tf.stack([_w for _w in w_init]))
        self.space_cov = self.space_noise_cov_class(size=self.v)
        self.time_cov = self.time_noise_cov_class(size=self.t)
        self.marg_cov = self.time_noise_cov_class(size=self.k)

        # sufficient statistics
        self.s_prime = tf.Variable(s_init, dtype=tf.float64, name="s_prime")
        self.scov_prime = tf.Variable(np.eye(self.k), name="wcov_prime")
        self.tcov_prime = tf.Variable(np.eye(self.t), name="wcov_prime")

        # Pymanopt setup

        # now we fool pymanopt into thinking we prepicked
        # number of args even though we use varargs
        def wrapped_Q(*args):
            return -self.Q_fun(args, X)

        sig = inspect.signature(wrapped_Q)
        newparams = [inspect.Parameter(
            f"w_{i}", inspect.Parameter.POSITIONAL_ONLY) for i in range(self.n)]
        newsig = sig.replace(parameters=newparams)
        wrapped_Q.__signature__ = newsig
        lossfn_Q = TensorFlow(wrapped_Q)

        w_manifold = Product([Stiefel(self.v, self.k) for i in range(self.n)])
        solver = TrustRegions(logverbosity=0)
        w_problem = Problem(manifold=w_manifold, cost=lossfn_Q)

        for em_iter in range(n_iter):
            q_start = self.Q_fun(self.W, X)
            logger.info("Iter %i, Q at start %f" % (em_iter, q_start))

            # ESTEP
            # compute all the terms with old vals
            s_prime_new, scov_prime_new, _ = self.estep(X)
            self.s_prime.assign(s_prime_new, read_value=False)
            self.scov_prime.assign(scov_prime_new, read_value=False)
            # don't assign tcov since it is not updated in margS SRM

            q_end_estep = self.Q_fun(self.W, X)
            logger.info("Iter %i, Q at estep end %f" % (em_iter, q_end_estep))

            # MSTEP

            # closed form parts
            self.b = self.mstep_b(X)
            self.rhoprec = self.mstep_rhoprec(X)

            # optimization parts:
            # Stiefel manifold for orthonormal W
            new_w = solver.solve(
                w_problem, x=[self.W[i].numpy() for i in range(self.n)])

            self.W.assign(new_w, read_value=False)

            # L-BFGS for residual covs
            for cov in [self.space_cov, self.time_cov, self.marg_cov]:
                if len(cov.get_optimize_vars()) > 0:
                    def lossfn(Q): return -self.Q_fun(self.W, X)
                    val_and_grad = make_val_and_grad(
                        lossfn, cov.get_optimize_vars())

                    x0 = pack_trainable_vars(cov.get_optimize_vars())

                    opt_results = minimize(
                        fun=val_and_grad, x0=x0, jac=True, method=self.optMethod,
                        **self.optCtrl
                    )
                    assert opt_results.success, "L-BFGS for covariances failed!"

            q_end_mstep = self.Q_fun(self.W, X)
            logger.info("Iter %i, Q at mstep end %f" % (em_iter, q_end_mstep))
            assert q_end_estep >= q_start, "Q increased in E-step!"
            assert q_end_mstep >= q_end_estep, "Q increased in M-step!"

        self.w_ = [self.W[i].numpy() for i in range(self.n)]
        self.s_ = self.s_prime.numpy()
        self.rho_ = 1/self.rhoprec.numpy()

    def transform(self, X):
        vprec = self.space_cov._prec.numpy()
        return np.array([np.linalg.solve((w.T @ vprec @ w)/r, (w.T @ vprec @ x)/r) for w, x, r in zip(self.w_, X, self.rho_)])
