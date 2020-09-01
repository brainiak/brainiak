import tensorflow as tf
import numpy as np

from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Euclidean, Product
from pymanopt.solvers import TrustRegions, ConjugateGradient

from sklearn.base import BaseEstimator

from brainiak.matnormal.covs import CovIdentity
from brainiak.matnormal.matnormal_likelihoods import (
        matnorm_logp_marginal_col,
        matnorm_logp)
from brainiak.matnormal.utils import make_val_and_grad


import logging


logger = logging.getLogger(__name__)


class MNSRM_OrthoW(BaseEstimator):
    """Probabilistic SRM, aka SRM with marginalization over S (and ortho W)
    """

    def __init__(self, n_features=5, time_noise_cov=CovIdentity,
                 space_noise_cov=CovIdentity, s_cov=CovIdentity,
                 optMethod="L-BFGS-B", optCtrl={}):

        self.k = n_features

        self.time_noise_cov_class = time_noise_cov
        self.space_noise_cov_class = space_noise_cov
        self.marg_cov_class = s_cov

        self.optCtrl, self.optMethod = optCtrl, optMethod

    def _eye(self, x):
        return tf.linalg.tensor_diag(tf.ones((x), dtype=tf.float64))

    def Q_fun(self, Wlist, X):
        W = tf.stack(Wlist)
        print(W.shape)
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

    def estep(self):

        tcov_prime = self.time_cov
        Xmb = X - self.b

        sprec_chol = tf.linalg.cholesky(self.marg_cov._prec + tf.reduce_sum(input_tensor=[tf.matmul(tf.transpose(
            a=self.w[j]), self.space_cov.solve(self.w[j]))*self.rhoprec[j] for j in range(self.n)], axis=0))

        wsig_x = tf.reduce_sum(input_tensor=[tf.matmul(tf.transpose(
            a=self.w[j]), self.space_cov.solve(Xmb[j]))*self.rhoprec[j] for j in range(self.n)], axis=0)

        scov_prime = tf.linalg.cholesky_solve(sprec_chol, self._eye(self.k))

        s_prime = tf.linalg.cholesky_solve(sprec_chol, wsig_x)

        return s_prime, scov_prime, tcov_prime

    def mstep_b(self):

        return tf.expand_dims(tf.reduce_sum(input_tensor=[self.time_cov.solve(tf.transpose(a=X[j] -
                                                                                           tf.matmul(self.w[j], self.s_prime)))
                                                          for j in range(self.n)], axis=1) /
                              tf.reduce_sum(input_tensor=self.time_cov._prec), -1)

    def mstep_rhoprec(self):

        mean = X - self.b - \
            tf.matmul(self.w, tf.tile(tf.expand_dims(
                self.s_prime, 0), [self.n, 1, 1]))

        mean_trace = tf.stack([tf.linalg.trace(tf.matmul(self.time_cov.solve(tf.transpose(a=mean[j])),
                                                         self.space_cov.solve(mean[j]))) for j in range(self.n)])

        trace_t_t = tf.linalg.trace(self.time_cov.solve(self.tcov_prime))

        w_trace = trace_t_t * tf.stack([tf.linalg.trace(tf.matmul(tf.matmul(self.scov_prime, tf.transpose(a=self.w[j])),
                                                                  self.space_cov.solve(self.w[j]))) for j in range(self.n)])

        rho_hat_unscaled = mean_trace + w_trace

        return (self.v*self.t) / rho_hat_unscaled
import numpy as np
voxels = 100
samples = 500
subjects = 2
features = 3

# Create a Shared response S with K = 3
theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
z = np.linspace(-2, 2, samples)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

S = np.vstack((x, y, z))

X = []
W = []

for subject in range(subjects):
    Q, R = np.linalg.qr(np.random.random((voxels, features)))
    W.append(Q)
    X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

self = MNSRM_OrthoW()

    def fit(self, X, n_iter=10, y=None, w_cov=None):
        """
        find W marginalizing S

        Parameters
        ----------
        X: 2d array
            Brain data matrix (voxels by TRs). Y in the math
        n_iter: int, default=10
            Number of iterations to run
        """

        self.n = len(X)

        self.v, self.t = X[0].shape

        X = tf.stack(X, name="X")

        xsvd = [np.linalg.svd(x)for x in X]

        # parameters
        self.b = tf.Variable(np.random.normal(
            size=(self.n, self.v, 1)), name="b")
        self.rhoprec = tf.Variable(np.ones(self.n), name="rhoprec")
        wlist_np = [sv[0][:, :self.k] for sv in xsvd]
        self.wlist = [tf.Variable(_w) for _w in wlist_np]
        self.w = tf.stack(self.wlist)
        self.space_cov = self.space_noise_cov_class(size=self.v)
        self.time_cov = self.time_noise_cov_class(size=self.t)
        self.marg_cov = self.time_noise_cov_class(size=self.k)

        # sufficient statistics
        self.s_prime = tf.Variable(np.average(
            [sv[2][:self.k, :] for sv in xsvd], 0), dtype=tf.float64, name="s_prime")
        self.scov_prime = tf.Variable(np.eye(self.k), name="wcov_prime")
        self.tcov_prime = tf.Variable(np.eye(self.t), name="wcov_prime")

        # self.Lambda = tf.diag(tf.ones(self.k, dtype=tf.float64)) * 1000 # just there for the q improvement assertion check

        # s_prime_op, scov_prime_op, tcov_prime_op = self.make_estep_ops()

        # can update these guys in closed form
        # b_op = self.make_mstep_b_op()
        # rhoprec_op = self.make_mstep_rhoprec_op()

        # q_op = self._make_Q_op()

        # sigma_v_opt = ScipyOptimizerInterface(-q_op,
        #                                       var_list=self.space_cov.get_optimize_vars(),
        #                                       method=self.optMethod,
        #                                       options=self.optCtrl)

        # sigma_t_opt = ScipyOptimizerInterface(-q_op,
        #                                       var_list=self.time_cov.get_optimize_vars(),
        #                                       method=self.optMethod,
        #                                       options=self.optCtrl)

        # sigma_s_opt = ScipyOptimizerInterface(-q_op,
        #                                       var_list=self.marg_cov.get_optimize_vars(),
        #                                       method=self.optMethod,
        #                                       options=self.optCtrl)

        w_manifold = Product([Stiefel(self.t, self.k) for i in range(self.n)])
        # s_trp_manifold = Euclidean(self.t, self.k)
        # solver = ConjugateGradient()
        # this would be faster but need to work through some dtype wrangling with
        # the internals of pymanopt
        solver = TrustRegions()

        @TensorFlow
        def lossfn_Q(arg1, arg2):
            print(arg1.shape)
            print(arg2.shape)
            return -self.Q_fun([arg1, arg2], X)

        egrad = lossfn_Q.compute_gradient()
        egrad(*[w.numpy() for w in self.wlist])

        ehess = lossfn_Q.compute_hessian_vector_product()
        ehess(*[w.numpy() for w in self.wlist], *[np.ones(self.v) for i in range(self.n)])

        # val_and_grad = make_val_and_grad(lossfn_Q, self.wlist)
        # x0 = pack_trainable_vars(self.train_variables)

        # opt_results = minimize(
        #     fun=val_and_grad, x0=x0, jac=True, method=self.optMethod,
        #     **self.optCtrl
        # )


        w_problem = Problem(manifold=w_manifold, cost=lossfn_Q)#  verbosity=0) for i in range(self.n)]
        em_iter = 0
        # for em_iter in range(n_iter):
            q_start = self.Q_fun(self.wlist, X)
            logger.info("Iter %i, Q at start %f" % (em_iter, q_start))

            # ESTEP
            # compute all the terms with old vals
            s_prime_new, scov_prime_new, tcov_prime_new = self.estep()

            q_end_estep = self.Q_fun(self.wlist, X)
            logger.info("Iter %i, Q at estep end %f" % (em_iter, q_end_estep))

            # MSTEP
            self.b = self.mstep_b()
            rhoprec_new = self.mstep_rhoprec()
            # rhoprec_norm = tf.norm(rhoprec_new - self.rhoprec).eval(session=self.sess) / self.n

            # optimization parts:

            solver.solve(w_problem, x=[w.numpy() for w in self.wlist])

            i = 0
            for i in range(self.n):
                new_w = solver.solve(
                    w_problems[i], x=self.wlist[i].numpy())
                self.wlist[i].load(new_w, session=self.sess)

            if self.space_noise_cov_class is not CovIdentity:
                sigma_v_opt.minimize(session=self.sess)

            if self.time_noise_cov_class is not CovIdentity:
                sigma_t_opt.minimize(session=self.sess)

            if self.marg_cov_class is not CovIdentity:
                sigma_s_opt.minimize(session=self.sess)

            q_end_mstep = q_op.eval(session=self.sess)
            logger.info("Iter %i, Q at mstep end %f" % (em_iter, q_end_mstep))
            assert q_end_estep >= q_start
            assert q_end_mstep >= q_end_estep

        self.w_ = self.w.eval(session=self.sess)
        self.s_ = self.s_prime.eval(session=self.sess)
        self.rho_ = 1/self.rhoprec.eval(session=self.sess)

    def transform(self, X):
        vprec = self.space_cov._prec.eval(session=self.sess)
        return np.array([np.linalg.solve((w.T @ vprec @ w)/r, (w.T @ vprec @ x)/r) for w, x, r in zip(self.w_, X, self.rho_)])

    def transform_orthow(self, X):
        # orthonormalize W
        w_ortho = [w @ np.linalg.svd(
            w.T @ w)[0] / np.sqrt(np.linalg.svd(w.T @ w)[1]) for w in self.w_]
        vprec = self.space_cov._prec.eval(session=self.sess)
        return np.array([np.linalg.solve((w.T @ vprec @ w)/r, (w.T @ vprec @ x)/r) for w, x, r in zip(self.w_, X, self.rho_)])
