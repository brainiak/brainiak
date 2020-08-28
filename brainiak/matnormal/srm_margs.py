import tensorflow as tf
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Euclidean
from pymanopt.solvers import TrustRegions, ConjugateGradient
from sklearn.base import BaseEstimator
from brainiak.matnormal.covs import CovIdentity
import numpy as np
from brainiak.matnormal.matnormal_likelihoods import (
        matnorm_logp_marginal_col,
        matnorm_logp)
from tensorflow.contrib.opt import ScipyOptimizerInterface
import logging


logger = logging.getLogger(__name__)

class MNSRM_OrthoW(BaseEstimator):
    """Probabilistic SRM, aka SRM with marginalization over S (and ortho W)
    """

    def __init__(self, n_features=5, time_noise_cov=CovIdentity,
                 space_noise_cov=CovIdentity, s_cov=CovIdentity,
                 optMethod="L-BFGS-B",optCtrl={}):

        self.k = n_features

        self.time_noise_cov_class = time_noise_cov
        self.space_noise_cov_class = space_noise_cov
        self.marg_cov_class = s_cov

        self.optCtrl, self.optMethod = optCtrl, optMethod

        # create a tf session we reuse for this object
        self.sess = tf.compat.v1.Session()

    def _eye(self, x):
        return tf.linalg.tensor_diag(tf.ones((x), dtype=tf.float64))

    def _make_Q_op(self):
        mean = self.X - self.b - tf.matmul(self.w, tf.tile(tf.expand_dims(self.s_prime, 0), [self.n, 1, 1]) )

        det_terms = -(self.v*self.n + self.k)*self.time_cov.logdet -\
                     (self.t*self.n)*self.space_cov.logdet -\
                    self.t*self.marg_cov.logdet +\
                     (self.t*self.v)*tf.reduce_sum(input_tensor=tf.math.log(self.rhoprec))

        # used twice below
        trace_t_t = tf.linalg.trace(self.time_cov.Sigma_inv_x(self.tcov_prime))

        # covs don't support batch ops (yet!) (TODO):
        x_quad_form = -tf.linalg.trace(tf.reduce_sum(input_tensor=[tf.matmul(self.time_cov.Sigma_inv_x(tf.transpose(a=mean[j])),
                                                         self.space_cov.Sigma_inv_x(mean[j]))*self.rhoprec[j]
                                               for j in range(self.n)], axis=0))

        w_quad_form = -tf.linalg.trace(tf.reduce_sum(input_tensor=[tf.matmul(tf.matmul(self.scov_prime, tf.transpose(a=self.w[j])),
                                                         self.space_cov.Sigma_inv_x(self.w[j]))*self.rhoprec[j]
                                               for j in range(self.n)], axis=0)) * trace_t_t

        s_quad_form = -tf.linalg.trace(tf.matmul(self.time_cov.Sigma_inv_x(tf.transpose(a=self.s_prime)),
                                                         self.marg_cov.Sigma_inv_x(self.s_prime)))

        sig_trace_prod = -trace_t_t * tf.linalg.trace(self.marg_cov.Sigma_inv_x(self.scov_prime))

        return 0.5 * (det_terms + x_quad_form + s_quad_form + w_quad_form + sig_trace_prod)#, det_terms, x_quad_form, s_quad_form, w_quad_form, sig_trace_prod

    def make_estep_ops(self):

        tcov_prime = self.time_cov.Sigma
        Xmb = self.X - self.b

        sprec_chol = tf.linalg.cholesky(self.marg_cov.Sigma_inv + tf.reduce_sum(input_tensor=[tf.matmul(tf.transpose(a=self.w[j]), self.space_cov.Sigma_inv_x(self.w[j]))*self.rhoprec[j] for j in range(self.n)], axis=0))

        wsig_x = tf.reduce_sum(input_tensor=[tf.matmul(tf.transpose(a=self.w[j]), self.space_cov.Sigma_inv_x(Xmb[j]))*self.rhoprec[j] for j in range(self.n)], axis=0)

        scov_prime = tf.linalg.cholesky_solve(sprec_chol, self._eye(self.k))

        s_prime = tf.linalg.cholesky_solve(sprec_chol, wsig_x)

        return s_prime, scov_prime, tcov_prime

    def make_mstep_b_op(self):

        return tf.expand_dims(tf.reduce_sum(input_tensor=[self.time_cov.Sigma_inv_x(tf.transpose(a=self.X[j] -
                                             tf.matmul(self.w[j],self.s_prime)))
                                             for j in range(self.n)], axis=1) /
                              tf.reduce_sum(input_tensor=self.time_cov.Sigma_inv), -1)

    def make_mstep_rhoprec_op(self):

        mean = self.X - self.b - tf.matmul(self.w, tf.tile(tf.expand_dims(self.s_prime,0), [self.n, 1, 1]) )

        mean_trace = tf.stack([tf.linalg.trace(tf.matmul(self.time_cov.Sigma_inv_x(tf.transpose(a=mean[j])),
                                        self.space_cov.Sigma_inv_x(mean[j]))) for j in range(self.n)])

        trace_t_t = tf.linalg.trace(self.time_cov.Sigma_inv_x(self.tcov_prime))

        w_trace = trace_t_t * tf.stack([tf.linalg.trace(tf.matmul(tf.matmul(self.scov_prime, tf.transpose(a=self.w[j])),
                                        self.space_cov.Sigma_inv_x(self.w[j]))) for j in range(self.n)])

        rho_hat_unscaled = mean_trace + w_trace 

        return (self.v*self.t) / rho_hat_unscaled

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

        self.X = tf.constant(X, name="X")

        xsvd = [np.linalg.svd(x)for x in X]

        # parameters
        self.b = tf.Variable(np.random.normal(size=(self.n, self.v,1)), name="b")
        self.rhoprec = tf.Variable(np.ones(self.n), name="rhoprec")
        wlist_np = [sv[0][:,:self.k] for sv in xsvd]
        self.wlist = [tf.Variable(_w) for _w in wlist_np]
        self.w = tf.stack(self.wlist)
        self.space_cov = self.space_noise_cov_class(size=self.v)
        self.time_cov = self.time_noise_cov_class(size=self.t)
        self.marg_cov = self.time_noise_cov_class(size=self.k)

        # sufficient statistics
        self.s_prime = tf.Variable(np.average([sv[2][:self.k, :]  for sv in xsvd], 0), dtype=tf.float64, name="s_prime")
        self.scov_prime =  tf.Variable(np.eye(self.k), name="wcov_prime")
        self.tcov_prime =  tf.Variable(np.eye(self.t), name="wcov_prime")

        # self.Lambda = tf.diag(tf.ones(self.k, dtype=tf.float64)) * 1000 # just there for the q improvement assertion check

        s_prime_op, scov_prime_op, tcov_prime_op = self.make_estep_ops()

        # can update these guys in closed form
        b_op = self.make_mstep_b_op()
        rhoprec_op = self.make_mstep_rhoprec_op()

        q_op = self._make_Q_op()

        sigma_v_opt = ScipyOptimizerInterface(-q_op,
                                              var_list=self.space_cov.get_optimize_vars(),
                                              method=self.optMethod,
                                              options=self.optCtrl)

        sigma_t_opt = ScipyOptimizerInterface(-q_op,
                                              var_list=self.time_cov.get_optimize_vars(),
                                              method=self.optMethod,
                                              options=self.optCtrl)

        sigma_s_opt = ScipyOptimizerInterface(-q_op,
                                              var_list=self.marg_cov.get_optimize_vars(),
                                              method=self.optMethod,
                                              options=self.optCtrl)

        w_manifold = Stiefel(self.t, self.k)
        # s_trp_manifold = Euclidean(self.t, self.k)
        solver = ConjugateGradient()
        # this would be faster but need to work through some dtype wrangling with
        # the internals of pymanopt
        # solver = TrustRegions()

        w_problems = [Problem(manifold=w_manifold, cost=-q_op, arg=_w, verbosity=0) for _w in self.wlist]

        # hacky hack hack to let us maintain state on the things we're not pymanopting
        for i in range(self.n):
            w_problems[i].backend._session = self.sess

        self.sess.run(tf.compat.v1.global_variables_initializer())

        for em_iter in range(n_iter):
            q_start = q_op.eval(session=self.sess)
            logger.info("Iter %i, Q at start %f" % (em_iter, q_start))

            # ESTEP
            # compute all the terms with old vals
            s_prime_new = s_prime_op.eval(session=self.sess)
            tcov_prime_new = tcov_prime_op.eval(session=self.sess)
            scov_prime_new = scov_prime_op.eval(session=self.sess)

            # then update (since we reuse wcov_prime in computing w_prime)
            self.s_prime.load(s_prime_new, session=self.sess)
            self.scov_prime.load(scov_prime_new, session=self.sess)
            self.tcov_prime.load(tcov_prime_new, session=self.sess)

            q_end_estep = q_op.eval(session=self.sess)
            logger.info("Iter %i, Q at estep end %f" % (em_iter, q_end_estep))

            # MSTEP
            # analytic parts: b and rho! that's sort of bad actually
            self.b.load(b_op.eval(session=self.sess), session=self.sess)
            rhoprec_new = rhoprec_op.eval(session=self.sess)
            # rhoprec_norm = tf.norm(rhoprec_new - self.rhoprec).eval(session=self.sess) / self.n
            self.rhoprec.load(rhoprec_new, session=self.sess)
            
            # optimization parts:
            for i in range(self.n):
                new_w = solver.solve(w_problems[i], x=self.wlist[i].eval(session=self.sess))
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
        vprec = self.space_cov.Sigma_inv.eval(session=self.sess)
        return np.array([np.linalg.solve((w.T @ vprec @ w)/r, (w.T @ vprec @ x)/r) for w, x, r in zip(self.w_, X, self.rho_)])

    def transform_orthow(self, X):
        # orthonormalize W
        w_ortho = [w @ np.linalg.svd(w.T @ w)[0] / np.sqrt(np.linalg.svd(w.T @ w)[1]) for w in self.w_]
        vprec = self.space_cov.Sigma_inv.eval(session=self.sess)
        return np.array([np.linalg.solve((w.T @ vprec @ w)/r, (w.T @ vprec @ x)/r) for w, x, r in zip(self.w_, X, self.rho_)])