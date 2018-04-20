import tensorflow as tf
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Euclidean
from pymanopt.solvers import TrustRegions, ConjugateGradient
from sklearn.base import BaseEstimator
from brainiak.matnormal.covs import (CovIdentity,
                                     CovScaleMixin,
                                     CovTFWrap,
                                     CovUnconstrainedCholesky)
import numpy as np
from brainiak.matnormal.matnormal_likelihoods import (
    matnorm_logp_marginal_col, matnorm_logp)
from tensorflow.contrib.opt import ScipyOptimizerInterface
import logging


logger = logging.getLogger(__name__)


class DPMNSRM(BaseEstimator):
    """Dual probabilistic SRM, aka SRM with marginalization over W
    """

    def __init__(self, n_features=5, time_noise_cov=CovIdentity,
                 space_noise_cov=CovIdentity, w_cov=CovIdentity,
                 s_constraint="gaussian", optMethod="L-BFGS-B", optCtrl={},
                 improvement_tol=1e-5, algorithm="ECM"):

        self.k = n_features
        self.s_constraint = s_constraint
        self.improvement_tol = improvement_tol
        self.algorithm = algorithm
        if s_constraint == "ortho":
            logger.info("Orthonormal S selected")
        elif s_constraint == "gaussian":
            logger.info("Gaussian S selected")
            if w_cov is not CovIdentity:
                logger.warn("Gaussian S means w_cov can be I w.l.o.g., using\
                 more general covs not recommended")
        else:
            logger.error("Unknown s_constraint! Defaulting to orthonormal.")
            self.s_constraint = "ortho"

        self.time_noise_cov_class = time_noise_cov
        self.space_noise_cov_class = space_noise_cov
        self.marg_cov_class = w_cov

        self.optCtrl, self.optMethod = optCtrl, optMethod

        # create a tf session we reuse for this object
        self.sess = tf.Session()

    def _eye(self, x):
        return tf.diag(tf.ones((x), dtype=tf.float64))

    def _make_logp_op(self):
        """ MatnormSRM Log-likelihood"""
        subj_space_covs = [CovScaleMixin(base_cov=self.space_cov,
                           scale=1/self.rhoprec[j]) for j in range(self.n)]
        if self.marg_cov_class is CovIdentity:
            return tf.reduce_sum(
                [matnorm_logp_marginal_col(self.X[j],
                 row_cov=subj_space_covs[j],
                 col_cov=self.time_cov,
                 marg=self.S,
                 marg_cov=CovIdentity(size=self.k))
                 for j in range(self.n)], name="lik_logp")

        elif self.marg_cov_class is CovUnconstrainedCholesky:
            return tf.reduce_sum(
                [matnorm_logp_marginal_col(self.X[j],
                 row_cov=subj_space_covs[j],
                 col_cov=self.time_cov,
                 marg=tf.matmul(self.marg_cov.L, self.S),
                 marg_cov=CovIdentity(size=self.k))
                 for j in range(self.n)], name="lik_logp")
        else:
            logger.warn("ECME with cov that is not identity or unconstrained may\
                        yield numerical instabilities! Use ECM for now.")
            return tf.reduce_sum(
                [matnorm_logp_marginal_col(self.X[j],
                 row_cov=subj_space_covs[j],
                 col_cov=self.time_cov,
                 marg=self.S,
                 marg_cov=self.marg_cov)
                 for j in range(self.n)], name="lik_logp")

    def _make_Q_op(self):

        mean = self.X - self.b - tf.matmul(self.w_prime,
                                           tf.tile(tf.expand_dims(self.S, 0),
                                                   [self.n, 1, 1]))

        # covs don't support batch ops (yet!) (TODO):
        x_quad_form = -tf.trace(tf.reduce_sum(
                                [tf.matmul(self.time_cov.Sigma_inv_x(
                                 tf.transpose(mean[j])),
                                 self.space_cov.Sigma_inv_x(mean[j])) *
                                 self.rhoprec[j]
                                 for j in range(self.n)], 0))

        w_quad_form = -tf.trace(tf.reduce_sum(
                                [tf.matmul(self.marg_cov.Sigma_inv_x(
                                 tf.transpose(self.w_prime[j])),
                                 self.space_cov.Sigma_inv_x(self.w_prime[j])) *
                                 self.rhoprec[j]
                                 for j in range(self.n)], 0))

        if self.s_constraint == "gaussian":
            s_quad_form = -tf.trace(tf.matmul(self.time_cov.Sigma_inv_x(tf.transpose(self.S)), self.S))
            det_terms = -(self.v*self.n+self.k) * self.time_cov.logdet -\
                (self.k+self.t)*self.n*self.space_cov.logdet +\
                (self.k+self.t)*self.v*tf.reduce_sum(tf.log(self.rhoprec)) -\
                (self.n*self.v)*self.marg_cov.logdet
        else:
            s_quad_form = 0
            det_terms = -(self.v*self.n)*self.time_cov.logdet -\
                (self.k+self.t)*self.n*self.space_cov.logdet +\
                (self.k+self.t)*self.v*tf.reduce_sum(tf.log(self.rhoprec)) -\
                (self.n*self.v)*self.marg_cov.logdet

        trace_prod = -tf.reduce_sum(self.rhoprec / self.rhoprec_prime) *\
            tf.trace(self.space_cov.Sigma_inv_x(self.vcov_prime)) *\
            (tf.trace(tf.matmul(self.wcov_prime, self.marg_cov.Sigma_inv +
             tf.matmul(self.S, self.time_cov.Sigma_inv_x(
                tf.transpose(self.S))))))

        return 0.5 * (det_terms +
                      x_quad_form +
                      w_quad_form +
                      trace_prod +
                      s_quad_form)

    def make_estep_ops(self):

        rhoprec_prime = self.rhoprec
        vcov_prime = self.space_cov.Sigma
        wchol = tf.cholesky(self.marg_cov.Sigma_inv +
                            tf.matmul(self.S, self.time_cov.Sigma_inv_x(
                                tf.transpose(self.S))))

        wcov_prime = tf.cholesky_solve(wchol, self._eye(self.k))

        stacked_rhs = tf.tile(tf.expand_dims(self.time_cov.Sigma_inv_x(
            tf.transpose(tf.cholesky_solve(wchol, self.S))), 0),
            [self.n, 1, 1])

        w_prime = tf.matmul(self.X-self.b, stacked_rhs)

        return w_prime, rhoprec_prime, vcov_prime, wcov_prime

    def make_mstep_b_op(self):
        return tf.expand_dims(tf.reduce_sum(
                    [self.time_cov.Sigma_inv_x(tf.transpose(self.X[j] -
                     tf.matmul(self.w_prime[j], self.S)))
                     for j in range(self.n)], 1) /
              tf.reduce_sum(self.time_cov.Sigma_inv), -1)

    def make_mstep_S_op(self):
        wtw = tf.reduce_sum(
            [tf.matmul(self.w_prime[j],
                       self.space_cov.Sigma_inv_x(self.w_prime[j]),
                       transpose_a=True) *
             self.rhoprec[j] for j in range(self.n)], 0)

        wtx = tf.reduce_sum(
            [tf.matmul(self.w_prime[j],
                       self.space_cov.Sigma_inv_x(self.X[j]-self.b[j]),
                       transpose_a=True) *
             self.rhoprec[j] for j in range(self.n)], 0)

        return tf.matrix_solve(wtw +
                               tf.reduce_sum(self.rhoprec /
                                             self.rhoprec_prime) *
                               tf.trace(self.space_cov.Sigma_inv_x(
                                 self.vcov_prime)) *
                               self.wcov_prime + self._eye(self.k), wtx)

    def make_mstep_rhoprec_op(self):

        mean = self.X - self.b -\
            tf.matmul(self.w_prime,
                      tf.tile(tf.expand_dims(self.S, 0),
                              [self.n, 1, 1]))

        mean_trace = tf.stack(
            [tf.trace(tf.matmul(self.time_cov.Sigma_inv_x(
                tf.transpose(mean[j])),
                self.space_cov.Sigma_inv_x(mean[j]))) for j in range(self.n)])

        w_trace = tf.stack(
            [tf.trace(tf.matmul(self.marg_cov.Sigma_inv_x(
                tf.transpose(self.w_prime[j])),
                self.space_cov.Sigma_inv_x(self.w_prime[j])))
             for j in range(self.n)])

        shared_term = (1/self.rhoprec_prime) *\
            tf.trace(self.space_cov.Sigma_inv_x(self.vcov_prime)) *\
            tf.trace(tf.matmul(self.wcov_prime,
                               self.marg_cov.Sigma_inv +
                               tf.matmul(self.S,
                                         self.time_cov.Sigma_inv_x(
                                            tf.transpose(self.S)))))
        rho_hat_unscaled = mean_trace + w_trace + shared_term

        return (self.v*(self.k+self.t)) / rho_hat_unscaled

    def _init_vars(self, X):
        self.n = len(X)

        self.v, self.t = X[0].shape

        self.X = tf.constant(X, name="X")

        xsvd = [np.linalg.svd(x)for x in X]

        # parameters
        self.b = tf.Variable(np.random.normal(size=(self.n, self.v, 1)),
                             name="b")
        self.rhoprec = tf.Variable(np.ones(self.n), name="rhoprec")

        self.w_prime = tf.Variable(np.array([s[0][:, :self.k] for s in xsvd]),
                                   name="w_prime")
        self.rhoprec_prime = tf.Variable(np.ones(self.n), name="rhoprec_prime")
        self.wcov_prime = tf.Variable(np.eye(self.k), name="wcov_prime")
        self.vcov_prime = tf.Variable(np.eye(self.v), name="vcov_prime")

        self.space_cov = self.space_noise_cov_class(size=self.v)
        self.time_cov = self.time_noise_cov_class(size=self.t)
        self.marg_cov = self.marg_cov_class(size=self.k)

        # we need Strp to be the actual param because stiefel is on the rows,
        # and might as well initialize with SVD

        self.S_trp = tf.Variable(np.average([s[2][:self.k, :] for s in xsvd],
                                            0).T,
                                 dtype=tf.float64, name="S_transpose")
        self.S = tf.transpose(self.S_trp)

    def fit(self, X, max_iter=10, y=None, convergence_tol=1e-3):
        """
        find S marginalizing W

        Parameters
        ----------
        X: 2d array
            Brain data matrix (voxels by TRs). Y in the math
        n_iter: int, default=10
            Max iterations to run
        """

        # in case we get a list, and/or int16s or float32s
        X = np.array(X).astype(np.float64)
        self._init_vars(X)

        (w_prime_op,
         rhoprec_prime_op,
         vcov_prime_op,
         wcov_prime_op) = self.make_estep_ops()

        b_op = self.make_mstep_b_op()
        rhoprec_op = self.make_mstep_rhoprec_op()

        s_op = self.make_mstep_S_op()

        if self.algorithm == "ECME":
            loss_op = -self._make_logp_op()
            loss_name = "-Marginal Lik"
        elif self.algorithm == "ECM":
            loss_op = -self._make_Q_op()
            loss_name = "-ELPD (Q)"
        else:
            logger.error("Unknown algorithm %s!" % self.algorithm)

        sigma_v_opt = ScipyOptimizerInterface(loss_op,
                                              var_list=self.space_cov.get_optimize_vars(),
                                              method=self.optMethod,
                                              options=self.optCtrl)

        sigma_t_opt = ScipyOptimizerInterface(loss_op,
                                              var_list=self.time_cov.get_optimize_vars(),
                                              method=self.optMethod,
                                              options=self.optCtrl)

        sigma_w_opt = ScipyOptimizerInterface(loss_op,
                                              var_list=self.marg_cov.get_optimize_vars(),
                                              method=self.optMethod,
                                              options=self.optCtrl)

        s_trp_manifold = Stiefel(self.t, self.k)
        solver = ConjugateGradient()

        problem = Problem(manifold=s_trp_manifold, cost=loss_op,
                          arg=self.S_trp, verbosity=1)

        # hacky hack hack to let us maintain state on the things
        # we're not pymanopting
        problem.backend._session = self.sess

        self.sess.run(tf.global_variables_initializer())

        converged = False
        for i in range(max_iter):
            loss_start = loss_op.eval(session=self.sess)
            logger.info("Iter %i, %s at start %f" % (i, loss_name, loss_start))

            # ESTEP
            # compute all the terms with old vals
            w_prime_new = w_prime_op.eval(session=self.sess)
            rhoprec_prime_new = rhoprec_prime_op.eval(session=self.sess)
            wcov_prime_new = wcov_prime_op.eval(session=self.sess)
            vcov_prime_new = vcov_prime_op.eval(session=self.sess)

            # for convergence, we check w, rho, and sigma_v (since we
            # use them for reconstruction/projection)
            w_norm = tf.norm(w_prime_new - self.w_prime).eval(
                session=self.sess) / (self.n*self.v*self.k)
            # update (since we reuse wcov_prime in computing w_prime)
            self.w_prime.load(w_prime_new, session=self.sess)
            self.rhoprec_prime.load(rhoprec_prime_new, session=self.sess)
            self.wcov_prime.load(wcov_prime_new, session=self.sess)
            self.vcov_prime.load(vcov_prime_new, session=self.sess)

            loss_end_estep = loss_op.eval(session=self.sess)
            logger.info("Iter %i, %s at estep end %f" %
                        (i, loss_name, loss_end_estep))

            # MSTEP
            self.b.load(b_op.eval(session=self.sess), session=self.sess)

            rhoprec_new = rhoprec_op.eval(session=self.sess)
            rhoprec_norm = tf.norm(rhoprec_new - self.rhoprec).eval(
                session=self.sess) / self.n
            self.rhoprec.load(rhoprec_new, session=self.sess)

            if self.s_constraint == "gaussian":
                s_hat = s_op.eval(session=self.sess).T
            elif self.s_constraint == "ortho":
                if i == 0:
                    # initial guess it the least squares op
                    s_hat = solver.solve(problem, x=s_op.eval(
                        session=self.sess).T)
                else:
                    s_hat = solver.solve(problem, x=self.S_trp.eval(
                        session=self.sess))

            self.S_trp.load(s_hat, session=self.sess)

            old_sigma_v = self.space_cov.Sigma.eval(session=self.sess)

            if self.space_noise_cov_class is not CovIdentity:
                sigma_v_opt.minimize(session=self.sess)

            sigv_norm = tf.norm(old_sigma_v - self.space_cov.Sigma).eval(
                session=self.sess) / (self.v**2)

            if self.time_noise_cov_class is not CovIdentity:
                sigma_t_opt.minimize(session=self.sess)

            if self.marg_cov_class is not CovIdentity:
                sigma_w_opt.minimize(session=self.sess)

            loss_end_mstep = loss_op.eval(session=self.sess)
            logger.info("Iter %i, %s at mstep end %f" %
                        (i, loss_name, loss_end_mstep))
            if loss_end_estep > loss_start:
                logger.warn("Warning! estep did not improve loss!\
                             Instead, worsened by %f" %
                            (loss_start-loss_end_estep))
            if loss_end_estep > loss_start:
                logger.warn("Warning! mstep did not improve loss!\
                             Instead, worsened by %f" %
                            (loss_end_estep-loss_end_mstep))

            logger.info("Iter %i end, W norm %f, sigV norm %f,\
                        rhoprec norm %f" %
                        (i, w_norm, sigv_norm, rhoprec_norm))

            delQ = loss_end_mstep - loss_start
            if np.max(np.r_[w_norm, sigv_norm,
                      rhoprec_norm, delQ]) <= convergence_tol:
                converged = True
                break

        if converged:
            logger.info("Converged in %i iterations" % i)
        else:
            logger.warn("Not converged to tolerance!\
                         Results may not be reliable")
        self.w_ = self.w_prime.eval(session=self.sess)
        self.s_ = self.S.eval(session=self.sess)
        self.rho_ = 1/self.rhoprec.eval(session=self.sess)

        self.final_loss_ = loss_op.eval(session=self.sess)
        self.logp_ = self._make_logp_op().eval(session=self.sess)

    def _condition(self, x):
        s = np.linalg.svd(x, compute_uv=False)
        return np.max(s)/np.min(s)

    def transform(self, X):
        vprec = self.space_cov.Sigma_inv.eval(session=self.sess)
        conditions = [self._condition((w.T @ vprec @ w)/r)
                      for (w, r) in zip(self.w_, self.rho_)]
        logger.info(["Condition #s for transformation"] + conditions)
        return [np.linalg.solve((w.T @ vprec @ w)/r, (w.T @ vprec @ x)/r)
                for w, x, r in zip(self.w_, X, self.rho_)]

    def transform_orthow(self, X):
        # orthonormalize W
        w_ortho = [w @ np.linalg.svd(w.T @ w)[0] /
                   np.sqrt(np.linalg.svd(w.T @ w)[1])
                   for w in self.w_]
        vprec = self.space_cov.Sigma_inv.eval(session=self.sess)
        conditions = [self._condition((w.T @ vprec @ w)/r)
                      for (w, r) in zip(self.w_, self.rho_)]
        logger.info(["Condition #s for transformation"] + conditions)
        return [np.linalg.solve((w.T @ vprec @ w)/r, (w.T @ vprec @ x)/r)
                for w, x, r in zip(self.w_, X, self.rho_)]
