import tensorflow as tf
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import TrustRegions
from sklearn.base import BaseEstimator
from brainiak.matnormal.covs import (CovIdentity,
                                     CovScaleMixin,
                                     CovUnconstrainedCholesky)
import numpy as np
from brainiak.matnormal.matnormal_likelihoods import (
    matnorm_logp_marginal_col)
from brainiak.matnormal.utils import pack_trainable_vars, make_val_and_grad
import logging
from pymanopt.function import TensorFlow
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class DPMNSRM(BaseEstimator):
    """Probabilistic SRM, aka SRM with marginalization over W (and optionally,
    orthonormal S). In contrast to SRM (Chen et al. 2015), this estimates
    far fewer parameters due to the W integral, and includes support for
    arbitrary kronecker-structured residual covariance. Inference is
    performed by ECM algorithm.
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
            raise RuntimeError(
                f"Unknown s_constraint! Expected 'ortho' or 'gaussian', got {s_constraint}!")

        if algorithm not in ["ECM", "ECME"]:
            raise RuntimeError(
                f"Unknown algorithm! Expected 'ECM' or 'ECME', got {algorithm}!")

        self.time_noise_cov_class = time_noise_cov
        self.space_noise_cov_class = space_noise_cov
        self.marg_cov_class = w_cov

        self.optCtrl, self.optMethod = optCtrl, optMethod

    def logp_margw(self, X):
        """ MatnormSRM Log-likelihood with marginal"""
        subj_space_covs = [CovScaleMixin(base_cov=self.space_cov,
                                         scale=1/self.rhoprec[j]) for j in range(self.n)]
        if self.marg_cov_class is CovIdentity:
            return tf.reduce_sum(
                input_tensor=[matnorm_logp_marginal_col(X[j],
                                                        row_cov=subj_space_covs[j],
                                                        col_cov=self.time_cov,
                                                        marg=self.S,
                                                        marg_cov=CovIdentity(size=self.k))
                              for j in range(self.n)], name="lik_logp")

        elif self.marg_cov_class is CovUnconstrainedCholesky:
            return tf.reduce_sum(
                input_tensor=[matnorm_logp_marginal_col(X[j],
                                                        row_cov=subj_space_covs[j],
                                                        col_cov=self.time_cov,
                                                        marg=tf.matmul(
                                               self.marg_cov.L, self.S),
                                           marg_cov=CovIdentity(size=self.k))
                              for j in range(self.n)], name="lik_logp")
        else:
            logger.warn("ECME with cov that is not identity or unconstrained may\
                        yield numerical instabilities! Use ECM for now.")
            return tf.reduce_sum(
                input_tensor=[matnorm_logp_marginal_col(X[j],
                                                        row_cov=subj_space_covs[j],
                                                        col_cov=self.time_cov,
                                                        marg=self.S,
                                                        marg_cov=self.marg_cov)
                              for j in range(self.n)], name="lik_logp")

    def Q_fun_margw(self, Strp, X):
        # shorthands for readability
        kt = self.k * self.t
        nv = self.n * self.v

        mean = X - self.b - tf.matmul(self.w_prime,
                                      tf.tile(tf.expand_dims(self.S, 0),
                                              [self.n, 1, 1]))

        # covs don't support batch ops (yet!) (TODO):
        x_quad_form = -tf.linalg.trace(tf.reduce_sum(
                                input_tensor=[tf.matmul(self.time_cov.solve(
                                 tf.transpose(a=mean[j])),
                                 self.space_cov.solve(mean[j])) *
                                 self.rhoprec[j]
                                 for j in range(self.n)], axis=0))

        w_quad_form = -tf.linalg.trace(tf.reduce_sum(
                                input_tensor=[tf.matmul(self.marg_cov.solve(
                                 tf.transpose(a=self.w_prime[j])),
                                 self.space_cov.solve(self.w_prime[j])) *
                                 self.rhoprec[j]
                                 for j in range(self.n)], axis=0))

        if self.s_constraint == "gaussian":
            s_quad_form = - \
                tf.linalg.trace(tf.matmul(self.time_cov.solve(
                    tf.transpose(a=self.S)), self.S))
            det_terms = -(self.v*self.n+self.k) * self.time_cov.logdet -\
                kt*self.n*self.space_cov.logdet +\
                kt*self.v*tf.reduce_sum(input_tensor=tf.math.log(self.rhoprec)) -\
                nv*self.marg_cov.logdet
        else:
            s_quad_form = 0
            det_terms = -(self.v*self.n)*self.time_cov.logdet -\
                kt*self.n*self.space_cov.logdet +\
                kt*self.v*tf.reduce_sum(input_tensor=tf.math.log(self.rhoprec)) -\
                nv*self.marg_cov.logdet

        trace_prod = -tf.reduce_sum(input_tensor=self.rhoprec / self.rhoprec_prime) *\
            tf.linalg.trace(self.space_cov.solve(self.vcov_prime)) *\
            (tf.linalg.trace(tf.matmul(self.wcov_prime, self.marg_cov._prec +
                                       tf.matmul(self.S, self.time_cov.solve(
                                           tf.transpose(a=self.S))))))

        return 0.5 * (det_terms +
                      x_quad_form +
                      w_quad_form +
                      trace_prod +
                      s_quad_form)

    def estep_margw(self, X):

        wchol = tf.linalg.cholesky(self.marg_cov._prec +
                                   tf.matmul(self.S, self.time_cov.solve(
                                       tf.transpose(a=self.S))))

        wcov_prime = tf.linalg.cholesky_solve(wchol, tf.eye(self.k, dtype=tf.float64))

        stacked_rhs = tf.tile(tf.expand_dims(self.time_cov.solve(
            tf.transpose(a=tf.linalg.cholesky_solve(wchol, self.S))), 0),
            [self.n, 1, 1])

        w_prime = tf.matmul(self.X-self.b, stacked_rhs)

        # rhoprec doesn't change
        # vcov doesn't change
        self.w_prime.assign(w_prime)
        self.wcov_prime.assign(wcov_prime)


    def mstep_b_margw(self, X):
        return tf.expand_dims(tf.reduce_sum(
                    input_tensor=[self.time_cov.solve(tf.transpose(a=X[j] -
                                                                   tf.matmul(self.w_prime[j], self.S)))
                                  for j in range(self.n)], axis=1) /
                              tf.reduce_sum(input_tensor=self.time_cov._prec), -1)

    def mstep_S_nonortho(self, X):
        wtw = tf.reduce_sum(
            input_tensor=[tf.matmul(self.w_prime[j],
                                    self.space_cov.solve(
                                        self.w_prime[j]),
                                    transpose_a=True) *
                          self.rhoprec[j] for j in range(self.n)], axis=0)

        wtx = tf.reduce_sum(
            input_tensor=[tf.matmul(self.w_prime[j],
                                    self.space_cov.solve(
                                        X[j]-self.b[j]),
                                    transpose_a=True) *
                          self.rhoprec[j] for j in range(self.n)], axis=0)

        return tf.linalg.solve(wtw + tf.reduce_sum(input_tensor=self.rhoprec_prime / self.rhoprec) *
                               tf.linalg.trace(self.space_cov.solve(self.vcov_prime)) *
                               self.wcov_prime + tf.eye(self.k, dtype=tf.float64), wtx)

    def mstep_rhoprec_margw(self, X):

        mean = X - self.b -\
            tf.matmul(self.w_prime,
                      tf.tile(tf.expand_dims(self.S, 0),
                              [self.n, 1, 1]))

        mean_trace = tf.stack(
            [tf.linalg.trace(tf.matmul(self.time_cov.solve(
                tf.transpose(a=mean[j])),
                self.space_cov.solve(mean[j]))) for j in range(self.n)])

        w_trace = tf.stack(
            [tf.linalg.trace(tf.matmul(self.marg_cov.solve(
                tf.transpose(a=self.w_prime[j])),
                self.space_cov.solve(self.w_prime[j])))
             for j in range(self.n)])

        shared_term = (1/self.rhoprec_prime) *\
            tf.linalg.trace(self.space_cov.solve(self.vcov_prime)) *\
            tf.linalg.trace(tf.matmul(self.wcov_prime,
                                      self.marg_cov._prec +
                                      tf.matmul(self.S,
                                                self.time_cov.solve(
                                                    tf.transpose(a=self.S)))))
        rho_hat_unscaled = mean_trace + w_trace + shared_term

        return (self.v*(self.k+self.t)) / rho_hat_unscaled

    def mstep_margw(self, X):
        # closed form parts
        self.b = self.mstep_b_margw(X)
        self.rhoprec = self.mstep_rhoprec_margw(X)

        # optimization parts:
        # Stiefel manifold for orthonormal S (if ortho_s)
        if self.s_constraint == "ortho":
            new_Strp = self.solver.solve(self.problem, x=self.S.numpy().T)
            self.S.assign(new_Strp.T)
        else:
            # if it's not ortho, it's just least squares update
            self.S.assign(self.mstep_S_nonortho(X))
        # L-BFGS for residual covs
        for cov in [self.space_cov, self.time_cov, self.marg_cov]:
            if len(cov.get_optimize_vars()) > 0:
                def lossfn(Q): return -self.Q_fun_margw(self.S, X)
                val_and_grad = make_val_and_grad(
                    lossfn, cov.get_optimize_vars())

                x0 = pack_trainable_vars(cov.get_optimize_vars())

                opt_results = minimize(
                    fun=val_and_grad, x0=x0, jac=True, method=self.optMethod,
                    **self.optCtrl
                )
                assert opt_results.success, "L-BFGS for covariances failed!"

    def _init_vars(self, X, svd_init=False):
        self.n = len(X)

        self.v, self.t = X[0].shape

        self.X = tf.constant(X, name="X")

        if svd_init:
            xinit = [np.linalg.svd(x) for x in X]
        else:
            xinit = [np.linalg.svd(np.random.normal(
                size=(self.v, self.t))) for i in range(self.n)]

        # parameters
        self.b = tf.Variable(np.random.normal(size=(self.n, self.v, 1)),
                             name="b")
        self.rhoprec = tf.Variable(np.ones(self.n), name="rhoprec")
        self.space_cov = self.space_noise_cov_class(size=self.v)
        self.time_cov = self.time_noise_cov_class(size=self.t)
        self.marg_cov = self.marg_cov_class(size=self.k)
        self.S = tf.Variable(np.average([s[2][:self.k, :] for s in xinit],0),
                                 dtype=tf.float64, name="S")

        # sufficient statistics
        self.w_prime = tf.Variable(np.array([s[0][:, :self.k] for s in xinit]),
                                   name="w_prime")
        self.rhoprec_prime = tf.Variable(np.ones(self.n), name="rhoprec_prime")
        self.wcov_prime = tf.Variable(np.eye(self.k), name="wcov_prime")
        self.vcov_prime = tf.Variable(np.eye(self.v), name="vcov_prime")

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

        if self.algorithm == "ECME":
            def loss(x): return -self.logp_margw(X)
            loss_name = "-Marginal Lik"
        elif self.algorithm == "ECM":
            def loss(x): return -self.Q_fun_margw(X)
            loss_name = "-ELPD (Q)"

        def wrapped_Q(Strp, X):
            return -self.Q_fun_margw(Strp, X)

        lossfn_Q = TensorFlow(wrapped_Q)

        s_trp_manifold = Stiefel(self.t, self.k)
        self.solver = TrustRegions(logverbosity=0)
        self.problem = Problem(manifold=s_trp_manifold, cost=lossfn_Q)

        for em_iter in range(max_iter):

            q_start = self.Q_fun_margw(self.S, X)
            logger.info(f"Iter {em_iter}, {loss_name} at start {q_start}")
            print(f"Iter {em_iter}, {loss_name} at start {q_start}")

            # ESTEP
            self.estep_margw(X)
            q_end_estep = self.Q_fun_margw(self.S, X)
            logger.info(f"Iter {em_iter}, {loss_name} at estep end {q_end_estep}")
            print(f"Iter {em_iter}, {loss_name} at estep end {q_end_estep}")

            # MSTEP
            self.mstep_margw(X)

            q_end_mstep = self.Q_fun_margw(self.S, X)
            logger.info("Iter %i, Q at mstep end %f" % (em_iter, q_end_mstep))
            print("Iter %i, Q at mstep end %f" % (em_iter, q_end_mstep))
            assert q_end_estep >= q_start, "Q increased in E-step!"
            assert q_end_mstep >= q_end_estep, "Q increased in M-step!"

            # converged = check_convergence()

            # Convergence checks: tol on just delta-loss or
            # we check w, rho, and sigma_v (since we
            # use them for reconstruction/projection)?

        # if converged:
        #     logger.info("Converged in %i iterations" % i)
        # else:
        #     logger.warn("Not converged to tolerance!\
        #                  Results may not be reliable")
        self.w_ = self.w_prime.numpy()
        self.s_ = self.S.numpy()
        self.rho_ = 1/self.rhoprec.numpy()

        self.final_loss_ = q_end_mstep
        self.logp_ = self.logp_margw(X)

    def _condition(self, x):
        s = np.linalg.svd(x, compute_uv=False)
        return np.max(s)/np.min(s)

    def transform(self, X, ortho_w=False):
        if ortho_w:
            w_local = [w @ np.linalg.svd(
                w.T @ w)[0] / np.sqrt(np.linalg.svd(w.T @ w)[1]) for w in self.w_]
        else:
            w_local = self.w_

        vprec_w = [self.space_cov.solve(w).numpy(
        ) / r for (w, r) in zip(w_local, self.rhoprec_)]
        vprec_x = [self.space_cov.solve(x).numpy(
        ) / r for (x, r) in zip(X, self.rhoprec_)]
        conditions = [self._condition(w.T @ vw)
                      for (w, vw) in zip(w_local, self.vprec_w)]
        logger.info(["Condition #s for transformation"] + conditions)
        return [np.linalg.solve(w.T @ vw, w.T @ vx) for (w, vw, vx) in zip(w_local, vprec_w, vprec_x)]
