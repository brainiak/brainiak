import tensorflow as tf
from sklearn.base import BaseEstimator
from brainiak.matnormal.covs import (CovIdentity,
                                     CovScaleMixin,
                                     CovUnconstrainedCholesky)
import numpy as np
from brainiak.matnormal.matnormal_likelihoods import (
    matnorm_logp_marginal_col)
from brainiak.matnormal.utils import pack_trainable_vars, make_val_and_grad
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def assert_monotonicity(fun, rtol=1e-3):
    """
    Check that the loss is monotonically decreasing
    after called function.
    tol > 0 allows for some slop due to numerics
    """
    def wrapper(classref, *args, **kwargs):
        loss_before = classref.lossfn(None)
        print(f"loss before {fun} is {loss_before}")
        res = fun(classref, *args, **kwargs)
        loss_after = classref.lossfn(None)
        print(f"loss after {fun} is {loss_after}")
        assert loss_after-loss_before <= abs(loss_before*rtol), f"loss increased on {fun}"
        return res
    return wrapper


class DPMNSRM(BaseEstimator):
    """Probabilistic SRM, aka SRM with marginalization over W (and optionally,
    orthonormal S). In contrast to SRM (Chen et al. 2015), this estimates
    far fewer parameters due to the W integral, and includes support for
    arbitrary kronecker-structured residual covariance. Inference is
    performed by ECM algorithm.
    """

    def __init__(self, n_features=5, time_noise_cov=CovIdentity,
                 space_noise_cov=CovIdentity,
                 optMethod="L-BFGS-B", optCtrl={},
                 improvement_tol=1e-5, algorithm="ECME"):

        self.k = n_features
        # self.s_constraint = s_constraint
        self.improvement_tol = improvement_tol
        self.algorithm = algorithm
        self.marg_cov_class = CovIdentity

        if algorithm not in ["ECM", "ECME"]:
            raise RuntimeError(
                f"Unknown algorithm! Expected 'ECM' or 'ECME', got {algorithm}!")

        self.time_noise_cov_class = time_noise_cov
        self.space_noise_cov_class = space_noise_cov

        self.optCtrl, self.optMethod = optCtrl, optMethod

    def logp(self, X, S=None):
        """ MatnormSRM marginal log-likelihood, integrating over W"""
        
        if S is None:
            S = self.S

        subj_space_covs = [CovScaleMixin(base_cov=self.space_cov,
                                         scale=1/self.rhoprec[j]) for j in range(self.n)]
        return tf.reduce_sum(
            input_tensor=[matnorm_logp_marginal_col(X[j],
                                                    row_cov=subj_space_covs[j],
                                                    col_cov=self.time_cov,
                                                    marg=S,
                                                    marg_cov=CovIdentity(size=self.k))
                            for j in range(self.n)], name="lik_logp")

    def Q_fun(self, X, S=None):
        
        if S is None:
            S = self.S

        # shorthands for readability
        kpt = self.k + self.t
        nv = self.n * self.v

        mean = X - self.b - tf.matmul(self.w_prime,
                                      tf.tile(tf.expand_dims(S, 0),
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

        s_quad_form = - \
            tf.linalg.trace(tf.matmul(self.time_cov.solve(
                tf.transpose(a=S)), S))
        det_terms = -(nv+self.k) * self.time_cov.logdet -\
            kpt*self.n*self.space_cov.logdet +\
            kpt*self.v*tf.reduce_sum(input_tensor=tf.math.log(self.rhoprec)) -\
            nv*self.marg_cov.logdet

        trace_prod = -tf.reduce_sum(input_tensor=self.rhoprec / self.rhoprec_prime) *\
            tf.linalg.trace(self.space_cov.solve(self.vcov_prime)) *\
            (tf.linalg.trace(tf.matmul(self.wcov_prime, self.marg_cov._prec +
                                       tf.matmul(S, self.time_cov.solve(
                                           tf.transpose(a=S))))))

        return 0.5 * (det_terms +
                      x_quad_form +
                      w_quad_form +
                      trace_prod +
                      s_quad_form)
    
    @assert_monotonicity
    def estep_margw(self, X):

        wchol = tf.linalg.cholesky(tf.eye(self.k, dtype=tf.float64) +
                                   tf.matmul(self.S, self.time_cov.solve(
                                       tf.transpose(a=self.S))))

        wcov_prime = tf.linalg.cholesky_solve(wchol, tf.eye(self.k, dtype=tf.float64))

        stacked_rhs = tf.tile(tf.expand_dims(self.time_cov.solve(
            tf.transpose(a=tf.linalg.cholesky_solve(wchol, self.S))), 0),
            [self.n, 1, 1])

        w_prime = tf.matmul(self.X-self.b, stacked_rhs)

        # rhoprec doesn't change
        # vcov doesn't change
        self.w_prime.assign(w_prime, read_value=False)
        self.wcov_prime.assign(wcov_prime, read_value=False)

    @assert_monotonicity
    def mstep_b_margw(self, X):
        resids_transpose = [tf.transpose(X[j] - self.w_prime[j] @ self.S) for j in range(self.n)]
        numerator = [tf.reduce_sum(tf.transpose(self.time_cov.solve(r)), axis=1) for r in resids_transpose]
        denominator = tf.reduce_sum(self.time_cov._prec)
        
        self.b.assign(tf.stack([n/denominator for n in numerator])[...,None], read_value=False)

    @assert_monotonicity
    def mstep_S(self, X):
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

        self.S.assign(tf.linalg.solve(wtw + tf.reduce_sum(input_tensor=self.rhoprec_prime / self.rhoprec) *
                                tf.linalg.trace(self.space_cov.solve(self.vcov_prime)) *
                                self.wcov_prime + tf.eye(self.k, dtype=tf.float64), wtx), read_value=False)

    @assert_monotonicity
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
            (tf.linalg.trace(self.marg_cov.solve(self.wcov_prime)) +
                tf.linalg.trace(self.S @ self.time_cov.solve(tf.transpose(self.S))))
        
        rho_hat_unscaled = mean_trace + w_trace + shared_term

        self.rhoprec.assign((self.v*(self.k+self.t)) / rho_hat_unscaled, read_value=False)

    @assert_monotonicity
    def mstep_covs(self):
        for cov in [self.space_cov, self.time_cov, self.marg_cov]:
            if len(cov.get_optimize_vars()) > 0:
                val_and_grad = make_val_and_grad(
                    self.lossfn, cov.get_optimize_vars())

                x0 = pack_trainable_vars(cov.get_optimize_vars())

                opt_results = minimize(
                    fun=val_and_grad, x0=x0, jac=True, method=self.optMethod,
                    **self.optCtrl
                )
                assert opt_results.success, f"L-BFGS for covariances failed with message: {opt_results.message}"

    def mstep_margw(self, X):
        # closed form parts
        self.mstep_b_margw(X)
        self.mstep_rhoprec_margw(X)
        self.mstep_S(X)

        # L-BFGS for residual covs
        self.mstep_covs()

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

    def fit(self, X, max_iter=10, y=None, svd_init=False, rtol=1e-3, gtol=1e-7):
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
        self._init_vars(X, svd_init=svd_init)

        if self.algorithm == "ECME":
            self.lossfn = lambda theta: -self.logp(X)
            loss_name = "-Marginal Lik"
        elif self.algorithm == "ECM":
            self.lossfn = lambda theta: -self.Q_fun(X)
            loss_name = "-ELPD (Q)"

        
        prevloss = self.lossfn(None)
        converged = False
        for em_iter in range(max_iter):

            logger.info(f"Iter {em_iter}, {loss_name} at start {prevloss}")
            # print(f"Iter {em_iter}, {loss_name} at start {q_start}")

            # ESTEP
            self.estep_margw(X)
            currloss = self.lossfn(None)
            logger.info(f"Iter {em_iter}, {loss_name} at estep end {currloss}")
            assert currloss - prevloss <= 0.1 , f"{loss_name} increased in E-step!"
            prevloss = currloss
            # MSTEP
            self.mstep_margw(X)

            currloss = self.lossfn(None)
            logger.info(f"Iter {em_iter}, {loss_name} at mstep end {currloss}")
            currloss = self.lossfn(None)
            assert currloss - prevloss <= 0.1, f"{loss_name} increased in M-step!"

            if prevloss - currloss < abs(rtol * prevloss):
                break
                converged = True
                converged_reason = "rtol"
            elif self._loss_gradnorm() < gtol:
                break
                converged = True
                converged_reason = "gtol"

        if converged:
            logger.info(f"Converged in {em_iter} iterations with by metric {converged_reason}")
        else:
            logger.warn("Not converged to tolerance!\
                         Results may not be reliable")
        self.w_ = self.w_prime.numpy()
        self.s_ = self.S.numpy()
        self.rho_ = 1/self.rhoprec.numpy()

        self.final_loss_ = self.lossfn(None)
        self.logp_ = self.logp(X)

    def _loss_gradnorm(self):

        params = [self.S, self.rhoprec]  +\
                    self.space_cov.get_optimize_vars() +\
                    self.time_cov.get_optimize_vars()  +\
                    self.marg_cov.get_optimize_vars()
        if self.algorithm == "ECM":
            # if ECME, marginal likelihood is independent 
            # of W sufficient statistic
            params.append(self.w_prime)

        val_and_grad = make_val_and_grad(self.lossfn, params)
        packed_params = pack_trainable_vars(params)
        _, grad = val_and_grad(packed_params)
        return np.linalg.norm(grad, np.inf)

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
