import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator
from tensorflow.contrib.distributions import Normal, MultivariateNormalCholesky, Bernoulli
from .helpers import define_scope, xx_t, x_tx, scaled_I, quad_form

class MatnormModelBase(BaseEstimator):
    """ Base class implementing shared functionality for matrix-variate normal 
     models in tensorflow. 

    Not intended to be used directly -- contains a wrapper for the TF optimizers
    that does convergence checks, plus likelihood for mnorm (incl. with marginal
    and conditional densities). 
    """

    def _optimize_impl(self, optfun, loss, optvars, feed_dict, max_iter, step, loss_tol, grad_tol):
        """Implementation method for optimization that does things like convergence checks and output
        """

        grad = tf.concat(0, [tf.reshape(g[0], [-1]) for g in self.optimizer.compute_gradients(loss, optvars)])

        max_abs_current_grad_op = tf.reduce_max(tf.abs(grad))

        past_loss = 0 
        current_loss = self.sess.run(loss, feed_dict=feed_dict)    

        for n in range(max_iter):
            self.sess.run(optfun, feed_dict=feed_dict)
            past_loss = current_loss
            current_loss = self.sess.run(loss, feed_dict=feed_dict)
            max_abs_current_grad = self.sess.run(max_abs_current_grad_op, feed_dict=feed_dict)
            
            # check tolerances
            if abs((current_loss - past_loss)/max(current_loss, past_loss)) < loss_tol: 
                print('loss tolerance reached on iter %i, %f, stopping' % (n+1, current_loss ))
                break 
            if max_abs_current_grad < grad_tol: 
                print('gradient tolerance reached on iter %i, %f, stopping' % (n+1, current_loss ))
                break 
            if (n+1) % step == 0: 
                print('iter %i, %f' % (n+1, current_loss ))


    def solve_det_marginal(self, x, sigma, A, Q):
        """(Sigma + AQA')^{-1} X = Sigma^{-1} - Sigma^{-1} A (Q^{-1} + A' Sigma^{-1} A)^{-1} A' Sigma^{-1}

        Likewise with determinant: 
        log|(Sigma + AQA')| = log|Q^{-1} + A' Sigma^{-1} A| + log|Q| + log|Sigma|
        """

        
        # cholesky of (Qinv + A' Sigma^{-1} A)
        i_qf_cholesky = tf.cholesky(Q.Sigma_inv + tf.matmul(A, sigma.Sigma_inv_x(A), transpose_a=True))

        logdet = Q.logdet + sigma.logdet + 2 * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(i_qf_cholesky))))

        # A' Sigma^{-1}
        Atrp_Sinv = tf.matmul(A, sigma.Sigma_inv, transpose_a = True)
        # (Qinv + A' Sigma^{-1} A)^{-1} A' Sigma^{-1}
        prod_term = tf.cholesky_solve(i_qf_cholesky, Atrp_Sinv)
        solve = tf.matmul(sigma.Sigma_inv - sigma.Sigma_inv_x(tf.matmul(A, prod_term)), x)
        return solve, logdet

    def solve_det_conditional(self, x, sigma, A, Q):
        """(Sigma - AQ^{-1}A')^{-1} X = Sigma^{-1} + Sigma^{-1} A (Q - A' Sigma^{-1} A)^{-1} A' Sigma^{-1}

        Likewise with determinant: 
        log|(Sigma - AQ^{-1}A')| = log|Q - A' Sigma^{-1} A| + log|Q^{-1}| + log|Sigma| 
        = log|Q - A' Sigma^{-1} A| - log|Q| + log|Sigma| 
        """
        
        # (Qinv - A' Sigma^{-1} A)
        i_qf_cholesky = tf.cholesky(Q.Sigma - tf.matmul(A, sigma.Sigma_inv_x(A), transpose_a=True))

        logdet = -Q.logdet + sigma.logdet + 2 * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(i_qf_cholesky)))) 

        # A' Sigma^{-1}
        Atrp_Sinv = tf.matmul(A, sigma.Sigma_inv, transpose_a = True)
        # (Q - A' Sigma^{-1} A)^{-1} A' Sigma^{-1}
        prod_term = tf.cholesky_solve(i_qf_cholesky, Atrp_Sinv)
        
        solve = tf.matmul(sigma.Sigma_inv + sigma.Sigma_inv_x(tf.matmul(A, prod_term)), x)

        return solve, logdet

    def _mnorm_logp_internal(self, colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col):
        """Construct logp from the solves and determinants. 
        """
        log2pi = 1.8378770664093453

        denominator = - rowsize * colsize * log2pi - colsize * logdet_row - rowsize * logdet_col
        numerator = - tf.trace(tf.matmul(solve_col, solve_row))
        return tf.reduce_sum(0.5 * (numerator + denominator))        

    def matnorm_logp(self, x, row_cov, col_cov):
        """Log likelihood for centered matrix-variate normal density. Assumes that row_cov
        and col_cov follow the API defined in NoiseCovBase. 
        """

        rowsize = tf.cast(tf.shape(x)[0], 'float64')
        colsize = tf.cast(tf.shape(x)[1], 'float64')

        # precompute sigma_col^{-1} * x'
        solve_col = col_cov.Sigma_inv_x(tf.transpose(x))
        logdet_col = col_cov.logdet

        # precompute sigma_row^{-1} * x
        solve_row = row_cov.Sigma_inv_x(x)
        logdet_row = row_cov.logdet

        return self._mnorm_logp_internal(colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col)

    def matnorm_logp_marginal_row(self, x, row_cov, col_cov, marg, marg_cov):
        """
        Log likelihood for centered matrix-variate normal density. Assumes that row_cov, 
        col_cov, and marg_cov follow the API defined in NoiseCovBase.

        When you marginalize in mnorm, you end up with a covariance S + APA', where
        P is the covariance of A in the relevant dimension. 

        This method exploits the matrix inversion and determinant lemmas to construct 
        S + APA' given the covariance API in in NoiseCovBase. 
        """
        rowsize = tf.cast(tf.shape(x)[0], 'float64')
        colsize = tf.cast(tf.shape(x)[1], 'float64')


        solve_col = col_cov.Sigma_inv_x(tf.transpose(x))
        logdet_col = col_cov.logdet

        solve_row, logdet_row = self.solve_det_marginal(x, row_cov, marg, marg_cov)
        
        return self._mnorm_logp_internal(colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col)

    def matnorm_logp_marginal_col(self, x, row_cov, col_cov, marg, marg_cov):
        """
        Log likelihood for centered matrix-variate normal density. Assumes that row_cov, 
        col_cov, and marg_cov follow the API defined in NoiseCovBase.

        When you marginalize in mnorm, you end up with a covariance S + APA', where
        P is the covariance of A in the relevant dimension. 

        This method exploits the matrix inversion and determinant lemmas to construct 
        S + APA' given the covariance API in in NoiseCovBase. 
        """
        rowsize = tf.cast(tf.shape(x)[0], 'float64')
        colsize = tf.cast(tf.shape(x)[1], 'float64')

        solve_row = row_cov.Sigma_inv_x(x)
        logdet_row = row_cov.logdet

        solve_col, logdet_col = self.solve_det_marginal(tf.transpose(x), col_cov, marg, marg_cov)

        return self._mnorm_logp_internal(colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col)

    def matnorm_logp_conditional_row(self, x, row_cov, col_cov, cond, cond_cov):
        """
        Log likelihood for centered matrix-variate normal density. Assumes that row_cov, 
        col_cov, and cond_cov follow the API defined in NoiseCovBase.

        When you go from joint to conditional in mnorm, you end up with a covariance S - APA',
        where P is the covariance of A in the relevant dimension. 

        This method exploits the matrix inversion and determinant lemmas to construct 
        S - APA' given the covariance API in in NoiseCovBase. 
        """

        rowsize = tf.cast(tf.shape(x)[0], 'float64')
        colsize = tf.cast(tf.shape(x)[1], 'float64')

        solve_col = col_cov.Sigma_inv_x(tf.transpose(x))
        logdet_col = col_cov.logdet

        solve_row, logdet_row = self.solve_det_conditional(x, row_cov, cond, cond_cov)
        
        return self._mnorm_logp_internal(colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col)

    def matnorm_logp_conditional_col(self, x, row_cov, col_cov, cond, cond_cov):
        """
        Log likelihood for centered matrix-variate normal density. Assumes that row_cov, 
        col_cov, and cond_cov follow the API defined in NoiseCovBase.

        When you go from joint to conditional in mnorm, you end up with a covariance S - APA',
        where P is the covariance of A in the relevant dimension. 

        This method exploits the matrix inversion and determinant lemmas to construct 
        S - APA' given the covariance API in in NoiseCovBase. 
        """
        rowsize = tf.cast(tf.shape(x)[0], 'float64')
        colsize = tf.cast(tf.shape(x)[1], 'float64')

        solve_row = row_cov.Sigma_inv_x(x)
        logdet_row = row_cov.logdet

        solve_col, logdet_col = self.solve_det_conditional(tf.transpose(x), col_cov, cond, cond_cov)

        return self._mnorm_logp_internal(colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col)