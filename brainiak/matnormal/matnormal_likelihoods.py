import tensorflow as tf
from .utils import scaled_I
import logging

logger = logging.getLogger(__name__)


def _condition(X):
    s = tf.svd(X, compute_uv=False)
    return tf.reduce_max(s)/tf.reduce_min(s)


def solve_det_marginal(x, sigma, A, Q):
    """
    Use matrix inversion lemma for the solve:
    .. math::
        (\Sigma + AQA')^{-1} X =\\
         \Sigma^{-1} - \Sigma^{-1} A (Q^{-1} + A' \Sigma^{-1} A)^{-1} A' \Sigma^{-1}

    Use matrix determinant lemma for determinant:
    ..math::
        \log|(\Sigma + AQA')| = \log|Q^{-1} + A' \Sigma^{-1} A| + \log|Q| + \log|\Sigma|
    """

    # we care about condition number of i_qf
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        A = tf.Print(A, [_condition(Q.Sigma_inv + tf.matmul(A,
                     sigma.Sigma_inv_x(A), transpose_a=True))], 'i_qf condition')
        # since the sigmas expose only inverse, we invert their
        # conditions to get what we want
        A = tf.Print(A, [1/_condition(Q.Sigma_inv)], 'Q condition')
        A = tf.Print(A, [1/_condition(sigma.Sigma_inv)], 'sigma condition')
        A = tf.Print(A, [tf.reduce_max(A), tf.reduce_min(A)], 'A minmax')

    # cholesky of (Qinv + A' Sigma^{-1} A)
    i_qf_cholesky = tf.cholesky(Q.Sigma_inv + tf.matmul(A,
                                sigma.Sigma_inv_x(A), transpose_a=True))

    logdet = Q.logdet + sigma.logdet +\
        2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(i_qf_cholesky)))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logdet = tf.Print(logdet, [Q.logdet], 'Q logdet')
        logdet = tf.Print(logdet, [sigma.logdet], 'sigma logdet')
        logdet = tf.Print(logdet, [2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(i_qf_cholesky)))],
                          'iqf logdet')

    # A' Sigma^{-1}
    Atrp_Sinv = tf.matmul(A, sigma.Sigma_inv, transpose_a=True)
    # (Qinv + A' Sigma^{-1} A)^{-1} A' Sigma^{-1}
    prod_term = tf.cholesky_solve(i_qf_cholesky, Atrp_Sinv)

    solve = tf.matmul(sigma.Sigma_inv_x(scaled_I(1.0, sigma.size) -
                      tf.matmul(A, prod_term)), x)

    return solve, logdet


def solve_det_conditional(x, sigma, A, Q):
    """
    Use matrix inversion lemma for the solve:
    .. math::
        (\Sigma - AQ^{-1}A')^{-1} X =\\
         \Sigma^{-1} + \Sigma^{-1} A (Q - A' \Sigma^{-1} A)^{-1} A' \Sigma^{-1} X 

    Use matrix determinant lemma for determinant:
    ..math::
        \log|(\Sigma - AQ^{-1}A')| = \log|Q - A' \Sigma^{-1} A| - \log|Q| + \log|\Sigma|
    """

    # (Q - A' Sigma^{-1} A)
    i_qf_cholesky = tf.cholesky(Q.Sigma - tf.matmul(A,
                                sigma.Sigma_inv_x(A), transpose_a=True))

    logdet = -Q.logdet + sigma.logdet +\
        2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(i_qf_cholesky)))

    # A' Sigma^{-1}
    Atrp_Sinv = tf.matmul(A, sigma.Sigma_inv, transpose_a=True)
    # (Q - A' Sigma^{-1} A)^{-1} A' Sigma^{-1}
    prod_term = tf.cholesky_solve(i_qf_cholesky, Atrp_Sinv)

    solve = tf.matmul(sigma.Sigma_inv_x(scaled_I(1.0, sigma.size) +
                      tf.matmul(A, prod_term)), x)

    return solve, logdet


def _mnorm_logp_internal(colsize, rowsize, logdet_row, logdet_col,
                         solve_row, solve_col):
    """Construct logp from the solves and determinants.
    """
    log2pi = 1.8378770664093453

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        solve_row = tf.Print(solve_row, [tf.trace(solve_col)], 'coltrace')
        solve_row = tf.Print(solve_row, [tf.trace(solve_row)], 'rowtrace')
        solve_row = tf.Print(solve_row, [logdet_row], 'logdet_row')
        solve_row = tf.Print(solve_row, [logdet_col], 'logdet_col')

    denominator = - rowsize * colsize * log2pi -\
        colsize * logdet_row - rowsize * logdet_col
    numerator = - tf.trace(tf.matmul(solve_col, solve_row))
    return 0.5 * (numerator + denominator)


def matnorm_logp(x, row_cov, col_cov):
    """Log likelihood for centered matrix-variate normal density.
    Assumes that row_cov and col_cov follow the API defined in CovBase.
    """

    rowsize = tf.cast(tf.shape(x)[0], 'float64')
    colsize = tf.cast(tf.shape(x)[1], 'float64')

    # precompute sigma_col^{-1} * x'
    solve_col = col_cov.Sigma_inv_x(tf.transpose(x))
    logdet_col = col_cov.logdet

    # precompute sigma_row^{-1} * x
    solve_row = row_cov.Sigma_inv_x(x)
    logdet_row = row_cov.logdet

    return _mnorm_logp_internal(colsize, rowsize, logdet_row,
                                logdet_col, solve_row, solve_col)


def matnorm_logp_marginal_row(x, row_cov, col_cov, marg, marg_cov):
    """
    Log likelihood for centered matrix-variate normal density.
    Assumes that row_cov, col_cov, and marg_cov follow the API defined
    in CovBase.

    When you marginalize in mnorm, you end up with a covariance S + APA',
    where P is the covariance of A in the relevant dimension.

    This method exploits the matrix inversion and determinant lemmas to
    construct S + APA' given the covariance API in in CovBase.
    """
    rowsize = tf.cast(tf.shape(x)[0], 'float64')
    colsize = tf.cast(tf.shape(x)[1], 'float64')

    solve_col = col_cov.Sigma_inv_x(tf.transpose(x))
    logdet_col = col_cov.logdet

    solve_row, logdet_row = solve_det_marginal(x, row_cov, marg,
                                               marg_cov)

    return _mnorm_logp_internal(colsize, rowsize, logdet_row,
                                logdet_col, solve_row, solve_col)


def matnorm_logp_marginal_col(x, row_cov, col_cov, marg, marg_cov):
    """
    Log likelihood for centered matrix-variate normal density. Assumes that
    row_cov, col_cov, and marg_cov follow the API defined in CovBase.

    When you marginalize in mnorm, you end up with a covariance S + APA',
    where P is the covariance of A in the relevant dimension.

    This method exploits the matrix inversion and determinant lemmas to
    construct S + APA' given the covariance API in in CovBase.
    """
    rowsize = tf.cast(tf.shape(x)[0], 'float64')
    colsize = tf.cast(tf.shape(x)[1], 'float64')

    solve_row = row_cov.Sigma_inv_x(x)
    logdet_row = row_cov.logdet

    solve_col, logdet_col = solve_det_marginal(tf.transpose(x),
                                               col_cov,
                                               tf.transpose(marg),
                                               marg_cov)

    return _mnorm_logp_internal(colsize, rowsize, logdet_row,
                                logdet_col, solve_row, solve_col)


def matnorm_logp_conditional_row(x, row_cov, col_cov, cond, cond_cov):
    """
    Log likelihood for centered matrix-variate normal density. Assumes that
    row_cov, col_cov, and cond_cov follow the API defined in CovBase.

    When you go from joint to conditional in mnorm, you end up with a
    covariance S - APA', where P is the covariance of A in the relevant
    dimension.

    This method exploits the matrix inversion and determinant lemmas to
    construct S - APA' given the covariance API in in CovBase.
    """

    rowsize = tf.cast(tf.shape(x)[0], 'float64')
    colsize = tf.cast(tf.shape(x)[1], 'float64')

    solve_col = col_cov.Sigma_inv_x(tf.transpose(x))
    logdet_col = col_cov.logdet

    solve_row, logdet_row = solve_det_conditional(x, row_cov, cond,
                                                  cond_cov)

    return _mnorm_logp_internal(colsize, rowsize, logdet_row,
                                logdet_col, solve_row, solve_col)


def matnorm_logp_conditional_col(x, row_cov, col_cov, cond, cond_cov):
    """
    Log likelihood for centered matrix-variate normal density. Assumes that
    row_cov, col_cov, and cond_cov follow the API defined in CovBase.

    When you go from joint to conditional in mnorm, you end up with a
    covariance S - APA', where P is the covariance of A in the relevant
    dimension.

    This method exploits the matrix inversion and determinant lemmas to
    construct S - APA' given the covariance API in in CovBase.
    """
    rowsize = tf.cast(tf.shape(x)[0], 'float64')
    colsize = tf.cast(tf.shape(x)[1], 'float64')

    solve_row = row_cov.Sigma_inv_x(x)
    logdet_row = row_cov.logdet

    solve_col, logdet_col = solve_det_conditional(tf.transpose(x),
                                                  col_cov,
                                                  tf.transpose(cond),
                                                  cond_cov)

    return _mnorm_logp_internal(colsize, rowsize, logdet_row,
                                logdet_col, solve_row, solve_col)
