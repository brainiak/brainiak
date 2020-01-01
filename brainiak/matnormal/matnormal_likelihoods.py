import tensorflow as tf
from .utils import scaled_I
import logging

logger = logging.getLogger(__name__)


def _condition(X):
    """
    Condition number, used for diagnostics
    """
    s = tf.svd(X, compute_uv=False)
    return tf.reduce_max(s)/tf.reduce_min(s)


def solve_det_marginal(x, sigma, A, Q):
    """
    Use matrix inversion lemma for the solve:
    .. math::
    (\Sigma + AQA')^{-1} X =\\
    \Sigma^{-1} - \Sigma^{-1} A (Q^{-1} +
    A' \Sigma^{-1} A)^{-1} A' \Sigma^{-1}

    Use matrix determinant lemma for determinant:
    .. math::
    \log|(\Sigma + AQA')| = \log|Q^{-1} + A' \Sigma^{-1} A|
    + \log|Q| + \log|\Sigma|
    """

    # For diagnostics, we want to check condition numbers 
    # of things we invert. This includes Q and Sigma, as well
    # as the "lemma factor" for lack of a better definition
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.log("Printing diagnostics for solve_det_marginal")
        A = tf.Print(A, [_condition(Q._prec + tf.matmul(A,
                     sigma.solve(A), transpose_a=True))],
                     'lemma_factor condition')
        A = tf.Print(A, [_condition(Q._cov)], 'Q condition')
        A = tf.Print(A, [_condition(sigma._cov)], 'sigma condition')
        A = tf.Print(A, [tf.reduce_max(A), tf.reduce_min(A)], 'A minmax')

    # cholesky of (Qinv + A' Sigma^{-1} A), which looks sort of like
    # a schur complement by isn't, so we call it the "lemma factor"
    # since we use it in woodbury and matrix determinant lemmas
    lemma_factor = tf.cholesky(Q._prec + tf.matmul(A,
                                sigma.solve(A), transpose_a=True))

    logdet = Q.logdet + sigma.logdet +\
        2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(lemma_factor)))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logdet = tf.Print(logdet, [Q.logdet], 'Q logdet')
        logdet = tf.Print(logdet, [sigma.logdet], 'sigma logdet')
        logdet = tf.Print(logdet, [2 * tf.reduce_sum(tf.log(
                          tf.matrix_diag_part(lemma_factor)))],
                          'iqf logdet')

    # A' Sigma^{-1}
    Atrp_Sinv = tf.matmul(A, sigma._prec, transpose_a=True)
    # (Qinv + A' Sigma^{-1} A)^{-1} A' Sigma^{-1}
    prod_term = tf.cholesky_solve(lemma_factor, Atrp_Sinv)

    solve = tf.matmul(sigma.solve(scaled_I(1.0, sigma.size) -
                      tf.matmul(A, prod_term)), x)

    return solve, logdet


def solve_det_conditional(x, sigma, A, Q):
    """
    Use matrix inversion lemma for the solve:
    .. math::
    (\Sigma - AQ^{-1}A')^{-1} X =\\
    \Sigma^{-1} + \Sigma^{-1} A (Q -
    A' \Sigma^{-1} A)^{-1} A' \Sigma^{-1} X

    Use matrix determinant lemma for determinant:
    .. math::
    \log|(\Sigma - AQ^{-1}A')| =
    \log|Q - A' \Sigma^{-1} A| - \log|Q| + \log|\Sigma|
    """

    # (Q - A' Sigma^{-1} A)
    lemma_factor = tf.cholesky(Q._cov - tf.matmul(A,
                                sigma.solve(A), transpose_a=True))

    logdet = -Q.logdet + sigma.logdet +\
        2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(lemma_factor)))

    # A' Sigma^{-1}
    Atrp_Sinv = tf.matmul(A, sigma._prec, transpose_a=True)
    # (Q - A' Sigma^{-1} A)^{-1} A' Sigma^{-1}
    prod_term = tf.cholesky_solve(lemma_factor, Atrp_Sinv)

    solve = tf.matmul(sigma.solve(scaled_I(1.0, sigma.size) +
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
    solve_col = col_cov.solve(tf.transpose(x))
    logdet_col = col_cov.logdet

    # precompute sigma_row^{-1} * x
    solve_row = row_cov.solve(x)
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

    solve_col = col_cov.solve(tf.transpose(x))
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

    solve_row = row_cov.solve(x)
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

    solve_col = col_cov.solve(tf.transpose(x))
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

    solve_row = row_cov.solve(x)
    logdet_row = row_cov.logdet

    solve_col, logdet_col = solve_det_conditional(tf.transpose(x),
                                                  col_cov,
                                                  tf.transpose(cond),
                                                  cond_cov)

    return _mnorm_logp_internal(colsize, rowsize, logdet_row,
                                logdet_col, solve_row, solve_col)
