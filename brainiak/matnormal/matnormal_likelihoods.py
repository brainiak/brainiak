import tensorflow as tf
from tensorflow import linalg as tlinalg
from .utils import scaled_I
import logging

logger = logging.getLogger(__name__)


def _condition(X):
    """
    Condition number (https://en.wikipedia.org/wiki/Condition_number)
    used for diagnostics.

    NOTE: this formulation is only defined for symmetric positive definite
    matrices (which covariances should be, and what we're using this for)

    Parameters
    ----------
    X: tf.Tensor
        Symmetric tensor to compute condition number of

    """
    s = tf.linalg.svd(X, compute_uv=False)
    return tf.reduce_max(input_tensor=s) / tf.reduce_min(input_tensor=s)


def solve_det_marginal(x, sigma, A, Q):
    """
    Use matrix inversion lemma for the solve:

    .. math::
        (\\Sigma + AQA^T)^{-1} X =\\
        (\\Sigma^{-1} - \\Sigma^{-1} A (Q^{-1} +
        A^T \\Sigma^{-1} A)^{-1} A^T \\Sigma^{-1}) X

    Use matrix determinant lemma for determinant:

    .. math::
        \\log|(\\Sigma + AQA^T)| = \\log|Q^{-1} + A^T \\Sigma^{-1} A|
        + \\log|Q| + \\log|\\Sigma|

    Parameters
    ----------
    x: tf.Tensor
        Tensor to multiply the solve by
    sigma: brainiak.matnormal.CovBase
        Covariance object implementing solve and logdet
    A: tf.Tensor
        Factor multiplying the variable we marginalized out
    Q: brainiak.matnormal.CovBase
        Covariance object of marginalized variable,
        implementing solve and logdet
    """

    # For diagnostics, we want to check condition numbers
    # of things we invert. This includes Q and Sigma, as well
    # as the "lemma factor" for lack of a better definition
    logging.log(logging.DEBUG, "Printing diagnostics for solve_det_marginal")
    lemma_cond = _condition(
        Q._prec + tf.matmul(A, sigma.solve(A), transpose_a=True))
    logging.log(
        logging.DEBUG,
        f"lemma_factor condition={lemma_cond}",
    )
    logging.log(logging.DEBUG, f"Q condition={_condition(Q._cov)}")
    logging.log(logging.DEBUG, f"sigma condition={_condition(sigma._cov)}")
    logging.log(
        logging.DEBUG,
        f"sigma max={tf.reduce_max(input_tensor=A)}," +
        f"sigma min={tf.reduce_min(input_tensor=A)}",
    )

    # cholesky of (Qinv + A^T Sigma^{-1} A), which looks sort of like
    # a schur complement but isn't, so we call it the "lemma factor"
    # since we use it in woodbury and matrix determinant lemmas
    lemma_factor = tlinalg.cholesky(
        Q._prec + tf.matmul(A, sigma.solve(A), transpose_a=True)
    )

    logdet = (
        Q.logdet
        + sigma.logdet
        + 2 *
        tf.reduce_sum(input_tensor=tf.math.log(
            tlinalg.diag_part(lemma_factor)))
    )

    logging.log(logging.DEBUG, f"Log-determinant of Q={Q.logdet}")
    logging.log(logging.DEBUG, f"sigma logdet={sigma.logdet}")
    lemma_logdet = 2 * \
        tf.reduce_sum(input_tensor=tf.math.log(
            tlinalg.diag_part(lemma_factor)))
    logging.log(
        logging.DEBUG,
        f"lemma factor logdet={lemma_logdet}",
    )

    # A^T Sigma^{-1}
    Atrp_Sinv = tf.matmul(A, sigma._prec, transpose_a=True)
    # (Qinv + A^T Sigma^{-1} A)^{-1} A^T Sigma^{-1}
    prod_term = tlinalg.cholesky_solve(lemma_factor, Atrp_Sinv)

    solve = tf.matmul(
        sigma.solve(scaled_I(1.0, sigma.size) - tf.matmul(A, prod_term)), x
    )

    return solve, logdet


def solve_det_conditional(x, sigma, A, Q):
    """
    Use matrix inversion lemma for the solve:

    .. math::
        (\\Sigma - AQ^{-1}A^T)^{-1} X =\\
        (\\Sigma^{-1} + \\Sigma^{-1} A (Q -
        A^T \\Sigma^{-1} A)^{-1} A^T \\Sigma^{-1}) X

    Use matrix determinant lemma for determinant:

    .. math::
        \\log|(\\Sigma - AQ^{-1}A^T)| =
        \\log|Q - A^T \\Sigma^{-1} A| - \\log|Q| + \\log|\\Sigma|

    Parameters
    ----------
    x: tf.Tensor
        Tensor to multiply the solve by
    sigma: brainiak.matnormal.CovBase
        Covariance object implementing solve and logdet
    A: tf.Tensor
        Factor multiplying the variable we conditioned on
    Q: brainiak.matnormal.CovBase
        Covariance object of conditioning variable,
        implementing solve and logdet

    """

    # (Q - A^T Sigma^{-1} A)
    lemma_factor = tlinalg.cholesky(
        Q._cov - tf.matmul(A, sigma.solve(A), transpose_a=True)
    )

    logdet = (
        -Q.logdet
        + sigma.logdet
        + 2 *
        tf.reduce_sum(input_tensor=tf.math.log(
            tlinalg.diag_part(lemma_factor)))
    )

    # A^T Sigma^{-1}
    Atrp_Sinv = tf.matmul(A, sigma._prec, transpose_a=True)
    # (Q - A^T Sigma^{-1} A)^{-1} A^T Sigma^{-1}
    prod_term = tlinalg.cholesky_solve(lemma_factor, Atrp_Sinv)

    solve = tf.matmul(
        sigma.solve(scaled_I(1.0, sigma.size) + tf.matmul(A, prod_term)), x
    )

    return solve, logdet


def _mnorm_logp_internal(
    colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col
):
    """Construct logp from the solves and determinants.

    Parameters
    ----------------
    colsize: int
        Column dimnesion of observation tensor
    rowsize: int
        Row dimension of observation tensor
    logdet_row: tf.Tensor (scalar)
        log-determinant of row covariance
    logdet_col: tf.Tensor (scalar)
        log-determinant of column covariance
    solve_row: tf.Tensor
        Inverse row covariance multiplying the observation tensor
    solve_col
        Inverse column covariance multiplying the transpose of
        the observation tensor
    """
    log2pi = 1.8378770664093453

    logging.log(logging.DEBUG,
                f"column precision trace ={tlinalg.trace(solve_col)}")
    logging.log(logging.DEBUG,
                f"row precision trace ={tlinalg.trace(solve_row)}")
    logging.log(logging.DEBUG, f"row cov logdet ={logdet_row}")
    logging.log(logging.DEBUG, f"col cov logdet ={logdet_col}")

    denominator = (
        -rowsize * colsize * log2pi - colsize * logdet_row -
        rowsize * logdet_col
    )
    numerator = -tlinalg.trace(tf.matmul(solve_col, solve_row))
    return 0.5 * (numerator + denominator)


def matnorm_logp(x, row_cov, col_cov):
    """Log likelihood for centered matrix-variate normal density.
    Assumes that row_cov and col_cov follow the API defined in CovBase.

    Parameters
    ----------------
    x: tf.Tensor
        Observation tensor
    row_cov: CovBase
        Row covariance implementing the CovBase API
    col_cov: CovBase
        Column Covariance implementing the CovBase API

    """

    rowsize = tf.cast(tf.shape(input=x)[0], "float64")
    colsize = tf.cast(tf.shape(input=x)[1], "float64")

    # precompute sigma_col^{-1} * x'
    solve_col = col_cov.solve(tf.transpose(a=x))
    logdet_col = col_cov.logdet

    # precompute sigma_row^{-1} * x
    solve_row = row_cov.solve(x)
    logdet_row = row_cov.logdet

    return _mnorm_logp_internal(
        colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col
    )


def matnorm_logp_marginal_row(x, row_cov, col_cov, marg, marg_cov):
    """
    Log likelihood for marginal centered matrix-variate normal density.

    .. math::
        X &\\sim \\mathcal{MN}(0, Q, C)\\

        Y \\mid \\X &\\sim \\mathcal{MN}(AX, R, C),\\

        Y &\\sim \\mathcal{MN}(0, R + AQA^T, C)

    This function efficiently computes the marginals by unpacking some
    info in the covariance classes and then dispatching to
    `solve_det_marginal`.

    Parameters
    ---------------
    x: tf.Tensor
        Observation tensor
    row_cov: CovBase
        Row covariance implementing the CovBase API (:math:`R` above).
    col_cov: CovBase
        Column Covariance implementing the CovBase API (:math:`C` above).
    marg: tf.Tensor
        Marginal factor (:math:`A` above).
    marg_cov: CovBase
        Prior covariance implementing the CovBase API (:math:`Q` above).
    """
    rowsize = tf.cast(tf.shape(input=x)[0], "float64")
    colsize = tf.cast(tf.shape(input=x)[1], "float64")

    solve_col = col_cov.solve(tf.transpose(a=x))
    logdet_col = col_cov.logdet

    solve_row, logdet_row = solve_det_marginal(x, row_cov, marg, marg_cov)

    return _mnorm_logp_internal(
        colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col
    )


def matnorm_logp_marginal_col(x, row_cov, col_cov, marg, marg_cov):
    """
    Log likelihood for centered marginal matrix-variate normal density.

    .. math::
        X &\\sim \\mathcal{MN}(0, R, Q)\\

        Y \\mid \\X &\\sim \\mathcal{MN}(XA, R, C),\\

        Y &\\sim \\mathcal{MN}(0, R, C + A^TQA)

    This function efficiently computes the marginals by unpacking some
    info in the covariance classes and then dispatching to
    `solve_det_marginal`.

    Parameters
    ---------------
    x: tf.Tensor
        Observation tensor
    row_cov: CovBase
        Row covariance implementing the CovBase API (:math:`R` above).
    col_cov: CovBase
        Column Covariance implementing the CovBase API (:math:`C` above).
    marg: tf.Tensor
        Marginal factor (:math:`A` above).
    marg_cov: CovBase
        Prior covariance implementing the CovBase API (:math:`Q` above).

    """
    rowsize = tf.cast(tf.shape(input=x)[0], "float64")
    colsize = tf.cast(tf.shape(input=x)[1], "float64")

    solve_row = row_cov.solve(x)
    logdet_row = row_cov.logdet

    solve_col, logdet_col = solve_det_marginal(
        tf.transpose(a=x), col_cov, tf.transpose(a=marg), marg_cov
    )

    return _mnorm_logp_internal(
        colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col
    )


def matnorm_logp_conditional_row(x, row_cov, col_cov, cond, cond_cov):
    """
    Log likelihood for centered conditional matrix-variate normal density.

    Consider the following partitioned matrix-normal density:

    .. math::
        \\begin{bmatrix}
        \\operatorname{vec}\\left[\\mathbf{X}_{i j}\\right] \\\\
        \\operatorname{vec}\\left[\\mathbf{Y}_{i k}\\right]\\end{bmatrix}
        \\sim \\mathcal{N}\\left(0,\\begin{bmatrix} \\Sigma_{j} \\otimes
        \\Sigma_{i} & \\Sigma_{j k} \\otimes \\Sigma_{i}\\\\
        \\Sigma_{k j} \\otimes \\Sigma_{i} & \\Sigma_{k} \\otimes \\Sigma_{i}
        \\end{bmatrix}\\right)

    Then we can write the conditional:

    .. math::
        \\mathbf{X}^T j i \\mid \\mathbf{Y}_{k i}^T
        \\sim \\mathcal{M}\\
        \\mathcal{N}\\left(0, \\Sigma_{j}-\\Sigma_{j k} \\Sigma_{k}^{-1}
        \\Sigma_{k j},\\
        \\Sigma_{i}\\right)

    This function efficiently computes the conditionals by unpacking some
    info in the covariance classes and then dispatching to
    `solve_det_conditional`.

    Parameters
    ---------------
    x: tf.Tensor
        Observation tensor
    row_cov: CovBase
        Row covariance (:math:`\\Sigma_{i}` in the notation above).
    col_cov: CovBase
        Column covariance (:math:`\\Sigma_{j}` in the notation above).
    cond: tf.Tensor
        Off-diagonal block of the partitioned covariance (:math:`\\Sigma_{jk}`
        in the notation above).
    cond_cov: CovBase
        Covariance of conditioning variable (:math:`\\Sigma_{k}` in the
        notation above).

    """
    rowsize = tf.cast(tf.shape(input=x)[0], "float64")
    colsize = tf.cast(tf.shape(input=x)[1], "float64")

    solve_col = col_cov.solve(tf.transpose(a=x))
    logdet_col = col_cov.logdet

    solve_row, logdet_row = solve_det_conditional(x, row_cov, cond, cond_cov)

    return _mnorm_logp_internal(
        colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col
    )


def matnorm_logp_conditional_col(x, row_cov, col_cov, cond, cond_cov):
    """
    Log likelihood for centered conditional matrix-variate normal density.

    Consider the following partitioned matrix-normal density:

    .. math::
        \\begin{bmatrix}
        \\operatorname{vec}\\left[\\mathbf{X}_{i j}\\right] \\\\
        \\operatorname{vec}\\left[\\mathbf{Y}_{i k}\\right]\\end{bmatrix}
        \\sim \\mathcal{N}\\left(0,\\begin{bmatrix} \\Sigma_{j} \\otimes
        \\Sigma_{i} & \\Sigma_{j k} \\otimes \\Sigma_{i}\\\\
        \\Sigma_{k j} \\otimes \\Sigma_{i} & \\Sigma_{k} \\otimes \\Sigma_{i}
        \\end{bmatrix}\\right)

    Then we can write the conditional:

    .. math::
        \\mathbf{X}_{i j} \\mid \\mathbf{Y}_{i k} \\sim \\mathcal{M}\\
        \\mathcal{N}\\left(0, \\Sigma_{i}, \\Sigma_{j}-\\Sigma_{j k}\\
        \\Sigma_{k}^{-1} \\Sigma_{k j}\\right)

    This function efficiently computes the conditionals by unpacking some
    info in the covariance classes and then dispatching to
    `solve_det_conditional`.

    Parameters
    ---------------
    x: tf.Tensor
        Observation tensor
    row_cov: CovBase
        Row covariance (:math:`\\Sigma_{i}` in the notation above).
    col_cov: CovBase
        Column covariance (:math:`\\Sigma_{j}` in the notation above).
    cond: tf.Tensor
        Off-diagonal block of the partitioned covariance (:math:`\\Sigma_{jk}`
        in the notation above).
    cond_cov: CovBase
        Covariance of conditioning variable (:math:`\\Sigma_{k}` in the
        notation above).

    """
    rowsize = tf.cast(tf.shape(input=x)[0], "float64")
    colsize = tf.cast(tf.shape(input=x)[1], "float64")

    solve_row = row_cov.solve(x)
    logdet_row = row_cov.logdet

    solve_col, logdet_col = solve_det_conditional(
        tf.transpose(a=x), col_cov, tf.transpose(a=cond), cond_cov
    )

    return _mnorm_logp_internal(
        colsize, rowsize, logdet_row, logdet_col, solve_row, solve_col
    )
