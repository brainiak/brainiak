import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import wishart, multivariate_normal
import tensorflow as tf

from brainiak.matnormal.utils import rmn
from brainiak.matnormal.matnormal_likelihoods import (
    matnorm_logp_conditional_col,
    matnorm_logp_conditional_row,
)
from brainiak.matnormal.covs import CovIdentity, CovUnconstrainedCholesky

# X is m x n, so A sould be m x p

m = 5
n = 4
p = 3

rtol = 1e-7


def test_against_scipy_mvn_row_conditional(seeded_rng):

    # have to be careful for constructing everything as a submatrix of a big
    # PSD matrix, else no guarantee that anything's invertible.
    cov_np = wishart.rvs(df=m + p + 2, scale=np.eye(m + p))

    # rowcov = CovConstant(cov_np[0:m, 0:m])
    rowcov = CovUnconstrainedCholesky(Sigma=cov_np[0:m, 0:m])
    A = cov_np[0:m, m:]

    colcov = CovIdentity(size=n)

    Q = CovUnconstrainedCholesky(Sigma=cov_np[m:, m:])

    X = rmn(np.eye(m), np.eye(n))

    A_tf = tf.constant(A, "float64")
    X_tf = tf.constant(X, "float64")

    Q_np = Q._cov

    rowcov_np = rowcov._cov - A.dot(np.linalg.inv(Q_np)).dot((A.T))

    scipy_answer = np.sum(multivariate_normal.logpdf(
        X.T, np.zeros([m]), rowcov_np))

    tf_answer = matnorm_logp_conditional_row(X_tf, rowcov, colcov, A_tf, Q)
    assert_allclose(scipy_answer, tf_answer, rtol=rtol)


def test_against_scipy_mvn_col_conditional(seeded_rng):

    # have to be careful for constructing everything as a submatrix of a big
    # PSD matrix, else no guarantee that anything's invertible.
    cov_np = wishart.rvs(df=m + p + 2, scale=np.eye(m + p))

    rowcov = CovIdentity(size=m)
    colcov = CovUnconstrainedCholesky(Sigma=cov_np[0:n, 0:n])
    A = cov_np[n:, 0:n]

    Q = CovUnconstrainedCholesky(Sigma=cov_np[n:, n:])

    X = rmn(np.eye(m), np.eye(n))

    A_tf = tf.constant(A, "float64")
    X_tf = tf.constant(X, "float64")

    Q_np = Q._cov

    colcov_np = colcov._cov - A.T.dot(np.linalg.inv(Q_np)).dot((A))

    scipy_answer = np.sum(multivariate_normal.logpdf(
        X, np.zeros([n]), colcov_np))

    tf_answer = matnorm_logp_conditional_col(X_tf, rowcov, colcov, A_tf, Q)

    assert_allclose(scipy_answer, tf_answer, rtol=rtol)
