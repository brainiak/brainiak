import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal
import tensorflow as tf

from brainiak.matnormal.utils import rmn
from brainiak.matnormal.matnormal_likelihoods import (
    matnorm_logp_marginal_col,
    matnorm_logp_marginal_row,
)

from brainiak.matnormal.covs import CovIdentity, CovUnconstrainedCholesky

# X is m x n, so A sould be m x p

m = 5
n = 4
p = 3

rtol = 1e-7


def test_against_scipy_mvn_row_marginal(seeded_rng):

    rowcov = CovUnconstrainedCholesky(size=m)
    colcov = CovIdentity(size=n)
    Q = CovUnconstrainedCholesky(size=p)

    X = rmn(np.eye(m), np.eye(n))
    A = rmn(np.eye(m), np.eye(p))

    A_tf = tf.constant(A, "float64")
    X_tf = tf.constant(X, "float64")

    Q_np = Q._cov

    rowcov_np = rowcov._cov + A.dot(Q_np).dot(A.T)

    scipy_answer = np.sum(multivariate_normal.logpdf(
        X.T, np.zeros([m]), rowcov_np))

    tf_answer = matnorm_logp_marginal_row(X_tf, rowcov, colcov, A_tf, Q)
    assert_allclose(scipy_answer, tf_answer, rtol=rtol)


def test_against_scipy_mvn_col_marginal(seeded_rng):

    rowcov = CovIdentity(size=m)
    colcov = CovUnconstrainedCholesky(size=n)
    Q = CovUnconstrainedCholesky(size=p)

    X = rmn(np.eye(m), np.eye(n))
    A = rmn(np.eye(p), np.eye(n))

    A_tf = tf.constant(A, "float64")
    X_tf = tf.constant(X, "float64")

    Q_np = Q._cov

    colcov_np = colcov._cov + A.T.dot(Q_np).dot(A)

    scipy_answer = np.sum(multivariate_normal.logpdf(
        X, np.zeros([n]), colcov_np))

    tf_answer = matnorm_logp_marginal_col(X_tf, rowcov, colcov, A_tf, Q)
    assert_allclose(scipy_answer, tf_answer, rtol=rtol)
