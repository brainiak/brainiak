import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal
import tensorflow as tf

from brainiak.matnormal.utils import rmn
from brainiak.matnormal.matnormal_likelihoods import matnorm_logp
from brainiak.matnormal.covs import CovIdentity, CovUnconstrainedCholesky

# X is m x n, so A sould be m x p

m = 5
n = 4
p = 3

rtol = 1e-7


def test_against_scipy_mvn_row(seeded_rng):

    rowcov = CovUnconstrainedCholesky(size=m)
    colcov = CovIdentity(size=n)
    X = rmn(np.eye(m), np.eye(n))
    X_tf = tf.constant(X, "float64")

    rowcov_np = rowcov._cov

    scipy_answer = np.sum(multivariate_normal.logpdf(
        X.T, np.zeros([m]), rowcov_np))
    tf_answer = matnorm_logp(X_tf, rowcov, colcov)
    assert_allclose(scipy_answer, tf_answer, rtol=rtol)


def test_against_scipy_mvn_col(seeded_rng):

    rowcov = CovIdentity(size=m)
    colcov = CovUnconstrainedCholesky(size=n)
    X = rmn(np.eye(m), np.eye(n))
    X_tf = tf.constant(X, "float64")

    colcov_np = colcov._cov

    scipy_answer = np.sum(multivariate_normal.logpdf(
        X, np.zeros([n]), colcov_np))
    tf_answer = matnorm_logp(X_tf, rowcov, colcov)
    assert_allclose(scipy_answer, tf_answer, rtol=rtol)
