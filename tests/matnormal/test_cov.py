import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm, wishart, invgamma
from brainiak.matnormal.covs import (CovIdentity,
                                     CovAR1,
                                     CovIsotropic,
                                     CovDiagonal,
                                     CovDiagonalGammaPrior,
                                     CovUnconstrainedCholesky,
                                     CovUnconstrainedCholeskyWishartReg,
                                     CovUnconstrainedInvCholesky,
                                     CovKroneckerFactored)
import tensorflow as tf
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)

# X is m x n, so A sould be m x p

m = 8
n = 4
p = 3

rtol = 1e-7
atol = 1e-7


def logdet_sinv_np(X, sigma):
    # logdet
    _, logdet_np = np.linalg.slogdet(sigma)
    # sigma-inv
    sinv_np = np.linalg.inv(sigma)
    # solve
    sinvx_np = np.linalg.solve(sigma, X)
    return logdet_np, sinv_np, sinvx_np


def logdet_sinv_np_mask(X, sigma, mask):
    mask_indices = np.nonzero(mask)[0]
    # logdet
    _, logdet_np = np.linalg.slogdet(sigma[np.ix_(mask_indices, mask_indices)])
    # sigma-inv
    sinv_np_ = np.linalg.inv(sigma[np.ix_(mask_indices, mask_indices)])
    # sigma-inverse *
    sinvx_np_ = sinv_np_.dot(X[mask_indices, :])

    sinv_np = np.zeros_like(sigma)
    sinv_np[np.ix_(mask_indices, mask_indices)] = sinv_np_
    sinvx_np = np.zeros_like(X)
    sinvx_np[mask_indices, :] = sinvx_np_

    return logdet_np, sinv_np, sinvx_np


X = norm.rvs(size=(m, n))
X_tf = tf.constant(X)
A = norm.rvs(size=(m, p))
A_tf = tf.constant(A)


def test_CovConstant():

    cov_np = wishart.rvs(df=m+2, scale=np.eye(m))
    cov = CovUnconstrainedCholesky(m, cov_np)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_CovIdentity():

    cov = CovIdentity(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = np.eye(m)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_CovIsotropic():

    cov = CovIsotropic(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.sigma.eval(session=sess) * np.eye(cov.size)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_CovDiagonal():

    cov = CovDiagonal(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = np.diag(1/cov.prec.eval(session=sess))
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_CovDiagonal_initialized():

    cov_np = np.diag(np.exp(np.random.normal(size=m)))
    cov = CovDiagonal(size=m, sigma=np.diag(cov_np))

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_CovDiagonalGammaPrior():

    cov_np = np.diag(np.exp(np.random.normal(size=m)))
    cov = CovDiagonalGammaPrior(size=m, sigma=np.diag(cov_np), alpha=1.5,
                                beta=1e-10)

    ig = invgamma(1.5, scale=1e-10)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        penalty_np = np.sum(ig.logpdf(1/np.diag(cov_np)))
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)
        assert_allclose(penalty_np, cov.logp.eval(session=sess), rtol=rtol)


def test_CovUnconstrainedCholesky():

    cov = CovUnconstrainedCholesky(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.Sigma.eval(session=sess)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_CovUnconstrainedCholeskyWishartReg():

    cov = CovUnconstrainedCholeskyWishartReg(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.Sigma.eval(session=sess)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)
        # now compute the regularizer
        reg = wishart.logpdf(cov_np, df=m+2, scale=1e10 * np.eye(m))
        assert_allclose(reg, cov.logp.eval(session=sess), rtol=rtol)


def test_CovUnconstrainedInvCholesky():

    cov = CovUnconstrainedInvCholesky(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.Sigma.eval(session=sess)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_Cov2FactorKron():
    assert(m % 2 == 0)
    dim1 = int(m/2)
    dim2 = 2

    with pytest.raises(TypeError) as excinfo:
        cov = CovKroneckerFactored(sizes=dim1)
    assert "sizes is not a list" in str(excinfo.value)

    cov = CovKroneckerFactored(sizes=[dim1, dim2])

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        L1 = (cov.L[0]).eval(session=sess)
        L2 = (cov.L[1]).eval(session=sess)
        cov_np = np.kron(np.dot(L1, L1.transpose()),
                         np.dot(L2, L2.transpose()))
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)

        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_Cov3FactorKron():
    assert(m % 4 == 0)
    dim1 = int(m/4)
    dim2 = 2
    dim3 = 2
    cov = CovKroneckerFactored(sizes=[dim1, dim2, dim3])

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        L1 = (cov.L[0]).eval(session=sess)
        L2 = (cov.L[1]).eval(session=sess)
        L3 = (cov.L[2]).eval(session=sess)
        cov_np = np.kron(np.kron(np.dot(L1, L1.transpose()),
                         np.dot(L2, L2.transpose())),
                         np.dot(L3, L3.transpose()))
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)

        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_Cov3FactorMaskedKron():
    assert(m % 4 == 0)
    dim1 = int(m/4)
    dim2 = 2
    dim3 = 2

    mask = np.random.binomial(1, 0.5, m).astype(np.int32)

    if sum(mask == 0):
        mask[0] = 1
    mask_indices = np.nonzero(mask)[0]

    cov = CovKroneckerFactored(sizes=[dim1, dim2, dim3], mask=mask)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        L1 = (cov.L[0]).eval(session=sess)
        L2 = (cov.L[1]).eval(session=sess)
        L3 = (cov.L[2]).eval(session=sess)
        cov_np_factor = np.kron(L1, np.kron(L2, L3))[np.ix_(mask_indices,
                                                            mask_indices)]
        cov_np = np.dot(cov_np_factor, cov_np_factor.transpose())
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X[mask_indices, :],
                                                      cov_np)

        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol,
                        atol=atol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess)[
            np.ix_(mask_indices, mask_indices)], rtol=rtol, atol=atol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess)[
            mask_indices, :], rtol=rtol, atol=atol)


def test_CovAR1():

    cov = CovAR1(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = np.linalg.inv(cov.Sigma_inv.eval(session=sess))

        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)


def test_CovAR1_scan_onsets():

    cov = CovAR1(size=m, scan_onsets=[0, m//2])

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = np.linalg.inv(cov.Sigma_inv.eval(session=sess))

        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess),
                        rtol=rtol)
