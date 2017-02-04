import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm, wishart
from brainiak.matnormal.covs import *
import tensorflow as tf
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)

# X is m x n, so A sould be m x p

m = 8
n = 4
p = 3

rtol = 1e-7


def logdet_sinv_np(X, sigma):
    # logdet
    _, logdet_np = np.linalg.slogdet(sigma)
    # sigma-inv
    sinv_np = np.linalg.inv(sigma)
    # sigma-inverse *
    sinvx_np = sinv_np.dot(X)
    return logdet_np, sinv_np, sinvx_np

X = norm.rvs(size=(m, n))
X_tf = tf.constant(X)
A = norm.rvs(size=(m, p))
A_tf = tf.constant(A)


def test_CovConstant():

    cov_np = wishart.rvs(df=m+2, scale=np.eye(m))
    cov = CovConstant(cov_np)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)


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
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)


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
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)


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
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)


def test_CovFullRankCholesky():

    cov = CovFullRankCholesky(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.Sigma.eval(session=sess)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)


def test_CovFullRankInvCholesky():

    cov = CovFullRankInvCholesky(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.Sigma.eval(session=sess)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)

def test_Cov2FactorKron():
    assert(m%2 == 0)
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
        cov_np = np.kron(np.dot(L1, L1.transpose()), np.dot(L2, L2.transpose()))
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)

        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)

def test_Cov3FactorKron():
    assert(m%4 == 0)
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
        cov_np = np.kron(np.kron(np.dot(L1, L1.transpose()),\
                         np.dot(L2, L2.transpose())), np.dot(L3, L3.transpose()))
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)

        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)

