import numpy as np 
from numpy.testing import assert_allclose
from scipy.stats import norm, wishart
from brainiak.matnorm_models.noise_covs import *
from brainiak.matnorm_models.kron_covs import *
import tensorflow as tf

# X is m x n, so A sould be m x p

m = 6
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

def test_NoiseCovConstant():
    
    cov_np = wishart.rvs(df=m+2, scale=np.eye(m))
    cov = NoiseCovConstant(cov_np)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)        


def test_NoiseCovIdentity():
    
    cov = NoiseCovIdentity(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = np.eye(m)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)
        

def test_NoiseCovIsotropic():

    cov = NoiseCovIsotropic(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.sigma.eval(session=sess) * np.eye(cov.size)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)


def test_NoiseCovDiagonal():

    cov = NoiseCovDiagonal(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = np.diag(1/cov.prec.eval(session=sess))
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)

def test_NoiseCovFullRank():

    cov = NoiseCovFullRank(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.Sigma.eval(session=sess)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)

def test_NoisePrecFullRank():

    cov = NoisePrecFullRank(size=m)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        cov_np = cov.Sigma.eval(session=sess)
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)

def test_NoiseCov2FactorKron():
    assert(m%2 == 0)
    dim1 = int(m/2)
    dim2 = 2
    cov = NoiseCov2FactorKron(size1=dim1, size2=dim2)

    with tf.Session() as sess:
        # initialize the random covariance
        sess.run(tf.variables_initializer(cov.get_optimize_vars()))
        # compute the naive version
        L1 = (cov.L1).eval(session=sess)
        L2 = (cov.L2).eval(session=sess)
        cov_np = np.kron(np.dot(L1, L1.transpose()), np.dot(L2, L2.transpose()) )
        logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)

        assert_allclose(logdet_np, cov.logdet.eval(session=sess), rtol=rtol)
        assert_allclose(sinv_np, cov.Sigma_inv.eval(session=sess), rtol=rtol)
        assert_allclose(sinvx_np, cov.Sigma_inv_x(X_tf).eval(session=sess), rtol=rtol)
