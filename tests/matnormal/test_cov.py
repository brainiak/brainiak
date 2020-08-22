import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm, wishart, invgamma, invwishart

import tensorflow as tf

from brainiak.matnormal.covs import (
    CovIdentity,
    CovAR1,
    CovIsotropic,
    CovDiagonal,
    CovDiagonalGammaPrior,
    CovUnconstrainedCholesky,
    CovUnconstrainedCholeskyWishartReg,
    CovUnconstrainedInvCholesky,
    CovKroneckerFactored,
)

# X is m x n, so A sould be m x p

m = 8
n = 4
p = 3

rtol = 1e-7
atol = 1e-7


def logdet_sinv_np(X, sigma):
    # logdet
    sign, logdet = np.linalg.slogdet(sigma)
    logdet_np = sign * logdet
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
eye = tf.eye(m, dtype=tf.float64)


def test_CovConstant(seeded_rng):

    cov_np = wishart.rvs(df=m + 2, scale=np.eye(m))
    cov = CovUnconstrainedCholesky(Sigma=cov_np)

    # verify what we pass is what we get
    cov_tf = cov._cov
    assert_allclose(cov_tf, cov_np)

    # compute the naive version
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_CovIdentity(seeded_rng):

    cov = CovIdentity(size=m)

    # compute the naive version
    cov_np = np.eye(m)
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_CovIsotropic(seeded_rng):

    cov = CovIsotropic(size=m)

    # compute the naive version
    cov_np = cov._cov * np.eye(cov.size)
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)

    # test initialization
    cov = CovIsotropic(var=0.123, size=3)
    assert_allclose(np.exp(cov.log_var.numpy()), 0.123)


def test_CovDiagonal(seeded_rng):

    cov = CovDiagonal(size=m)

    # compute the naive version
    cov_np = cov._cov
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_CovDiagonal_initialized(seeded_rng):

    cov_np = np.diag(np.exp(np.random.normal(size=m)))
    cov = CovDiagonal(size=m, diag_var=np.diag(cov_np))

    # compute the naive version
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_CovDiagonalGammaPrior(seeded_rng):

    cov_np = np.diag(np.exp(np.random.normal(size=m)))
    cov = CovDiagonalGammaPrior(size=m, sigma=np.diag(cov_np), alpha=1.5,
                                beta=1e-10)

    ig = invgamma(1.5, scale=1e-10)

    # compute the naive version
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    penalty_np = np.sum(ig.logpdf(1 / np.diag(cov_np)))
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)
    assert_allclose(penalty_np, cov.logp, rtol=rtol)


def test_CovUnconstrainedCholesky(seeded_rng):

    cov = CovUnconstrainedCholesky(size=m)

    L = cov.L.numpy()
    cov_np = L @ L.T
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_CovUnconstrainedCholeskyWishartReg(seeded_rng):

    cov = CovUnconstrainedCholeskyWishartReg(size=m)

    L = cov.L.numpy()
    cov_np = L @ L.T

    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)
    # now compute the regularizer
    reg = wishart.logpdf(cov_np, df=m + 2, scale=1e10 * np.eye(m))
    assert_allclose(reg, cov.logp, rtol=rtol)


def test_CovUnconstrainedInvCholesky(seeded_rng):

    init = invwishart.rvs(scale=np.eye(m), df=m + 2)
    cov = CovUnconstrainedInvCholesky(invSigma=init)

    Linv = cov.Linv
    L = np.linalg.inv(Linv)
    cov_np = L @ L.T

    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_Cov2FactorKron(seeded_rng):
    assert m % 2 == 0
    dim1 = int(m / 2)
    dim2 = 2

    with pytest.raises(TypeError) as excinfo:
        cov = CovKroneckerFactored(sizes=dim1)
    assert "sizes is not a list" in str(excinfo.value)

    cov = CovKroneckerFactored(sizes=[dim1, dim2])

    L1 = (cov.L[0]).numpy()
    L2 = (cov.L[1]).numpy()
    cov_np = np.kron(np.dot(L1, L1.transpose()), np.dot(L2, L2.transpose()))
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)

    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_Cov3FactorKron(seeded_rng):
    assert m % 4 == 0
    dim1 = int(m / 4)
    dim2 = 2
    dim3 = 2
    cov = CovKroneckerFactored(sizes=[dim1, dim2, dim3])

    L1 = (cov.L[0]).numpy()
    L2 = (cov.L[1]).numpy()
    L3 = (cov.L[2]).numpy()
    cov_np = np.kron(
        np.kron(np.dot(L1, L1.transpose()), np.dot(L2, L2.transpose())),
        np.dot(L3, L3.transpose()),
    )
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)

    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinv_np, cov.solve(eye), rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_Cov3FactorMaskedKron(seeded_rng):
    assert m % 4 == 0
    dim1 = int(m / 4)
    dim2 = 2
    dim3 = 2

    mask = np.random.binomial(1, 0.5, m).astype(np.int32)

    if sum(mask == 0):
        mask[0] = 1
    mask_indices = np.nonzero(mask)[0]

    cov = CovKroneckerFactored(sizes=[dim1, dim2, dim3], mask=mask)

    L1 = (cov.L[0]).numpy()
    L2 = (cov.L[1]).numpy()
    L3 = (cov.L[2]).numpy()
    cov_np_factor = np.kron(L1, np.kron(L2, L3))[
                            np.ix_(mask_indices, mask_indices)]
    cov_np = np.dot(cov_np_factor, cov_np_factor.transpose())
    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X[mask_indices, :], cov_np)

    assert_allclose(logdet_np, cov.logdet, rtol=rtol, atol=atol)
    assert_allclose(
        sinv_np,
        cov.solve(eye).numpy()[np.ix_(mask_indices, mask_indices)],
        rtol=rtol,
        atol=atol,
    )
    assert_allclose(
        sinvx_np, cov.solve(X_tf).numpy()[
                            mask_indices, :], rtol=rtol, atol=atol
    )


def test_CovAR1(seeded_rng):

    cov = CovAR1(size=m)

    cov_np = np.linalg.inv(cov.solve(eye))

    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)

    # test initialization
    cov = CovAR1(rho=0.3, sigma=1.3, size=3)
    assert_allclose(np.exp(cov.log_sigma.numpy()), 1.3)
    assert_allclose((2 * tf.sigmoid(cov.rho_unc) - 1).numpy(), 0.3)


def test_CovAR1_scan_onsets(seeded_rng):

    cov = CovAR1(size=m, scan_onsets=[0, m // 2])

    # compute the naive version
    cov_np = np.linalg.inv(cov.solve(eye))

    logdet_np, sinv_np, sinvx_np = logdet_sinv_np(X, cov_np)
    assert_allclose(logdet_np, cov.logdet, rtol=rtol)
    assert_allclose(sinvx_np, cov.solve(X_tf), rtol=rtol)


def test_raises(seeded_rng):

    with pytest.raises(RuntimeError):
        CovUnconstrainedCholesky(Sigma=np.eye(3), size=4)

    with pytest.raises(RuntimeError):
        CovUnconstrainedCholesky()

    with pytest.raises(RuntimeError):
        CovUnconstrainedInvCholesky(invSigma=np.eye(3), size=4)

    with pytest.raises(RuntimeError):
        CovUnconstrainedInvCholesky()
