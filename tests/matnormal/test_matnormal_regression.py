import pytest
import numpy as np
from scipy.stats import norm, wishart, pearsonr

from brainiak.matnormal.covs import (
    CovIdentity,
    CovUnconstrainedCholesky,
    CovUnconstrainedInvCholesky,
    CovDiagonal,
)
from brainiak.matnormal.regression import MatnormalRegression
from brainiak.matnormal.utils import rmn

m = 100
n = 4
p = 5

corrtol = 0.8  # at least this much correlation between true and est to pass


def test_matnorm_regression_unconstrained(seeded_rng):

    # Y = XB + eps
    # Y is m x p, B is n x p, eps is m x p
    X = norm.rvs(size=(m, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)
    rowcov_true = np.eye(m)
    colcov_true = wishart.rvs(p + 2, np.eye(p))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = CovIdentity(size=m)
    col_cov = CovUnconstrainedCholesky(size=p)

    model = MatnormalRegression(time_cov=row_cov, space_cov=col_cov)

    model.fit(X, Y, naive_init=False)

    assert pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol

    pred_y = model.predict(X)
    assert pearsonr(pred_y.flatten(), Y_hat.flatten())[0] >= corrtol

    model = MatnormalRegression(time_cov=row_cov, space_cov=col_cov)

    model.fit(X, Y, naive_init=True)

    assert pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol

    pred_y = model.predict(X)
    assert pearsonr(pred_y.flatten(), Y_hat.flatten())[0] >= corrtol


def test_matnorm_regression_unconstrainedprec(seeded_rng):

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    X = norm.rvs(size=(m, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)
    rowcov_true = np.eye(m)
    colcov_true = wishart.rvs(p + 2, np.eye(p))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = CovIdentity(size=m)
    col_cov = CovUnconstrainedInvCholesky(size=p)

    model = MatnormalRegression(time_cov=row_cov, space_cov=col_cov)

    model.fit(X, Y, naive_init=False)

    assert pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol

    pred_y = model.predict(X)
    assert pearsonr(pred_y.flatten(), Y_hat.flatten())[0] >= corrtol


def test_matnorm_regression_optimizerChoice(seeded_rng):

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    X = norm.rvs(size=(m, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)
    rowcov_true = np.eye(m)
    colcov_true = wishart.rvs(p + 2, np.eye(p))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = CovIdentity(size=m)
    col_cov = CovUnconstrainedInvCholesky(size=p)

    model = MatnormalRegression(time_cov=row_cov, space_cov=col_cov,
                                optimizer="CG")

    model.fit(X, Y, naive_init=False)

    assert pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol

    pred_y = model.predict(X)
    assert pearsonr(pred_y.flatten(), Y_hat.flatten())[0] >= corrtol


def test_matnorm_regression_scaledDiag(seeded_rng):

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    X = norm.rvs(size=(m, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)

    rowcov_true = np.eye(m)
    colcov_true = np.diag(np.abs(norm.rvs(size=p)))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = CovIdentity(size=m)
    col_cov = CovDiagonal(size=p)

    model = MatnormalRegression(time_cov=row_cov, space_cov=col_cov)

    model.fit(X, Y, naive_init=False)

    assert pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol

    pred_y = model.predict(X)
    assert pearsonr(pred_y.flatten(), Y_hat.flatten())[0] >= corrtol

    # we only do calibration test on the scaled diag
    # model because to hit corrtol on unconstrainedCov
    # we'd need a lot more data, which would make the test slow
    X_hat = model.calibrate(Y)
    assert pearsonr(X_hat.flatten(), X.flatten())[0] >= corrtol


def test_matnorm_calibration_raises(seeded_rng):

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    X = norm.rvs(size=(2, 5))
    B = norm.rvs(size=(5, 3))
    Y_hat = X.dot(B)

    rowcov_true = np.eye(2)
    colcov_true = np.diag(np.abs(norm.rvs(size=3)))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = CovIdentity(size=2)
    col_cov = CovDiagonal(size=3)

    model = MatnormalRegression(time_cov=row_cov, space_cov=col_cov)

    model.fit(X, Y, naive_init=False)

    with pytest.raises(RuntimeError):
        model.calibrate(Y)
