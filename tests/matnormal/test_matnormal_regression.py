import numpy as np
from scipy.stats import norm, wishart, pearsonr
from brainiak.matnormal.noise_covs import\
    NoiseCovIdentity, NoiseCovFullRank, NoisePrecFullRank, NoiseCovDiagonal
from brainiak.matnormal import MatnormRegression
from brainiak.matnormal.helpers import rmn
import pytest

m = 100
n = 4
p = 5

corrtol = 0.8  # at least this much correlation between true and est to pass


def test_matnorm_regression_fullrank():

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    X = norm.rvs(size=(m, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)
    rowcov_true = np.eye(m)
    colcov_true = wishart.rvs(p+2, np.eye(p))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = NoiseCovIdentity(size=m)
    col_cov = NoiseCovFullRank(size=p)

    model = MatnormRegression(n_v=p, n_c=n, time_noise_cov=row_cov,
                              space_noise_cov=col_cov, learnRate=0.01)

    model.fit(X, Y, max_iter=10000, step=10000)

    assert(pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol)


def test_matnorm_regression_fullrankprec():

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    X = norm.rvs(size=(m, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)
    rowcov_true = np.eye(m)
    colcov_true = wishart.rvs(p+2, np.eye(p))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = NoiseCovIdentity(size=m)
    col_cov = NoisePrecFullRank(size=p)

    model = MatnormRegression(n_v=p, n_c=n, time_noise_cov=row_cov,
                              space_noise_cov=col_cov, learnRate=0.01)

    model.fit(X, Y, max_iter=10000, step=10000)

    assert(pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol)


def test_matnorm_regression_scaledDiag():

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    X = norm.rvs(size=(m, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)

    rowcov_true = np.eye(m)
    colcov_true = np.diag(np.abs(norm.rvs(size=p)))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = NoiseCovIdentity(size=m)
    col_cov = NoiseCovDiagonal(size=p)

    model = MatnormRegression(n_v=p, n_c=n, time_noise_cov=row_cov,
                              space_noise_cov=col_cov, learnRate=0.01)

    model.fit(X, Y, max_iter=10000, step=10000)

    assert(pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol)


def test_matnorm_regression_predict_calibrate():

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    X = norm.rvs(size=(m*2, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)
    rowcov_true = np.eye(m*2)
    colcov_true = np.diag(np.abs(norm.rvs(size=p)))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = NoiseCovIdentity(size=m)
    col_cov = NoiseCovDiagonal(size=p)

    Y_train = Y[:m, :]
    Y_test = Y[m:, :]

    X_train = X[:m, :]
    X_test = X[m:, :]

    model = MatnormRegression(n_v=p, n_c=n, time_noise_cov=row_cov,
                              space_noise_cov=col_cov, learnRate=0.01)

    model.fit(X_train, Y_train, max_iter=10000, step=10000)

    Yhat_test = model.predict(X=X_test)

    assert(pearsonr(Yhat_test.flatten(), Y_test.flatten())[0] >= corrtol)

    Xhat_test = model.calibrate(Y=Y_test)

    assert(pearsonr(Xhat_test.flatten(), X_test.flatten())[0] >= corrtol)


def test_matnorm_regression_raises_calibrate_rank_defficient():

    m = 100
    n = 4
    p = 3

    X = norm.rvs(size=(m*2, n))
    B = norm.rvs(size=(n, p))
    Y_hat = X.dot(B)
    rowcov_true = np.eye(m*2)
    colcov_true = np.diag(np.abs(norm.rvs(size=p)))

    Y = Y_hat + rmn(rowcov_true, colcov_true)

    row_cov = NoiseCovIdentity(size=m)
    col_cov = NoiseCovDiagonal(size=p)

    Y_train = Y[:m, :]
    Y_test = Y[m:, :]

    X_train = X[:m, :]

    model = MatnormRegression(n_v=p, n_c=n, time_noise_cov=row_cov,
                              space_noise_cov=col_cov, learnRate=0.01)

    model.fit(X_train, Y_train, max_iter=10000, step=10000)

    with pytest.raises(RuntimeError):
        model.calibrate(Y=Y_test)
