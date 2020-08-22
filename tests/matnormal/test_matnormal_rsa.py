from brainiak.matnormal.mnrsa import MNRSA
from brainiak.utils.utils import cov2corr
from brainiak.matnormal.covs import CovIdentity, CovDiagonal
from scipy.stats import norm
from brainiak.matnormal.utils import rmn
import numpy as np


def gen_U_nips2016_example():

    n_C = 16
    U = np.zeros([n_C, n_C])
    U = np.eye(n_C) * 0.6
    U[8:12, 8:12] = 0.8
    for cond in range(8, 12):
        U[cond, cond] = 1

    return U


def gen_brsa_data_matnorm_model(U, n_T, n_V, space_cov, time_cov, n_nureg):

    n_C = U.shape[0]
    beta = rmn(U, space_cov)
    X = rmn(np.eye(n_T), np.eye(n_C))
    beta_0 = rmn(np.eye(n_nureg), space_cov)
    X_0 = rmn(np.eye(n_T), np.eye(n_nureg))
    Y_hat = X.dot(beta) + X_0.dot(beta_0)
    Y = Y_hat + rmn(time_cov, space_cov)
    sizes = {"n_C": n_C, "n_T": n_T, "n_V": n_V}
    train = {"beta": beta, "X": X, "Y": Y, "U": U, "X_0": X_0}

    return train, sizes


def test_brsa_rudimentary(seeded_rng):
    """this test is super loose"""

    # this is Mingbo's synth example from the paper
    U = gen_U_nips2016_example()

    n_T = 150
    n_V = 250
    n_nureg = 5

    spacecov_true = np.eye(n_V)

    timecov_true = np.diag(np.abs(norm.rvs(size=(n_T))))

    tr, sz = gen_brsa_data_matnorm_model(
        U,
        n_T=n_T,
        n_V=n_V,
        n_nureg=n_nureg,
        space_cov=spacecov_true,
        time_cov=timecov_true,
    )

    spacecov_model = CovIdentity(size=n_V)
    timecov_model = CovDiagonal(size=n_T)

    model_matnorm = MNRSA(time_cov=timecov_model, space_cov=spacecov_model)

    model_matnorm.fit(tr["Y"], tr["X"], naive_init=False)

    RMSE = np.mean((model_matnorm.C_ - cov2corr(tr["U"])) ** 2) ** 0.5

    assert RMSE < 0.1

    model_matnorm = MNRSA(time_cov=timecov_model, space_cov=spacecov_model)

    model_matnorm.fit(tr["Y"], tr["X"], naive_init=True)

    RMSE = np.mean((model_matnorm.C_ - cov2corr(tr["U"])) ** 2) ** 0.5

    assert RMSE < 0.1
