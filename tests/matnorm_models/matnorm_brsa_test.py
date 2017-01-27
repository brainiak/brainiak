from brainiak.matnorm_models import MatnormBRSA
from brsa_gendata import gen_brsa_data_from_model, gen_U_nips2016_example
from simbba.noise_covs import NoiseCovIdentity, NoiseCovDiagonal
from scipy.stats import norm
import numpy as np


def test_brsa_rudimentary():
    """this test is super loose"""

    # this is Mingbo's synth example from the paper
    U = gen_U_nips2016_example()

    n_T = 100
    n_V = 125
    n_nureg = 2

    spacecov_true = np.eye(n_V)

    timecov_true = np.diag(np.abs(norm.rvs(size=(n_T))))

    tr, sz = gen_brsa_data_from_model(U, n_T=n_T, n_V=n_V, n_nureg=n_nureg,
                                      space_cov=spacecov_true,
                                      time_cov=timecov_true)

    spacecov_model = NoiseCovIdentity(size=n_V)
    timecov_model = NoiseCovDiagonal(size=n_T)
    model_matnorm = MatnormBRSA(n_TRs=n_T, n_V=n_V, n_C=16, n_nureg=n_nureg,
                                time_noise_cov=timecov_model,
                                space_noise_cov=spacecov_model, learnRate=0.1)

    model_matnorm.fit(tr['Y'], tr['X'], max_iter=10000)

    RMSE = np.mean((model_matnorm.U_ - tr['U'])**2)**0.5

    assert(RMSE < 0.1)
