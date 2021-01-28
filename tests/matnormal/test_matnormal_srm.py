import pytest
from brainiak.matnormal.dpsrm import DPMNSRM
from brainiak.matnormal.covs import CovIdentity, CovAR1, CovIsotropic, CovUnconstrainedCholesky

import numpy as np
import itertools

from scipy.stats import pearsonr, wishart

import tensorflow as tf

@pytest.fixture
def mnsrm_fakedata(): 
    np.random.seed(1)
    tf.random.set_seed(1)
    voxels = 10
    samples = 50
    subjects = 2
    features = 3

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))
    
    rho = (0.1+np.random.normal(subjects)) ** 2

    X = []
    W = []

    for subject in range(subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + rho*np.random.random((voxels, samples)))

    data = X, W, S
    sizes = voxels, samples, features, subjects
    return data, sizes



@pytest.mark.parametrize("svd_init,algo,space_cov,time_cov", 
                         itertools.product([True, False], ["ECM", "ECME"],
                         [CovIdentity],
                         [CovIdentity]))
def test_mnsrm_margw(mnsrm_fakedata, svd_init, algo,
                     space_cov, time_cov):
    """ DPMNSRM test
    """
    
    data, sizes = mnsrm_fakedata
    X, W, S = data
    voxels, samples, features, subjects = sizes
    
    model = DPMNSRM(n_features=features,algorithm=algo,
                    time_noise_cov=time_cov, space_noise_cov=space_cov)
    assert model, "Cannot instantiate DPMNSRM!"
    model.fit(X, max_iter=10, svd_init=svd_init, rtol=0.01, gtol=1e-3)

    assert model.s_.shape == (features, samples), "S wrong shape!"

    for i in range(subjects):
        assert model.w_[i].shape == (voxels, features), f"W[{i}] wrong shape!"

    assert model.rho_.shape[0] == subjects, "rho wrong shape!"

    # check that reconstruction isn't terrible
    reconstructions = [model.w_[i] @ model.s_ for i in range(subjects)]
    corrs = [pearsonr(r.flatten(), x.flatten())[0]
             for r, x in zip(reconstructions, X)]
    for corr in corrs:
        assert corr > 0.8, f"Reconstruction corr={corr}<0.8 (svd_init={svd_init} algo={algo} space_cov={space_cov} time_cov={time_cov})"
 