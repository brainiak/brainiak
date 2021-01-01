import pytest
from brainiak.matnormal.srm_margs import MNSRM
from brainiak.matnormal.srm_margw import DPMNSRM
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

    def make_noise(noise_distr='iid', noise_scale=0.1):
        if noise_distr == 'iid':
            noise = noise_scale*np.random.random((voxels, samples))
        elif noise_distr == "unconstrained":
            space_chol = np.linalg.cholesky(wishart.rvs(df=voxels+2, scale=np.eye(voxels)))
            time_chol = np.linalg.cholesky(wishart.rvs(df=samples+2, scale=np.eye(samples)))
            noise = noise_scale * space_chol @ np.random.random((voxels, samples)) @ time_chol
        return noise
    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))

    X = []
    W = []

    for subject in range(subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + make_noise())

    data = X, W, S
    sizes = voxels, samples, features, subjects
    return data, sizes


@pytest.mark.parametrize("svd_init", [True, False])
def test_mnsrm_margs(mnsrm_fakedata, svd_init):
    # Test that MNSRM-MargS

    data, sizes = mnsrm_fakedata
    X, W, S = data
    voxels, samples, features, subjects = sizes

    model = MNSRM(n_features=features)
    assert model, "Cannot instantiate MNSRM!"

    model.fit(X, n_iter=5, svd_init=svd_init)

    assert model.s_.shape == (features, samples), "S wrong shape!"

    for i in range(subjects):
        assert model.w_[i].shape == (voxels, features), f"W[{i}] wrong shape!"

    assert model.rho_.shape[0] == subjects, "rho wrong shape!"

    # check that reconstruction isn't terrible
    reconstructions = [model.w_[i] @ model.s_ for i in range(subjects)]
    corrs = [pearsonr(r.flatten(), x.flatten())[0]
             for r, x in zip(reconstructions, X)]
    for corr in corrs:
        assert corr > 0.9, f"Reconstruction with svd_init={svd_init} is bad!"


@pytest.mark.parametrize("svd_init,algo,s_constraint,space_cov,time_cov", 
                         itertools.product([True, False], ["ECM", "ECME"],
                         ['gaussian','ortho'], [CovIdentity,CovIsotropic],
                         [CovIdentity, CovAR1]))
def test_mnsrm_margw(mnsrm_fakedata, svd_init, algo, s_constraint, 
                     space_cov, time_cov):
    """ DPMNSRM test
    """
    
    data, sizes = mnsrm_fakedata
    X, W, S = data
    voxels, samples, features, subjects = sizes
    
    if s_constraint == "ortho":
        w_cov = CovUnconstrainedCholesky
    else:
        w_cov = CovIdentity

    model = DPMNSRM(n_features=features,
                    s_constraint=s_constraint, algorithm=algo,
                    time_noise_cov=time_cov, w_cov=w_cov, space_noise_cov=space_cov)
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
        assert corr > 0.8, f"Reconstruction corr={corr}<0.8 (svd_init={svd_init} algo={algo} s_constraint={s_constraint}  space_cov={space_cov} time_cov={time_cov})"
 