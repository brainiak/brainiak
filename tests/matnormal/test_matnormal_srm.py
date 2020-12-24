from brainiak.matnormal.srm_margs import MNSRM

import numpy as np

from scipy.stats import pearsonr

import tensorflow as tf


def test_mnsrm_margs():
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

    X = []
    W = []

    for subject in range(subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    model = MNSRM(n_features=features)
    assert model, "Cannot instantiate MNSRM!"

    # test that it works without svd init
    model.fit(X, n_iter=5, svd_init=False)

    assert model.s_.shape == (features, samples), "S wrong shape!"

    for i in range(subjects):
        assert model.w_[i].shape == (voxels, features), f"W[{i}] wrong shape!"

    assert model.rho_.shape[0] == subjects, "rho wrong shape!"

    # check that reconstruction isn't terrible
    reconstructions = [model.w_[i] @ model.s_ for i in range(subjects)]
    corrs = [pearsonr(r.flatten(), x.flatten())[0]
             for r, x in zip(reconstructions, X)]
    for corr in corrs:
        assert corr > 0.9, "Reconstruction with svd_init=False is bad! "

    model = MNSRM(n_features=features)

    # test that it works with svd init
    model.fit(X, n_iter=5, svd_init=True)

    assert model.s_.shape == (features, samples), "S wrong shape!"

    for i in range(subjects):
        assert model.w_[i].shape == (voxels, features), f"W[{i}] wrong shape!"

    assert model.rho_.shape[0] == subjects, "rho wrong shape!"

    # check that reconstruction isn't terrible
    reconstructions = [model.w_[i] @ model.s_ for i in range(subjects)]
    corrs = [pearsonr(r.flatten(), x.flatten())[0]
             for r, x in zip(reconstructions, X)]
    for corr in corrs:
        assert corr > 0.9, "Reconstruction svd_init=True is bad! "
