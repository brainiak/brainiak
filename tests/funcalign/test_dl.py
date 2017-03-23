#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pytest

def test_instance():
    import os
    import brainiak.funcalign.dl

    model = brainiak.funcalign.dl.MSDL()
    assert model, "Invalid MSDL instance!"


def test_wrong_input():
    from sklearn.utils.validation import NotFittedError
    import numpy as np
    import brainiak.funcalign.dl

    voxels_x = 5
    voxels_y = 5
    voxels_z = 5
    voxels = voxels_x * voxels_y *voxels_z
    samples = 500
    subjects = 2
    features = 3
    n_labels = 4

    model = brainiak.funcalign.dl.MSDL(n_iter=5, factors=features, lam=0.1, mu=0.1)
    assert model, "Invalid MSDL instance!"

    # Create a Shared signal S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))
    X = []
    W = []
    Q, _ = np.linalg.qr(np.random.random((voxels, features)))
    W.append(Q)
    X.append(Q.dot(S) + 0.1 * np.random.random((voxels, samples)))

    RX, RY, RZ = np.meshgrid(np.arange(voxels_x), np.arange(voxels_y), np.arange(voxels_z))
    R = np.hstack((RX.flatten()[:,np.newaxis], RY.flatten()[:,np.newaxis], RZ.flatten()[:,np.newaxis]))

    # Check that transform does NOT run before fitting the model
    with pytest.raises(NotFittedError) as excinfo:
        model.transform(X)
    print("Test: transforming before fitting the model")

    # Check that it does NOT run with 1 subject on X
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, None, R)
    print("Test: running MSDL with 1 subject (alignment)")

    for subject in range(1, subjects):
        Q, _ = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that runs with 2 subject
    model.fit(X)

    assert len(model.Vs_) == subjects, "Invalid computation of MSDL! (wrong # subjects in Vs)"
    assert len(model.Us_) == subjects, "Invalid computation of MSDL! (wrong # subjects in Us)"
    assert model.V_.shape[0] == voxels, "Invalid computation of MSDL! (wrong # voxels in V)"
    assert model.V_.shape[1] == features, "Invalid computation of MSDL! (wrong # features in V)"
    for subject in range(subjects):
        assert model.Vs_[subject].shape[0] == voxels, "Invalid computation of MSDL! (wrong # voxels in Vs)"
        assert model.Vs_[subject].shape[1] == features, "Invalid computation of MSDL! (wrong # features in Vs)"
        assert model.Us_[subject].shape[1] == features, "Invalid computation of MSDL! (wrong # features in Us)"
        assert model.Us_[subject].shape[0] == samples, "Invalid computation of MSDL! (wrong # samples in Us)"

    # Check that it does run to compute the loadings after the model computation
    new_Us = model.transform(X)

    assert len(new_Us) == subjects, "Invalid computation of MSDL! (wrong # subjects after transform)"
    for subject in range(subjects):
        assert new_Us[subject].shape[1] == features, "Invalid computation of MSDL! (wrong # features after transform)"
        assert new_Us[subject].shape[0] == samples, "Invalid computation of MSDL! (wrong # samples after transform)"

    # Check that it does NOT run with non-matching number of subjects
    with pytest.raises(ValueError) as excinfo:
        model.transform(X[1])
    print("Test: transforming with non-matching number of subjects")

    # Check that it does not run without enough samples (TRs).
    with pytest.raises(ValueError) as excinfo:
        model.set_params(features=(samples+1))
        model.fit(X)
    print("Test: not enough samples")

    # Check that it does not run with different number of samples (TRs)
    S2 = S[:,:-2]
    X.append(Q.dot(S2))
    with pytest.raises(ValueError) as excinfo:
        model.fit(X)
    print("Test: different number of samples per subject")

    # Check that kappa is in (0,1) range
    model_bad = brainiak.funcalign.dl.MSDL(n_iter=1, factors=features, kappa=1.5)
    assert model_bad, "Invalid MSDL instance!"
    with pytest.raises(ValueError) as excinfo:
        model_bad.fit(X, None, R)
    print("Test: running MSDL with wrong kappa (> 1.0)")

    model_bad = brainiak.funcalign.dl.MSDL(n_iter=1, factors=features, kappa=-0.5)
    assert model_bad, "Invalid MSDL instance!"
    with pytest.raises(ValueError) as excinfo:
        model_bad.fit(X, None, R)
    print("Test: running MSDL with wrong kappa (< 0.0)")

    # Check that mu is positive
    model_bad = brainiak.funcalign.dl.MSDL(n_iter=1, factors=features, mu=-0.1)
    assert model_bad, "Invalid MSDL instance!"
    with pytest.raises(ValueError) as excinfo:
        model_bad.fit(X, None, R)
    print("Test: running MSDL with wrong mu")

    # Check that lambda is positive
    model_bad = brainiak.funcalign.dl.MSDL(n_iter=1, factors=features, lam=-0.1)
    assert model_bad, "Invalid MSDL instance!"
    with pytest.raises(ValueError) as excinfo:
        model_bad.fit(X, None, R)
    print("Test: running MSDL with wrong lambda")

