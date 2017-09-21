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
from sklearn.exceptions import NotFittedError
import pytest


def test_can_instantiate():
    import brainiak.funcalign.srm
    s = brainiak.funcalign.srm.SRM()
    assert s, "Invalid SRM instance!"

    import numpy as np
    np.random.seed(0)

    voxels = 100
    samples = 500
    subjects = 2
    features = 3

    s = brainiak.funcalign.srm.SRM(n_iter=5, features=features)
    assert s, "Invalid SRM instance!"

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))

    X = []
    W = []
    Q, R = np.linalg.qr(np.random.random((voxels, features)))
    W.append(Q)
    X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that transform does NOT run before fitting the model
    with pytest.raises(NotFittedError):
        s.transform(X)
    print("Test: transforming before fitting the model")

    # Check that it does NOT run with 1 subject
    with pytest.raises(ValueError):
        s.fit(X)
    print("Test: running SRM with 1 subject")

    for subject in range(1, subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that runs with 2 subject
    s.fit(X)
    from pathlib import Path
    sr_v0_4 = np.load(Path(__file__).parent / "sr_v0_4.npz")['sr']
    assert(np.allclose(sr_v0_4, s.s_))

    assert len(s.w_) == subjects, (
        "Invalid computation of SRM! (wrong # subjects in W)")
    for subject in range(subjects):
        assert s.w_[subject].shape[0] == voxels, (
            "Invalid computation of SRM! (wrong # voxels in W)")
        assert s.w_[subject].shape[1] == features, (
            "Invalid computation of SRM! (wrong # features in W)")
        ortho = np.linalg.norm(s.w_[subject].T.dot(s.w_[subject])
                               - np.eye(s.w_[subject].shape[1]),
                               'fro')
        assert ortho < 1e-7, "A Wi mapping is not orthonormal in SRM."
        difference = np.linalg.norm(X[subject] - s.w_[subject].dot(s.s_),
                                    'fro')
        datanorm = np.linalg.norm(X[subject], 'fro')
        assert difference/datanorm < 1.0, "Model seems incorrectly computed."
    assert s.s_.shape[0] == features, (
        "Invalid computation of SRM! (wrong # features in S)")
    assert s.s_.shape[1] == samples, (
        "Invalid computation of SRM! (wrong # samples in S)")

    # Check that it does run to compute the shared response after the model
    # computation
    new_s = s.transform(X)

    assert len(new_s) == subjects, (
        "Invalid computation of SRM! (wrong # subjects after transform)")
    for subject in range(subjects):
        assert new_s[subject].shape[0] == features, (
            "Invalid computation of SRM! (wrong # features after transform)")
        assert new_s[subject].shape[1] == samples, (
            "Invalid computation of SRM! (wrong # samples after transform)")

    # Check that it does NOT run with non-matching number of subjects
    with pytest.raises(ValueError):
        s.transform(X[1])
    print("Test: transforming with non-matching number of subjects")

    # Check that it does not run without enough samples (TRs).
    with pytest.raises(ValueError):
        s.set_params(features=(samples+1))
        s.fit(X)
    print("Test: not enough samples")

    # Check that it does not run with different number of samples (TRs)
    S2 = S[:, :-2]
    X.append(Q.dot(S2))
    with pytest.raises(ValueError):
        s.fit(X)
    print("Test: different number of samples per subject")


def test_det_srm():
    import brainiak.funcalign.srm
    model = brainiak.funcalign.srm.DetSRM()
    assert model, "Invalid DetSRM instance!"

    import numpy as np

    voxels = 100
    samples = 500
    subjects = 2
    features = 3

    model = brainiak.funcalign.srm.DetSRM(n_iter=5, features=features)
    assert model, "Invalid DetSRM instance!"

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))

    X = []
    W = []
    Q, R = np.linalg.qr(np.random.random((voxels, features)))
    W.append(Q)
    X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that transform does NOT run before fitting the model
    with pytest.raises(NotFittedError):
        model.transform(X)
    print("Test: transforming before fitting the model")

    # Check that it does NOT run with 1 subject
    with pytest.raises(ValueError):
        model.fit(X)
    print("Test: running DetSRM with 1 subject")

    for subject in range(1, subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that runs with 2 subject
    model.fit(X)

    assert len(model.w_) == subjects, (
        "Invalid computation of DetSRM! (wrong # subjects in W)")
    for subject in range(subjects):
        assert model.w_[subject].shape[0] == voxels, (
            "Invalid computation of DetSRM! (wrong # voxels in W)")
        assert model.w_[subject].shape[1] == features, (
            "Invalid computation of DetSRM! (wrong # features in W)")
        ortho = np.linalg.norm(model.w_[subject].T.dot(model.w_[subject])
                               - np.eye(model.w_[subject].shape[1]),
                               'fro')
        assert ortho < 1e-7, "A Wi mapping is not orthonormal in DetSRM."
        difference = np.linalg.norm(X[subject]
                                    - model.w_[subject].dot(model.s_),
                                    'fro')
        datanorm = np.linalg.norm(X[subject], 'fro')
        assert difference/datanorm < 1.0, "Model seems incorrectly computed."
    assert model.s_.shape[0] == features, (
        "Invalid computation of DetSRM! (wrong # features in S)")
    assert model.s_.shape[1] == samples, (
        "Invalid computation of DetSRM! (wrong # samples in S)")

    # Check that it does run to compute the shared response after the model
    # computation
    new_s = model.transform(X)

    assert len(new_s) == subjects, (
        "Invalid computation of DetSRM! (wrong # subjects after transform)")
    for subject in range(subjects):
        assert new_s[subject].shape[0] == features, (
            "Invalid computation of DetSRM! (wrong # features after "
            "transform)")
        assert new_s[subject].shape[1] == samples, (
            "Invalid computation of DetSRM! (wrong # samples after transform)")

    # Check that it does NOT run with non-matching number of subjects
    with pytest.raises(ValueError):
        model.transform(X[1])
    print("Test: transforming with non-matching number of subjects")

    # Check that it does not run without enough samples (TRs).
    with pytest.raises(ValueError):
        model.set_params(features=(samples+1))
        model.fit(X)
    print("Test: not enough samples")

    # Check that it does not run with different number of samples (TRs)
    S2 = S[:, :-2]
    X.append(Q.dot(S2))
    with pytest.raises(ValueError):
        model.fit(X)
    print("Test: different number of samples per subject")
