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
    import brainiak.funcalign.rsrm
    s = brainiak.funcalign.rsrm.RSRM()
    assert s, "Invalid RSRM instance!"

    import numpy as np
    np.random.seed(0)

    voxels = 100
    samples = 500
    subjects = 2
    features = 3

    s = brainiak.funcalign.rsrm.RSRM(n_iter=5, features=features)
    assert s, "Invalid RSRM instance!"

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
    print("Test: running RSRM with 1 subject")

    for subject in range(1, subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that runs with 2 subject
    s.fit(X)

    assert len(s.w_) == subjects, (
        "Invalid computation of RSRM! (wrong # subjects in W)")
    for subject in range(subjects):
        assert s.s_[subject].shape[0] == voxels, (
            "Invalid computation of RSRM! (wrong # voxels in S)")
        assert s.s_[subject].shape[1] == samples, (
            "Invalid computation of RSRM! (wrong # samples in S)")
        assert s.w_[subject].shape[0] == voxels, (
            "Invalid computation of RSRM! (wrong # voxels in W)")
        assert s.w_[subject].shape[1] == features, (
            "Invalid computation of RSRM! (wrong # features in W)")
        ortho = np.linalg.norm(s.w_[subject].T.dot(s.w_[subject])
                               - np.eye(s.w_[subject].shape[1]),
                               'fro')
        assert ortho < 1e-7, "A Wi mapping is not orthonormal in RSRM."
        difference = np.linalg.norm(X[subject] - s.w_[subject].dot(s.r_),
                                    'fro')
        datanorm = np.linalg.norm(X[subject], 'fro')
        assert difference/datanorm < 1.0, "Model seems incorrectly computed."
    assert s.r_.shape[0] == features, (
        "Invalid computation of RSRM! (wrong # features in R)")
    assert s.r_.shape[1] == samples, (
        "Invalid computation of RSRM! (wrong # samples in R)")

    # Check that it does run to compute the shared response after the model
    # computation
    new_r, _ = s.transform(X)

    assert len(new_r) == subjects, (
        "Invalid computation of RSRM! (wrong # subjects after transform)")
    for subject in range(subjects):
        assert new_r[subject].shape[0] == features, (
            "Invalid computation of RSRM! (wrong # features after transform)")
        assert new_r[subject].shape[1] == samples, (
            "Invalid computation of RSRM! (wrong # samples after transform)")

    # Check that it does run to compute a new subject
    new_w, new_s = s.transform_subject(X[0])
    assert new_w.shape[1] == features, (
            "Invalid computation of RSRM! (wrong # features for new subject)")
    assert new_s.shape[1] == samples, (
            "Invalid computation of RSRM! (wrong # samples for new subject)")
    assert new_s.shape[0] == voxels, (
            "Invalid computation of RSRM! (wrong # voxels for new subject)")
    assert new_w.shape[0] == voxels, (
            "Invalid computation of RSRM! (wrong # voxels for new subject)")

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
