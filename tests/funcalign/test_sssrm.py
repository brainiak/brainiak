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
    os.environ['THEANO_FLAGS'] = 'device=cpu, floatX=float64'
    import brainiak.funcalign.sssrm

    model = brainiak.funcalign.sssrm.SSSRM()
    assert model, "Invalid SSSRM instance!"


def test_wrong_input():
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu, floatX=float64'

    from sklearn.utils.validation import NotFittedError
    import numpy as np
    import brainiak.funcalign.sssrm

    voxels = 100
    align_samples = 400
    samples = 500
    subjects = 2
    features = 3
    n_labels = 4

    model = brainiak.funcalign.sssrm.SSSRM(n_iter=5, features=features,
                                           gamma=10.0, alpha=0.1)
    assert model, "Invalid SSSRM instance!"

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))
    S_align = S[:, :align_samples]
    S_classify = S[:, align_samples:]
    X = []
    Z = []
    Z2 = []
    W = []
    y = []
    Q, R = np.linalg.qr(np.random.random((voxels, features)))
    W.append(Q)
    X.append(Q.dot(S_align) + 0.1 * np.random.random((voxels, align_samples)))
    Z.append(Q.dot(S_classify)
             + 0.1 * np.random.random((voxels, samples - align_samples)))
    Z2.append(Q.dot(S_classify)
              + 0.1 * np.random.random((voxels, samples - align_samples)))
    y.append(np.repeat(
        np.arange(n_labels), (samples - align_samples)/n_labels))

    # Check that transform does NOT run before fitting the model
    with pytest.raises(NotFittedError):
        model.transform(X)
    print("Test: transforming before fitting the model")

    # Check that predict does NOT run before fitting the model
    with pytest.raises(NotFittedError):
        model.predict(X)
    print("Test: predicting before fitting the model")

    # Check that it does NOT run with 1 subject on X
    with pytest.raises(ValueError):
        model.fit(X, y, Z)
    print("Test: running SSSRM with 1 subject (alignment)")

    # Create more subjects align and classification data
    for subject in range(1, subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S_align)
                 + 0.1 * np.random.random((voxels, align_samples)))
        Z2.append(Q.dot(S_classify)
                  + 0.1 * np.random.random((voxels, samples - align_samples)))

    # Check that it does NOT run with 1 subject on y
    with pytest.raises(ValueError):
        model.fit(X, y, Z)
    print("Test: running SSSRM with 1 subject (labels)")

    # Create more subjects labels data
    for subject in range(1, subjects):
        y.append(np.repeat(
            np.arange(n_labels), (samples - align_samples)/n_labels))

    # Check that it does NOT run with 1 subject on Z
    with pytest.raises(ValueError):
        model.fit(X, y, Z)
    print("Test: running SSSRM with 1 subject (classif.)")

    # Check that alpha is in (0,1) range
    model_bad = brainiak.funcalign.sssrm.SSSRM(n_iter=1, features=features,
                                               gamma=10.0, alpha=1.5)
    assert model_bad, "Invalid SSSRM instance!"
    with pytest.raises(ValueError):
        model_bad.fit(X, y, Z)
    print("Test: running SSSRM with wrong alpha")

    # Check that gamma is positive
    model_bad = brainiak.funcalign.sssrm.SSSRM(n_iter=1, features=features,
                                               gamma=-0.1, alpha=0.2)
    assert model_bad, "Invalid SSSRM instance!"
    with pytest.raises(ValueError):
        model_bad.fit(X, y, Z)
    print("Test: running SSSRM with wrong gamma")


def test_sssrm():
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu, floatX=float64'

    import numpy as np
    import brainiak.funcalign.sssrm

    voxels = 100
    align_samples = 400
    samples = 500
    subjects = 2
    features = 3
    n_labels = 4

    model = brainiak.funcalign.sssrm.SSSRM(n_iter=5, features=features,
                                           gamma=10.0, alpha=0.1)
    assert model, "Invalid SSSRM instance!"

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))
    S_align = S[:, :align_samples]
    S_classify = S[:, align_samples:]
    X = []
    Z = []
    Z2 = []
    y = []
    Q, R = np.linalg.qr(np.random.random((voxels, features)))
    X.append(Q.dot(S_align) + 0.1 * np.random.random((voxels, align_samples)))
    Z.append(Q.dot(S_classify)
             + 0.1 * np.random.random((voxels, samples - align_samples)))
    Z2.append(Q.dot(S_classify)
              + 0.1 * np.random.random((voxels, samples - align_samples)))
    y.append(np.repeat(
        np.arange(n_labels), (samples - align_samples)/n_labels))

    # Create more subjects align and classification data
    for subject in range(1, subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        X.append(Q.dot(S_align)
                 + 0.1 * np.random.random((voxels, align_samples)))
        Z2.append(Q.dot(S_classify)
                  + 0.1 * np.random.random((voxels, samples - align_samples)))

    # Create more subjects labels data
    for subject in range(1, subjects):
        y.append(np.repeat(
            np.arange(n_labels), (samples - align_samples)/n_labels))

    # Set the logging level to INFO
    import logging
    logging.basicConfig(level=logging.INFO)

    # Check that runs with 2 subject
    model.fit(X, y, Z2)
    print("Test: fitting SSSRM successfully")

    assert len(model.w_) == subjects, (
        "Invalid computation of SSSRM! (wrong # subjects in W)")
    for subject in range(subjects):
        assert model.w_[subject].shape[0] == voxels, (
            "Invalid computation of SSSRM! (wrong # voxels in W)")
        assert model.w_[subject].shape[1] == features, (
            "Invalid computation of SSSRM! (wrong # features in W)")
        ortho = np.linalg.norm(model.w_[subject].T.dot(model.w_[subject])
                               - np.eye(model.w_[subject].shape[1]),
                               'fro')
        assert ortho < 1e-7, "A Wi mapping is not orthonormal in SSSRM."
        difference = np.linalg.norm(X[subject]
                                    - model.w_[subject].dot(model.s_),
                                    'fro')
        datanorm = np.linalg.norm(X[subject], 'fro')
        assert difference/datanorm < 1.0, "Model seems incorrectly computed."
    assert model.s_.shape[0] == features, (
        "Invalid computation of SSSRM! (wrong # features in S)")
    assert model.s_.shape[1] == align_samples, (
        "Invalid computation of SSSRM! (wrong # samples in S)")

    # Check that it does run to compute the shared response after the model
    # computation
    new_s = model.transform(X)
    print("Test: transforming with SSSRM successfully")

    assert len(new_s) == subjects, (
        "Invalid computation of SSSRM! (wrong # subjects after transform)")
    for subject in range(subjects):
        assert new_s[subject].shape[0] == features, (
            "Invalid computation of SSSRM! (wrong # features after transform)")
        assert new_s[subject].shape[1] == align_samples, (
            "Invalid computation of SSSRM! (wrong # samples after transform)")

    # Check that it predicts with the model
    pred = model.predict(Z2)
    print("Test: predicting with SSSRM successfully")
    assert len(pred) == subjects, (
        "Invalid computation of SSSRM! (wrong # subjects after predict)")
    for subject in range(subjects):
        assert pred[subject].size == samples - align_samples, (
            "SSSRM: wrong # answers in predict")
        pred_labels = np.logical_and(pred[subject] >= 0,
                                     pred[subject] < n_labels)
        assert np.all(pred_labels), (
            "SSSRM: wrong class number output in predict")

    # Check that it does NOT run with non-matching number of subjects
    with pytest.raises(ValueError):
        model.transform(X[1])
    print("Test: transforming with non-matching number of subjects")

    # Check that it does not run without enough samples (TRs).
    with pytest.raises(ValueError):
        model.set_params(features=(align_samples + 1))
        model.fit(X, y, Z2)
    print("Test: not enough samples")

    # Check that it does not run with different number of samples (TRs)
    X2 = X
    X2[0] = Q.dot(S[:, :-2])
    with pytest.raises(ValueError):
        model.fit(X2, y, Z2)
    print("Test: different number of samples per subject")

    # Create one more subject
    Q, R = np.linalg.qr(np.random.random((voxels, features)))
    X.append(Q.dot(S_align) + 0.1 * np.random.random((voxels, align_samples)))
    Z2.append(Q.dot(S_classify)
              + 0.1 * np.random.random((voxels, samples - align_samples)))

    # Check that it does not run with different number of subjects in each
    # input
    with pytest.raises(ValueError):
        model.fit(X, y, Z2)
    print("Test: different number of subjects in the inputs")

    y.append(np.repeat(
        np.arange(n_labels), (samples - align_samples)/n_labels))
    with pytest.raises(ValueError):
        model.fit(X, y, Z)
    print("Test: different number of subjects in the inputs")
