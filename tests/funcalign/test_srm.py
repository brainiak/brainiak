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
from sklearn.utils.validation import NotFittedError


def test_can_instantiate():
    import brainiak.funcalign.srm
    s = brainiak.funcalign.srm.SRM()
    assert s, "Invalid SRM instance!"

    import numpy as np

    voxels = 100
    samples = 500
    subjects = 2
    features = 3

    s = brainiak.funcalign.srm.SRM(verbose=True, n_iter=5,
                                   features=features)
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
    try:
        s.transform(X)
        assert True, "Success transforming before fitting the model!"
    except NotFittedError:
        print("Caught Exception number 1: transforming before fitting the model")

    # Check that it does NOT run with 1 subject
    try:
        s.fit(X)
        assert True, "Success running SRM with one subject!"
    except ValueError:
        print("Caught Exception number 1: running SRM with 1 subject")

    for subject in range(1, subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that runs with 2 subject
    try:
        s.fit(X)
    except ValueError:
        assert True, "Problem running SRM."

    assert len(s.w_) == subjects, "Invalid computation of SRM! (wrong # subjects in W)"
    for subject in range(subjects):
        assert s.w_[subject].shape[0] == voxels, "Invalid computation of SRM! (wrong # voxels in W)"
        assert s.w_[subject].shape[1] == features, "Invalid computation of SRM! (wrong # features in W)"
    assert s.s_.shape[0] == features, "Invalid computation of SRM! (wrong # features in S)"
    assert s.s_.shape[1] == samples, "Invalid computation of SRM! (wrong # samples in S)"

    # Check that it does NOT run with non-matcing number of subjects
    try:
        s.transform(X[1])
        assert True, "Success transforming with non-matching number of subjects"
    except ValueError:
        print("Caught Exception number 2: transforming with non-matching number of subjects")

    # Check that it does not run without enough samples (TRs).
    try:
        s.set_params(features=(samples+1))
        s.fit(X)
        assert True, "Success running SRM with more features than samples!"
    except ValueError as e:
        print("Catched Exception number 3: not enough samples")

    # Check that it does not run with different number of samples (TRs)
    S2 = S[:,:-2]
    X.append(Q.dot(S2))
    try:
        s.fit(X)
        assert True, "Success running SRM with different number of samples!"
    except ValueError:
        print("Catched Exception number 2: different number of samples per subject")


