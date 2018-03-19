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
from mpi4py import MPI


def test_distributed_srm():  # noqa: C901
    import brainiak.funcalign.srm
    s = brainiak.funcalign.srm.SRM()
    assert s, "Invalid SRM instance!"

    import numpy as np
    np.random.seed(0)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nrank = comm.Get_size()

    voxels = 100
    samples = 500
    subjects = 2
    features = 3

    s = brainiak.funcalign.srm.SRM(n_iter=5, features=features, comm=comm)
    assert s, "Invalid SRM instance!"

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))

    # DSRM: broadcast S
    S = comm.bcast(S)

    X = []
    W = []
    # DSRM: only append on rank 0
    Q, R = np.linalg.qr(np.random.random((voxels, features)))
    tmp_noise = 0.1*np.random.random((voxels, samples))
    if rank == 0:
        W.append(Q)
        X.append(Q.dot(S) + tmp_noise)
    else:
        W.append(None)
        X.append(None)

    # Check that transform does NOT run before fitting the model
    with pytest.raises(NotFittedError):
        s.transform(X)
    if rank == 0:
        print("Test: transforming before fitting the model")

    # Check that it does NOT run with 1 subject
    with pytest.raises(ValueError):
        s.fit(X)
    if rank == 0:
        print("Test: running SRM with 1 subject")

    # DSRM: cyclic distribution of subject data, otherwise None
    for subject in range(1, subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        tmp_noise = 0.1*np.random.random((voxels, samples))
        if subject % nrank == rank:
            W.append(Q)
            X.append(Q.dot(S) + tmp_noise)
        else:
            W.append(None)
            X.append(None)

    # Check that runs with 2 subject
    s.fit(X)
    from pathlib import Path
    sr_v0_4 = np.load(Path(__file__).parent / "sr_v0_4.npz")['sr']
    assert(np.allclose(sr_v0_4, s.s_))

    assert len(s.w_) == subjects, (
        "Invalid computation of SRM! (wrong # subjects in W)")
    for subject in range(subjects):
        if s.w_[subject] is not None:
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
            assert difference/datanorm < 1.0, (
                "Model seems incorrectly computed.")

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
        if new_s[subject] is not None:
            assert new_s[subject].shape[0] == features, (
                "Invalid computation of SRM! (wrong # features after "
                "transform)")
            assert new_s[subject].shape[1] == samples, (
                "Invalid computation of SRM! (wrong # samples after "
                "transform)")

    # Check that it does NOT run with non-matching number of subjects
    with pytest.raises(ValueError):
        s.transform([X[1]])
    if rank == 0:
        print("Test: transforming with non-matching number of subjects")

    # Check that it does not run without enough samples (TRs).
    with pytest.raises(ValueError):
        s.set_params(features=(samples+1))
        s.fit(X)
    if rank == 0:
        print("Test: not enough samples")

    # Check that it does not run with different number of samples (TRs)
    if rank == 0:
        S2 = S[:, :-2]
        X.append(Q.dot(S2))
    else:
        X.append(None)

    with pytest.raises(ValueError):
        s.fit(X)
    if rank == 0:
        print("Test: different number of samples per subject")


test_distributed_srm()
