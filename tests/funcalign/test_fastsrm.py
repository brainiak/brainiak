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
import numpy as np
import hashlib
import tempfile
from brainiak.funcalign.fastsrm import FastSRM, reduce_data, _reduced_space_compute_shared_response
from brainiak.funcalign.fastsrm import _compute_basis_subject_online, fast_srm
import os
from numpy.testing import assert_array_almost_equal


def to_path(X, dirpath):
    """
    Save list of list of array to path and returns the path_like array
    Parameters
    ----------
    X: list of list of array
        input data
    dirpath: str
        dirpath
    Returns
    -------
    paths: array of str
        path arrays where all data are stored
    """
    paths = []
    for i, sessions in enumerate(X):
        sessions_path = []
        for j, session in enumerate(sessions):
            pth = "%i_%i" % (i, j)
            np.save(os.path.join(dirpath, pth), session)
            sessions_path.append(os.path.join(dirpath, pth + ".npy"))
        paths.append(sessions_path)
    return np.array(paths)


def generate_data(n_voxels, n_timeframes, n_subjects, n_components, datadir, noise_level=0.1):
    n_sessions = len(n_timeframes)
    cumsum_timeframes = np.cumsum([0] + n_timeframes)
    slices_timeframes = [slice(cumsum_timeframes[i], cumsum_timeframes[i + 1]) for i in range(n_sessions)]

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, int(np.sum(n_timeframes)))
    z = np.linspace(-2, 2, int(np.sum(n_timeframes)))
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))

    # Generate fake data
    W = []
    X = []
    for subject in range(n_subjects):
        Q, R = np.linalg.qr(np.random.random((n_voxels, n_components)))
        W.append(Q.T)
        X_ = []
        for session in range(n_sessions):
            S_s = S[:, slices_timeframes[session]]
            S_s = S_s - np.mean(S_s, axis=1, keepdims=True)
            noise = noise_level * np.random.random((n_voxels, n_timeframes[session]))
            noise = noise - np.mean(noise, axis=1, keepdims=True)
            data = Q.dot(S_s) \
                   + noise
            X_.append(data.T)
        X.append(X_)

    # create paths such that paths[i, j] contains data of subject i during session j
    paths = to_path(X, datadir)
    S = [(S[:, s] - np.mean(S[:, s], axis=1, keepdims=True)).T for s in slices_timeframes]
    return paths, W, S


def test_reduced_data():
    with tempfile.TemporaryDirectory() as datadir:

        # We authorize different timeframes for different sessions but they should be the same across subject
        n_voxels = 100
        n_timeframes = [250, 245]
        n_subjects = 2
        n_components = 3  # number of components used for SRM model
        n_supervoxels = 10  # number of components of the atlas
        n_jobs = 1
        n_sessions = len(n_timeframes)

        np.random.seed(0)
        paths, _, _ = generate_data(n_voxels, n_timeframes, n_subjects, n_components, datadir)

        # Test if generated data has the good shape
        for subject in range(n_subjects):
            for session in range(n_sessions):
                assert np.load(paths[subject, session]).shape == (n_timeframes[session], n_voxels)

        # Test if reduced data has the good shape

        # probabilistic atlas
        atlas = np.random.rand(n_supervoxels, n_voxels)
        data = reduce_data(paths, atlas=atlas, n_jobs=n_jobs, low_ram=False, temp_dir=None)

        for subject in range(n_subjects):
            for session in range(n_sessions):
                assert data[subject, session].shape == (n_timeframes[session], n_supervoxels)

        # deterministic atlas
        det_atlas = np.round(np.random.rand(n_voxels) * n_supervoxels)
        n_unique = len(np.unique(det_atlas)[1:])
        while n_unique != n_supervoxels:
            det_atlas = np.round(np.random.rand(n_voxels) * n_supervoxels)
            n_unique = len(np.unique(det_atlas)[1:])

        data = reduce_data(paths, atlas=det_atlas, n_jobs=n_jobs, low_ram=True, temp_dir=datadir)
        for subject in range(n_subjects):
            for session in range(n_sessions):
                assert np.load(data[subject, session]).shape == (n_timeframes[session], n_supervoxels)

        # test atlas that reduces nothing
        atlas = np.arange(1, n_voxels + 1)
        data = reduce_data(paths, atlas=atlas, n_jobs=n_jobs, low_ram=False)
        for i in range(n_subjects):
            for j in range(n_sessions):
                assert_array_almost_equal(data[i, j], np.load(paths[i, j]))

        # test atlas that reduces everything
        atlas = np.ones(n_voxels)
        data = reduce_data(paths, atlas=atlas, n_jobs=n_jobs, low_ram=False)
        for i in range(n_subjects):
            for j in range(n_sessions):
                assert_array_almost_equal(data[i, j].flatten(), np.mean(np.load(paths[i, j]), axis=1))


def test_reduced_data_srm():
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)

        # We authorize different timeframes for different sessions but they should be the same across subject
        n_voxels = 100
        n_timeframes = [250, 245]
        n_subjects = 5
        n_components = 3  # number of components used for SRM model
        n_jobs = 1
        n_sessions = len(n_timeframes)

        np.random.seed(0)
        paths, W, S = generate_data(n_voxels, n_timeframes, n_subjects, n_components, datadir, 0)

        # Test if generated data has the good shape
        for subject in range(n_subjects):
            for session in range(n_sessions):
                assert np.load(paths[subject, session]).shape == (n_timeframes[session], n_voxels)

        # Test if generated basis have good shape
        assert len(W) == n_subjects
        for w in W:
            assert w.shape == (n_components, n_voxels)

        assert len(S) == n_sessions
        for j, s in enumerate(S):
            assert s.shape == (n_timeframes[j], n_components)

        atlas = np.arange(1, n_voxels + 1)
        data = reduce_data(paths, atlas=atlas, n_jobs=n_jobs, low_ram=False, temp_dir=None)

        # Test if shared response has the good shape
        shared_response_list = _reduced_space_compute_shared_response(data,
                                                                      reduced_basis_list=None,
                                                                      n_components=n_components)
        assert len(shared_response_list) == n_sessions

        for session in range(n_sessions):
            assert shared_response_list[session].shape == (n_timeframes[session], n_components)

        # Test basis from shared response
        for i, sessions in enumerate(paths):
            basis = _compute_basis_subject_online(sessions, S)
            # test shape
            assert basis.shape == (n_components, n_voxels)
            # test orthogonality
            assert np.allclose(basis.dot(basis.T), np.eye(n_components))
            # test correctness
            assert_array_almost_equal(basis, W[i], 2)

        # Test reduced_data_shared_response
        shared_response_list = _reduced_space_compute_shared_response(data,
                                                                      reduced_basis_list=W,
                                                                      n_components=n_components
                                                                      )
        for session in range(n_sessions):
            S_real = np.mean([data[i, session].dot(W[i].T) for i in range(n_subjects)], axis=0)
            assert_array_almost_equal(shared_response_list[session], S_real)
            assert_array_almost_equal(shared_response_list[session], S[session])

        # Test fast_srm for reduced_data
        shared_response_list = fast_srm(data, n_components=n_components)

        for i, sessions in enumerate(paths):
            basis = _compute_basis_subject_online(sessions, shared_response_list)
            for j, session in enumerate(sessions):
                assert_array_almost_equal(shared_response_list[j].dot(basis), np.load(paths[i, j]))


def test_fastsrm_class():
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)

        # We authorize different timeframes for different sessions but they should be the same across subject
        n_voxels = 100
        n_timeframes = [250, 245]
        n_subjects = 5
        n_components = 3  # number of components used for SRM model
        n_jobs = 1
        n_sessions = len(n_timeframes)

        np.random.seed(0)
        paths, W, S = generate_data(n_voxels, n_timeframes, n_subjects, n_components, datadir, 0)

        atlas = np.arange(1, n_voxels + 1)
        srm = FastSRM(atlas=atlas,
                      n_components=n_components,
                      n_iter=10,
                      temp_dir=datadir,
                      low_ram=True,
                      verbose=True)

        srm.fit(paths)
        basis = srm.basis_list
        shared_response = srm.transform(paths)

        for i in range(n_subjects):
            for j in range(n_sessions):
                basis_i = np.load(basis[i])
                assert_array_almost_equal(shared_response[j].dot(basis_i), np.load(paths[i, j]))
















        # for subject in range(n_subjects):
        #     for session in range(n_sessions):
        #         assert data[subject, session].shape == (n_timeframes, n_supervoxels)
        #
        # # Try with deterministic atlas
        # det_atlas = np.round(np.random.rand(n_voxels) * n_supervoxels)
        # n_unique = len(np.unique(det_atlas)[1:])
        # while n_unique != n_supervoxels:
        #     det_atlas = np.round(np.random.rand(n_voxels) * n_supervoxels)
        #     n_unique = len(np.unique(det_atlas)[1:])
        #
        # data = reduce_data(paths, atlas=det_atlas, n_jobs=n_jobs, low_ram=True, temp_dir=datadir)
        # for subject in range(n_subjects):
        #     for session in range(n_sessions):
        #         assert np.load(data[subject, session]).shape == (n_timeframes, n_supervoxels)




    # def test_can_instantiate():
    #     s = FastSRM(atlas=np.random.rand(10, 10))
    #     assert s, "Invalid SRM instance!"
    #
    #
    #
    #     s = FastSRM(atlas=np.ones(n_voxels), n_iter=5, n_components=n_components)
    #     assert s, "Invalid SRM instance!"
    #
    #     # Create a Shared response S with K = 3
    #     theta = np.linspace(-4 * np.pi, 4 * np.pi, n_timeframes)
    #     z = np.linspace(-2, 2, n_timeframes)
    #     r = z**2 + 1
    #     x = r * np.sin(theta)
    #     y = r * np.cos(theta)
    #
    #     S = np.vstack((x, y, z))
    #
    #     X = []
    #     W = []
    #     Q, R = np.linalg.qr(np.random.random((n_voxels, n_timeframes)))
    #     W.append(Q)
    #
    #     data = Q.dot(S) + 0.1*np.random.random((n_voxels, n_timeframes))
    #     X.append([data.T])
    #
    #     paths = to_path(X, datadir)
    #
    #     # Check that transform does NOT run before fitting the model
    #     with pytest.raises(NotFittedError):
    #         s.transform(paths)
    #     print("Test: transforming before fitting the model")
    #
    #     # Check that it does NOT run with 1 subject
    #     with pytest.raises(ValueError):
    #         s.fit(paths)
    #     print("Test: running SRM with 1 subject")
    #
    #     for subject in range(1, n_subjects):
    #         Q, R = np.linalg.qr(np.random.random((n_voxels, n_components)))
    #         W.append(Q)
    #         data = Q.dot(S) + 0.1*np.random.random((n_voxels, samples))
    #         X.append([data.T])
    #
    #     # Check that runs with 2 subject
    #     paths = to_path(X, datadir)
    #     s.fit(paths)
    #     shared_response = s.transform(paths)[0]
    #     from pathlib import Path
    #     sr_v0_4 = np.load(Path(__file__).parent / "sr_v0_4.npz")['sr']
    #     assert(np.allclose(sr_v0_4, shared_response))
    #
    #     assert len(s.basis_list) == n_subjects, (
    #         "Invalid computation of SRM! (wrong # subjects in W)")
    #     for subject in range(n_subjects):
    #         assert s.basis_list[subject].shape[0] == n_voxels, (
    #             "Invalid computation of SRM! (wrong # voxels in W)")
    #         assert s.basis_list[subject].shape[1] == n_components, (
    #             "Invalid computation of SRM! (wrong # n_components in W)")
    #         ortho = np.linalg.norm(s.basis_list[subject].T.dot(s.basis_list[subject])
    #                                - np.eye(s.basis_list[subject].shape[1]),
    #                                'fro')
    #         assert ortho < 1e-7, "A Wi mapping is not orthonormal in SRM."
    #         difference = np.linalg.norm(X[subject] - s.basis_list[subject].dot(shared_response),
    #                                     'fro')
    #         datanorm = np.linalg.norm(X[subject], 'fro')
    #         assert difference/datanorm < 1.0, "Model seems incorrectly computed."
    #     assert shared_response.shape[0] == n_components, (
    #         "Invalid computation of SRM! (wrong # n_components in S)")
    #     assert shared_response.shape[1] == samples, (
    #         "Invalid computation of SRM! (wrong # samples in S)")
    #
    #     # Check that it does run to compute the shared response after the model
    #     # computation
    #     new_s = s.transform(paths)[0]
    #
    #     assert len(new_s) == n_subjects, (
    #         "Invalid computation of SRM! (wrong # subjects after transform)")
    #     for subject in range(n_subjects):
    #         assert new_s[subject].shape[0] == n_components, (
    #             "Invalid computation of SRM! (wrong # n_components after transform)")
    #         assert new_s[subject].shape[1] == samples, (
    #             "Invalid computation of SRM! (wrong # samples after transform)")
    #
    #     # Check that it does not run without enough n_timeframes (TRs).
    #     with pytest.raises(ValueError):
    #         s.set_params(n_components=(n_timeframes+1))
    #         s.fit(paths)
    #     print("Test: not enough n_timeframes")
    #
    #     # Check that it does not run with different number of n_timeframes (TRs)
    #     S2 = S[:, :-2]
    #     data = Q.dot(S2)
    #     X.append([data.T])
    #     paths = to_path(X, datadir)
    #     with pytest.raises(ValueError):
    #         s.fit(paths)
    #     print("Test: different number of n_timeframes per subject")
