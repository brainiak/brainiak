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
from sklearn.datasets import make_spd_matrix


def test_distributed_mdms():  # noqa: C901
    import brainiak.funcalign.mdms
    s = brainiak.funcalign.mdms.MDMS()
    assert s, "Invalid MDMS instance!"

    import numpy as np
    np.random.seed(0)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nrank = comm.Get_size()

    # set parameters
    subj_ds_list = {'D1': ['Adam', 'Bob', 'Carol'], 'D2': ['Tom', 'Bob']}
    voxels = {'Adam': 80, 'Bob': 100, 'Carol': 120, 'Tom': 90}
    samples = {'D1': 100, 'D2': 50}
    features = 3
    noise_level = 0.1

    s = brainiak.funcalign.mdms.MDMS(n_iter=5, features=features, comm=comm)
    assert s, "Invalid MDMS instance!"

    # generate data on rank 0
    if rank == 0:
        # generate S
        S = {}
        for ds in samples:
            mean = np.zeros((samples[ds],))
            cov = make_spd_matrix(samples[ds])
            S[ds] = np.random.multivariate_normal(mean, cov, size=features)
        # generate W
        W = {}
        for subj in voxels:
            rnd_matrix = np.random.rand(voxels[subj], features)
            W[subj], _ = np.linalg.qr(rnd_matrix)
        # compute X with noise
        X = {}
        for ds in samples:
            X[ds] = {}
            for subj in subj_ds_list[ds]:
                noise = np.random.normal(loc=0, scale=noise_level *
                                         abs(np.random.randn()),
                                         size=(voxels[subj], samples[ds]))
                X[ds][subj] = W[subj].dot(S[ds]) + noise
        # compute data structure
        ds_struct = brainiak.funcalign.mdms.Dataset()
        ds_struct.build_from_data(X)
        assert ds_struct, "Invalid Dataset instance!"

    else:
        X = {}
        for ds in samples:
            X[ds] = {}
            for subj in subj_ds_list[ds]:
                X[ds][subj] = None
        ds_struct = None

    # MDMS: broadcast ds_struct
    ds_struct = comm.bcast(ds_struct)

    # Check that transform does NOT run before fitting the model
    with pytest.raises(NotFittedError):
        s.transform([X['D1']['Adam']], ['Adam'])
    if rank == 0:
        print("Test: transforming before fitting the model")

    # Check that it does NOT run with wrong X structure
    with pytest.raises(Exception):
        s.fit({'D1': X['D1'], 'D2': [X['D2']['Bob']]}, ds_struct)
    if rank == 0:
        print("Test: running MDMS with wrong X data structure")

    # random distribution of data, otherwise None
    if rank == 0:
        data_mem = {}
        tag = 0  # tag start from 0
        for ds in X:
            data_mem[ds] = {}
            for subj in X[ds]:
                data_mem[ds][subj] = [np.random.randint(low=0, high=nrank),
                                      tag]
                tag += 1
    else:
        data_mem = None
    data_mem = comm.bcast(data_mem)
    if rank == 0:
        X_new = {}
        for ds in X:
            X_new[ds] = {}
            for subj in X[ds]:
                mem, tag = data_mem[ds][subj]
                if mem != 0:
                    X_new[ds][subj] = None
                    comm.send(X[ds][subj], dest=mem, tag=tag)
                else:
                    X_new[ds][subj] = X[ds][subj]
        X = X_new
    else:
        for ds in X:
            for subj in X[ds]:
                mem, tag = data_mem[ds][subj]
            if mem == rank:
                X[ds][subj] = comm.recv(source=0, tag=tag)

    # Check that runs with 4 subject
    s.fit(X, ds_struct)
    assert len(s.s_) == len(samples), (
            "Invalid computation of MDMS! (wrong # datasets in S)")

    assert len(s.w_) == len(voxels), (
        "Invalid computation of MDMS! (wrong # subjects in W)")

    # Check W
    for subj in voxels:
        assert s.w_[subj].shape[0] == voxels[subj], (
            "Invalid computation of MDMS! (wrong # voxels in W)")
        assert s.w_[subj].shape[1] == features, (
            "Invalid computation of MDMS! (wrong # features in W)")
        ortho = np.linalg.norm(s.w_[subj].T.dot(s.w_[subj])
                               - np.eye(s.w_[subj].shape[1]),
                               'fro')
        assert ortho < 1e-7, "A Wi mapping is not orthonormal in MDMS."

    # Check S
    for ds in samples:
        assert s.s_[ds].shape[0] == features, (
            "Invalid computation of MDMS! (wrong # features in S)")
        assert s.s_[ds].shape[1] == samples[ds], (
            "Invalid computation of MDMS! (wrong # samples in S)")

    # Check X reconstruction
    for ds in X:
        for subj in X[ds]:
            if X[ds][subj] is not None:
                difference = np.linalg.norm(X[ds][subj] -
                                            s.w_[subj].dot(s.s_[ds]),
                                            'fro')
                datanorm = np.linalg.norm(X[ds][subj], 'fro')
                assert difference/datanorm < 2.0, (
                    "Model seems incorrectly computed.")

    # Check that it does run to compute the shared response of each
    # dataset after the model computation
    for ds in samples:
        data, subjects = [], []
        for subj in X[ds]:
            data.append(X[ds][subj])
            subjects.append(subj)
        new_s = s.transform(data, subjects)

        assert len(new_s) == len(data), (
            "Invalid computation of MDMS! (wrong #"
            " subjects after transform)")
        for subj in range(len(new_s)):
            if new_s[subj] is not None:
                assert new_s[subj].shape[0] == features, (
                    "Invalid computation of MDMS! (wrong # features after "
                    "transform)")
                assert new_s[subj].shape[1] == samples[ds], (
                    "Invalid computation of MDMS! (wrong # samples after "
                    "transform)")

    # Check that it does NOT run with non-matching number of subjects
    with pytest.raises(ValueError):
        s.transform(data, subjects+['new'])
    if rank == 0:
        print("Test: transforming with non-matching number of subjects")

    # Check that it does not run with different number of voxels for the same
    # subjects across datasets
    # Only subject 'Bob' is in two datasets, so we change his data
    if X['D1']['Bob'] is not None:
        tmp = X['D1']['Bob']
        X['D1']['Bob'] = X['D1']['Bob'][: -2, :]
    else:
        tmp = None
    with pytest.raises(ValueError):
        s.fit(X, ds_struct)
    if rank == 0:
        print("Test: different number of voxels for the same subject")

    # Check that it does not run with different number of samples (TRs)
    # within the same dataset
    X['D1']['Bob'] = tmp  # put back the data
    if X['D2']['Tom'] is not None:
        X['D2']['Tom'] = X['D2']['Tom'][:, : -2]
    with pytest.raises(ValueError):
        s.fit(X, ds_struct)
    if rank == 0:
        print("Test: different number of samples within dataset")


test_distributed_mdms()
