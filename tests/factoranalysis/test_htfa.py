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


def test_R():
    from brainiak.factoranalysis.htfa import HTFA
    with pytest.raises(TypeError) as excinfo:
        HTFA()
    assert "missing 2 required positional arguments" in str(excinfo.value)


def test_X():
    from brainiak.factoranalysis.htfa import HTFA
    import numpy as np

    n_voxel = 100
    n_tr = 20
    K = 5
    max_global_iter = 3
    max_local_iter = 3
    max_voxel = n_voxel
    max_tr = n_tr

    R = []
    n_subj = 2
    for s in np.arange(n_subj):
        R.append(np.random.randint(2, high=102, size=(n_voxel, 3)))

    htfa = HTFA(
        K,
        n_subj=n_subj,
        max_global_iter=max_global_iter,
        max_local_iter=max_local_iter,
        max_voxel=max_voxel,
        max_tr=max_tr)

    X = np.random.rand(n_voxel, n_tr)
    # Check that does NOT run with wrong data type
    with pytest.raises(TypeError) as excinfo:
        htfa.fit(X, R=R)
    assert "Input data should be a list" in str(excinfo.value)

    X = []
    # Check that does NOT run with wrong array dimension
    with pytest.raises(ValueError) as excinfo:
        htfa.fit(X, R=R)
    assert "Need at leat one subject to train the model" in str(excinfo.value)

    X = []
    X.append([1, 2, 3])
    # Check that does NOT run with wrong array dimension
    with pytest.raises(TypeError) as excinfo:
        htfa.fit(X, R=R)
    assert "data should be an array" in str(excinfo.value)

    X = []
    X.append(np.random.rand(n_voxel))
    # Check that does NOT run with wrong array dimension
    with pytest.raises(TypeError) as excinfo:
        htfa.fit(X, R=R)
    assert "subject data should be 2D array" in str(excinfo.value)

    X = []
    for s in np.arange(n_subj):
        X.append(np.random.rand(n_voxel, n_tr))
    R = np.random.randint(2, high=102, size=(n_voxel, 3))

    # Check that does NOT run with wrong data type
    with pytest.raises(TypeError) as excinfo:
        htfa.fit(X, R=R)
    assert "Coordinates should be a list" in str(excinfo.value)

    R = []
    R.append([1, 2, 3])
    # Check that does NOT run with wrong data type
    with pytest.raises(TypeError) as excinfo:
        htfa.fit(X, R=R)
    assert ("Each scanner coordinate matrix should be an array"
            in str(excinfo.value))

    R = []
    R.append(np.random.rand(n_voxel))
    # Check that does NOT run with wrong array dimension
    with pytest.raises(TypeError) as excinfo:
        htfa.fit(X, R=R)
    assert ("Each scanner coordinate matrix should be 2D array"
            in str(excinfo.value))

    R = []
    for s in np.arange(n_subj):
        R.append(np.random.rand(n_voxel - 1, 3))
    # Check that does NOT run with wrong array dimension
    with pytest.raises(TypeError) as excinfo:
        htfa.fit(X, R=R)
    assert ("n_voxel should be the same in X[idx] and R[idx]"
            in str(excinfo.value))


def test_can_run():
    import numpy as np
    from brainiak.factoranalysis.htfa import HTFA
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_voxel = 100
    n_tr = 20
    K = 5
    max_global_iter = 3
    max_local_iter = 3
    max_voxel = n_voxel
    max_tr = n_tr
    R = []
    n_subj = 2
    for s in np.arange(n_subj):
        R.append(np.random.randint(2, high=102, size=(n_voxel, 3)))
    my_R = []
    for idx in np.arange(n_subj):
        if idx % size == rank:
            my_R.append(R[idx])

    htfa = HTFA(
        K,
        n_subj=n_subj,
        max_global_iter=max_global_iter,
        max_local_iter=max_local_iter,
        max_voxel=max_voxel,
        max_tr=max_tr,
        verbose=True)
    assert htfa, "Invalid HTFA instance!"

    X = []
    for s in np.arange(n_subj):
        X.append(np.random.rand(n_voxel, n_tr))
    my_data = []
    for idx in np.arange(n_subj):
        if idx % size == rank:
            my_data.append(X[idx])

    if rank == 0:
        htfa.fit(my_data, R=my_R)
        assert True, "Root successfully running HTFA"
        assert htfa.global_prior_.shape[0] == htfa.prior_bcast_size,\
            "Invalid result of HTFA! (wrong # element in global_prior)"
        assert htfa.global_posterior_.shape[0] == htfa.prior_bcast_size,\
            "Invalid result of HTFA! (wrong # element in global_posterior)"

    else:
        htfa.fit(my_data, R=my_R)
        assert True, "worker successfully running HTFA"
        print(htfa.local_weights_.shape)
        assert htfa.local_weights_.shape[0] == n_tr * K,\
            "Invalid result of HTFA! (wrong # element in local_weights)"
        assert htfa.local_posterior_.shape[0] == htfa.prior_size,\
            "Invalid result of HTFA! (wrong # element in local_posterior)"
