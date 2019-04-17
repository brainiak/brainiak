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


def test_tfa():
    from brainiak.factoranalysis.tfa import TFA
    import numpy as np

    n_voxel = 100
    n_tr = 20
    K = 5
    max_iter = 5
    max_num_voxel = n_voxel
    max_num_tr = n_tr
    tfa = TFA(
        K=K,
        max_iter=max_iter,
        verbose=True,
        max_num_voxel=max_num_voxel,
        max_num_tr=max_num_tr)
    assert tfa, "Invalid TFA instance!"

    R = np.random.randint(2, high=102, size=(n_voxel, 3))
    X = [1, 2, 3]
    # Check that does NOT run with wrong data type
    with pytest.raises(TypeError) as excinfo:
        tfa.fit(X, R=R)
    assert "Input data should be an array" in str(excinfo.value)

    X = np.random.rand(n_voxel)
    # Check that does NOT run with wrong array dimension
    with pytest.raises(TypeError) as excinfo:
        tfa.fit(X, R=R)
    assert "Input data should be 2D array" in str(excinfo.value)

    X = np.random.rand(n_voxel, n_tr)
    R = [1, 2, 3]
    # Check that does NOT run with wrong data type
    with pytest.raises(TypeError) as excinfo:
        tfa.fit(X, R=R)
    assert "coordinate matrix should be an array" in str(excinfo.value)

    R = np.random.rand(n_voxel)
    # Check that does NOT run with wrong array dimension
    with pytest.raises(TypeError) as excinfo:
        tfa.fit(X, R=R)
    assert "coordinate matrix should be 2D array" in str(excinfo.value)

    R = np.random.randint(2, high=102, size=(n_voxel - 1, 3))
    # Check that does NOT run if n_voxel in X and R does not match
    with pytest.raises(TypeError) as excinfo:
        tfa.fit(X, R=R)
    assert "The number of voxels should be the same in X and R" in str(
        excinfo.value)

    R = np.random.randint(2, high=102, size=(n_voxel, 3))
    tfa.fit(X, R=R)
    assert True, "Success running TFA with one subject!"
    posterior_size = K * (tfa.n_dim + 1)
    assert tfa.local_posterior_.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"

    weight_method = 'ols'
    tfa = TFA(
        weight_method=weight_method,
        K=K,
        max_iter=max_iter,
        verbose=True,
        max_num_voxel=max_num_voxel,
        max_num_tr=max_num_tr)
    assert tfa, "Invalid TFA instance!"

    X = np.random.rand(n_voxel, n_tr)
    tfa.fit(X, R=R)
    assert True, "Success running TFA with one subject!"

    template_prior, _, _ = tfa.get_template(R)
    tfa.set_K(K)
    tfa.set_seed(200)
    tfa.fit(X, R=R, template_prior=template_prior)
    assert True, "Success running TFA with one subject and template prior!"
    assert tfa.local_posterior_.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"

    weight_method = 'odd'
    tfa = TFA(
        weight_method=weight_method,
        K=K,
        max_iter=max_iter,
        verbose=True,
        max_num_voxel=max_num_voxel,
        max_num_tr=max_num_tr)
    with pytest.raises(ValueError) as excinfo:
        tfa.fit(X, R=R)
    assert "'rr' and 'ols' are accepted as weight_method!" in str(
        excinfo.value)
