import pytest

def test_one_step():
    from brainiak.factor_analysis.tfa import TFA
    import numpy as np

    n_voxel = 100
    n_tr = 20
    K = 5
    max_iter = 5
    max_num_voxel = n_voxel
    max_num_tr = n_tr
    sample_scaling = 0.5
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

    global_prior, _, _ = tfa.get_global_prior(R)
    tfa.set_K(K)
    tfa.set_seed(200)
    tfa.fit(X, R=R, global_prior=global_prior)
    assert True, "Success running TFA with one subject and global prior!"
    assert tfa.local_posterior_.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"
