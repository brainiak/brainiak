def test_miss_argument():
    from brainiak.factor_analysis.tfa import TFA
    try:
        tfa = TFA()
    except TypeError:
        print("Catched exception #1: three input arguments needed!")


def test_two_step():
    from brainiak.factor_analysis.tfa import TFA
    import numpy as np

    n_voxel = 100
    n_tr = 20
    K = 5 
    max_iter = 5
    max_num_voxel = n_voxel
    max_num_tr = n_tr
    sample_scaling = 0.5
    R = np.random.randint(2, high=102, size=(n_voxel, 3))
    tfa = TFA(
        R,
        sample_scaling,
        K=K,
        verbose=True,
        max_iter=max_iter,
        max_num_voxel=max_num_voxel,
        max_num_tr=max_num_tr)
    tfa.init_prior()
    assert tfa, "Invalid TFA instance!"

    X = [1, 2, 3]

    # Check that does NOT run with wrong data type
    try:
        tfa.fit(X)
        assert True, "Success running TFA with one subject!"
    except TypeError:
        print("Catched exception #2: Input data should be an array")

    X = np.random.rand(n_voxel)

    # Check that does NOT run with wrong array dimension
    try:
        tfa.fit(X)
    except ValueError:
        print("Catched exception #3: Input data should be 2D array")

    K = 5
    tfa.set_K(K)
    tfa.set_R(R)
    prior = tfa.local_prior
    tfa.set_prior(prior)
    seed = 500
    tfa.set_seed(seed)
    X = np.random.rand(n_voxel, n_tr)
    tfa.fit(X)
    assert True, "Success running TFA with one subject!"
    posterior_size = K * (tfa.n_dim + 1)
    assert tfa.local_posterior.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"

    tfa.get_global_prior(R)
    global_prior = tfa.global_prior
    tfa.fit(X, global_prior)
    assert True, "Success running TFA with one subject and global prior!"
    assert tfa.local_posterior.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"

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
    R = np.random.randint(2, high=102, size=(n_voxel, 3))

    tfa = TFA(
        R,
        sample_scaling,
        K=K,
        max_iter=max_iter,
        two_step=False,
        verbose=True,
        max_num_voxel=max_num_voxel,
        max_num_tr=max_num_tr)
    tfa.init_prior()
    assert tfa, "Invalid TFA instance!"

    X = np.random.rand(n_voxel, n_tr)
    tfa.fit(X)
    assert True, "Success running TFA with one subject!"
    posterior_size = K * (tfa.n_dim + 1)
    assert tfa.local_posterior.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"

    tfa.get_global_prior(R)
    global_prior = tfa.global_prior
    tfa.fit(X, global_prior)
    assert True, "Success running TFA with one subject and global prior!"
    assert tfa.local_posterior.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"


