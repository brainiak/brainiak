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
    try:
        tfa.fit(X, R=R)
        assert True, "Success running TFA with one subject!"
    except TypeError:
        print("Catched exception #1: Input data should be an array")

    X = np.random.rand(n_voxel)
    # Check that does NOT run with wrong array dimension
    try:
        tfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #2: Input data should be 2D array")

    X = np.random.rand(n_voxel, n_tr)
    R = [1, 2, 3]
    # Check that does NOT run with wrong data type
    try:
        tfa.fit(X, R=R)
        assert True, "Success running TFA with one subject!"
    except TypeError:
        print("Catched exception #3: Input data should be an array")

    R = np.random.rand(n_voxel)
    # Check that does NOT run with wrong array dimension
    try:
        tfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #4: Input data should be 2D array")

    R = np.random.randint(2, high=102, size=(n_voxel-1, 3))
    # Check that does NOT run if n_voxel in X and R does not match 
    try:
        tfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #5: X and R should have same n_voxel")

    R = np.random.randint(2, high=102, size=(n_voxel, 3))
    tfa.fit(X, R=R)
    assert True, "Success running TFA with one subject!"
    posterior_size = K * (tfa.n_dim + 1)
    assert tfa.local_posterior.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"
    
    weight_method='ols'
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
    

    global_prior, _ = tfa.get_global_prior(R)
    tfa.set_K(K)
    tfa.set_seed(200)
    tfa.fit(X, R=R, global_prior=global_prior)
    assert True, "Success running TFA with one subject and global prior!"
    assert tfa.local_posterior.shape[
        0] == posterior_size,\
        "Invalid result of TFA! (wrong # element in local_posterior)"


