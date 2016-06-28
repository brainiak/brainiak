def test_R():
    from brainiak.factor_analysis.htfa import HTFA
    try:
        htfa = HTFA()
    except TypeError:
        print("Catched exception #1: two input arguments needed!")

def test_X():
    from brainiak.factor_analysis.htfa import HTFA
    import numpy as np

    n_voxel = 100
    n_tr = 20
    K = 5
    max_outer_iter = 3
    max_inner_iter = 3
    max_voxel = n_voxel
    max_tr = n_tr

    R = []
    n_subj = 2
    for s in np.arange(n_subj):
        R.append(np.random.randint(2, high=102, size=(n_voxel, 3)))

    htfa = HTFA(
        K,
        max_outer_iter=max_outer_iter,
        max_inner_iter=max_inner_iter,
        max_voxel=max_voxel,
        max_tr=max_tr)

    X = np.random.rand(n_voxel, n_tr)
    # Check that does NOT run with wrong data type
    try:
        htfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #3: Input data should be a list")

    X = []
    # Check that does NOT run with wrong array dimension
    try:
        htfa.fit(X, R=R)
    except ValueError:
        print("Catched exception #4: should at least have one subject")

    X = []
    X.append([1, 2, 3])
    # Check that does NOT run with wrong array dimension
    try:
        htfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #5: subject data should be an array")

    X = []
    X.append(np.random.rand(n_voxel))
    # Check that does NOT run with wrong array dimension
    try:
        htfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #6: subject data should be 2D array")

    X = []
    for s in np.arange(n_subj):
        X.append(np.random.rand(n_voxel, n_tr))
    R = np.random.randint(2, high=102, size=(n_voxel, 3))
    
    # Check that does NOT run with wrong data type
    try:
        htfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #7: subject coordinate should be a list")

    R = []
    R.append([1, 2, 3])
    # Check that does NOT run with wrong data type
    try:
        htfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #8: subject coordinate should be an array")

    R = []
    R.append(np.random.rand(n_voxel))
    # Check that does NOT run with wrong array dimension
    try:
        htfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #9: subject coordinate should be 2D array")

    R = []
    for s in np.arange(n_subj):
        R.append(np.random.rand(n_voxel-1, 3))
    # Check that does NOT run with wrong array dimension
    try:
        htfa.fit(X, R=R)
    except TypeError:
        print("Catched exception #10: n_voxel should match in data and coordinate") 

def test_can_run():
    import numpy as np
    from brainiak.factor_analysis.htfa import HTFA
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_voxel = 100
    n_tr = 20
    K = 5
    max_outer_iter = 3
    max_inner_iter = 3
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
        max_outer_iter=max_outer_iter,
        max_inner_iter=max_inner_iter,
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
        assert htfa.global_prior_.shape[
            0] == htfa.prior_bcast_size,\
            "Invalid result of HTFA! (wrong # element in global_prior)"
        assert htfa.global_posterior_.shape[
            0] == htfa.prior_bcast_size,\
            "Invalid result of HTFA! (wrong # element in global_posterior)"

    else:
        htfa.fit(my_data, R=my_R)
        assert True, "worker successfully running HTFA"
        print(htfa.local_weights.shape)
        assert htfa.local_weights.shape[
            0] == n_tr * K,\
            "Invalid result of HTFA! (wrong # element in local_weights)"
        assert htfa.local_posterior.shape[
            0] == htfa.prior_size,\
            "Invalid result of HTFA! (wrong # element in local_posterior)"
