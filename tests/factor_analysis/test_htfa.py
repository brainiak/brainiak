def test_R():
    from brainiak.factor_analysis.htfa import HTFA

    try:
        htfa = HTFA()
    except TypeError:
        print("Catched exception #1: three input arguments needed!")

    import numpy as np

    n_voxel = 100
    n_tr = 20
    K = 5
    max_outer_iter = 3
    max_inner_iter = 3
    max_voxel = n_voxel
    max_tr = n_tr
    R = np.random.randint(2, high=102, size=(n_voxel, 3))
    try:
        htfa = HTFA(
            R,
            K,
            max_outer_iter=max_outer_iter,
            max_inner_iter=max_inner_iter,
            max_voxel=max_voxel,
            max_tr=max_tr)
        htfa.fit()
    except TypeError:
        print("Catched exception #2: R should be a list")


def test_X():
    from brainiak.factor_analysis.htfa import HTFA

    try:
        htfa = HTFA()
    except TypeError:
        print("Catched exception #1: three input arguments needed!")

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
        R,
        K,
        max_outer_iter=max_outer_iter,
        max_inner_iter=max_inner_iter,
        max_voxel=max_voxel,
        max_tr=max_tr)

    X = np.random.rand(n_voxel, n_tr)

    # Check that does NOT run with wrong data type
    try:
        htfa.fit(X)
        assert True, "Success running HTFA with one subject!"
    except TypeError:
        print("Catched exception #3: Input data should be a list")

    X = []
    X.append([1, 2, 3])
    # Check that does NOT run with wrong array dimension
    try:
        htfa.fit(X)
    except TypeError:
        print("Catched exception #4: Input data should be an array")

    X = []
    X.append(np.random.rand(n_voxel))
    # Check that does NOT run with wrong array dimension
    try:
        htfa.fit(X)
    except ValueError:
        print("Catched exception #5: Input data should be 2D array")


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
        my_R,
        K,
        n_subj=n_subj,
        max_outer_iter=max_outer_iter,
        max_inner_iter=max_inner_iter,
        max_voxel=max_voxel,
        max_tr=max_tr,
        verbose=True,
        output_path='/tmp/',
        output_prefix='htfa')
    assert htfa, "Invalid HTFA instance!"

    X = []
    for s in np.arange(n_subj):
        X.append(np.random.rand(n_voxel, n_tr))
    my_data = []
    for idx in np.arange(n_subj):
        if idx % size == rank:
            my_data.append(X[idx])

    if rank == 0:
        global_prior, global_posterior, _, _, _ = htfa.fit(my_data)
        assert True, "Root successfully running HTFA"
        assert global_prior.shape[
            0] == htfa.prior_bcast_size,\
            "Invalid result of HTFA! (wrong # element in global_prior)"
        assert global_posterior.shape[
            0] == htfa.prior_size,\
            "Invalid result of HTFA! (wrong # element in global_posterior)"

    else:
        local_weights, local_posterior = htfa.fit(my_data)
        assert True, "worker successfully running HTFA"
        print(local_weights.shape)
        assert local_weights.shape[
            0] == n_tr * K,\
            "Invalid result of HTFA! (wrong # element in local_weights)"
        assert local_posterior.shape[
            0] == htfa.prior_size,\
            "Invalid result of HTFA! (wrong # element in local_posterior)"
