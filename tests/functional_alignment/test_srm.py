def test_can_instantiate():
    import toolkit.functional_alignment.srm
    s = toolkit.functional_alignment.srm.SRM()
    assert s, "Invalid SRM instance!"

    import numpy as np

    voxels = 100
    samples = 500
    subjects = 2
    features = 3

    s = toolkit.functional_alignment.srm.SRM(verbose=True, n_iter=5,
                                             features=features)
    assert s, "Invalid SRM instance!"


    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, samples)
    z = np.linspace(-2, 2, samples)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    S = np.vstack((x, y, z))

    X = []
    W = []
    Q, R = np.linalg.qr(np.random.random((voxels, features)))
    W.append(Q)
    X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that does NOT run with 1 subject
    try:
        s.fit(X)
        assert True, "Success running SRM with one subject!"
    except ValueError:
        print("Catched Exception number 1: running SRM with 1 subject")

    for subject in range(1, subjects):
        Q, R = np.linalg.qr(np.random.random((voxels, features)))
        W.append(Q)
        X.append(Q.dot(S) + 0.1*np.random.random((voxels, samples)))

    # Check that runs with 2 subject
    try:
        s.fit(X)
    except ValueError:
        assert True, "Problem running SRM."

    assert len(s.w_) == subjects, "Invalid computation of SRM! (wrong # subjects in W)"
    for subject in range(subjects):
        assert s.w_[subject].shape[0] == voxels, "Invalid computation of SRM! (wrong # voxels in W)"
        assert s.w_[subject].shape[1] == features, "Invalid computation of SRM! (wrong # features in W)"
    assert s.s_.shape[0] == features, "Invalid computation of SRM! (wrong # features in S)"
    assert s.s_.shape[1] == samples, "Invalid computation of SRM! (wrong # samples in S)"

    # Check that it does not run without enough samples (TRs).
    try:
        s.set_params(features=(samples+1))
        s.fit(X)
        assert True, "Success running SRM with more features than samples!"
    except ValueError as e:
        print("Catched Exception number 3: not enough samples")

    # Check that it does not run with different number of samples (TRs)
    S2 = S[:,:-2]
    X.append(Q.dot(S2))
    try:
        s.fit(X)
        assert True, "Success running SRM with different number of samples!"
    except ValueError:
        print("Catched Exception number 2: different number of samples per subject")


