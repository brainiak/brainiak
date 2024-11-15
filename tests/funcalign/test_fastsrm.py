import os
import tempfile
import re

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.exceptions import NotFittedError

from brainiak.funcalign.fastsrm import (
    FastSRM, _compute_and_save_corr_mat, _compute_and_save_subject_basis,
    _compute_basis_subject_online, _reduced_space_compute_shared_response,
    check_atlas, check_imgs, check_shared_response, create_temp_dir, fast_srm,
    reduce_data, safe_load)

from brainiak.funcalign.srm import DetSRM


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


def generate_data(n_voxels,
                  n_timeframes,
                  n_subjects,
                  n_components,
                  datadir,
                  noise_level=0.1,
                  input_format="array"):
    n_sessions = len(n_timeframes)
    cumsum_timeframes = np.cumsum([0] + n_timeframes)
    slices_timeframes = [
        slice(cumsum_timeframes[i], cumsum_timeframes[i + 1])
        for i in range(n_sessions)
    ]

    # Create a Shared response S with K = 3
    theta = np.linspace(-4 * np.pi, 4 * np.pi, int(np.sum(n_timeframes)))
    z = np.linspace(-2, 2, int(np.sum(n_timeframes)))
    r = z**2 + 1
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
            noise = noise_level * np.random.random(
                (n_voxels, n_timeframes[session]))
            noise = noise - np.mean(noise, axis=1, keepdims=True)
            data = Q.dot(S_s) + noise
            X_.append(data)
        X.append(X_)

    # create paths such that paths[i, j] contains data
    # of subject i during session j
    S = [(S[:, s] - np.mean(S[:, s], axis=1, keepdims=True))
         for s in slices_timeframes]

    if input_format == "array":
        paths = to_path(X, datadir)
        return paths, W, S

    elif input_format == "list_of_list":
        return X, W, S

    elif input_format == "list_of_array":
        return [
            np.concatenate([X[i][j].T for j in range(n_sessions)]).T
            for i in range(n_subjects)
        ], W, S
    else:
        raise ValueError("Wrong input_format")


def test_generated_data():
    with tempfile.TemporaryDirectory() as datadir:

        # We authorize different timeframes for different sessions
        # but they should be the same across subject
        n_voxels = 10
        n_timeframes = [25, 24]
        n_subjects = 2
        n_components = 3  # number of components used for SRM model
        n_sessions = len(n_timeframes)

        np.random.seed(0)
        paths, W, S = generate_data(n_voxels, n_timeframes, n_subjects,
                                    n_components, datadir)

        # Test if generated data has the good shape
        for subject in range(n_subjects):
            for session in range(n_sessions):
                assert (np.load(
                    paths[subject, session]).shape == (n_voxels,
                                                       n_timeframes[session]))

        # Test if generated basis have good shape
        assert len(W) == n_subjects
        for w in W:
            assert w.shape == (n_components, n_voxels)

        assert len(S) == n_sessions
        for j, s in enumerate(S):
            assert s.shape == (n_components, n_timeframes[j])


def test_bad_aggregate():
    with pytest.raises(ValueError,
                       match="aggregate can have only value mean or None"):
        FastSRM(aggregate="invalid")


def test_check_atlas():
    assert check_atlas(None) is None
    with pytest.raises(ValueError,
                       match=("Atlas is stored using type <class 'list'> "
                              "which is neither np.ndarray or str")):
        check_atlas([])

    A = np.random.rand(10, 100)
    assert check_atlas(A) == (10, 100)

    with tempfile.TemporaryDirectory() as datadir:
        f = os.path.join(datadir, "atlas")
        np.save(f, A)
        assert check_atlas(f + ".npy") == (10, 100)

    A = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5])
    assert check_atlas(A) == (5, 11)

    with tempfile.TemporaryDirectory() as datadir:
        f = os.path.join(datadir, "atlas")
        np.save(f, A)
        assert check_atlas(f + ".npy") == (5, 11)

    with pytest.raises(ValueError,
                       match=("Atlas has 3 axes. It should have either "
                              "1 or 2 axes.")):
        check_atlas(np.random.rand(5, 1, 2))

    with pytest.raises(ValueError,
                       match=(r"Number of regions in the atlas is lower than "
                              r"the number of components \(3 < 5\)")):
        check_atlas(np.random.rand(3, 10), n_components=5)

    with pytest.raises(ValueError,
                       match=(r"Number of regions in the atlas is bigger than "
                              r"the number of voxels \(5 > 2\)")):
        check_atlas(np.random.rand(5, 2))


empty_list_error = "%s is a list of length 0 which is not valid"
array_type_error = ("%s should be of type np.ndarray but is of type %s")
array_2axis_error = ("%s must have exactly 2 axes but has %i axes")


def test_check_imgs():
    with pytest.raises(
            ValueError,
            match=(r"Since imgs is a list, it should be a list of list "
                   r"of arrays or a list "
                   r"of arrays but imgs\[0\] has type <class 'str'>")):
        check_imgs(["bla"])

    with pytest.raises(
            ValueError,
            match=("Input imgs should either be a list or an array but "
                   "has type <class 'str'>")):
        check_imgs("bla")

    with pytest.raises(ValueError, match=empty_list_error % "imgs"):
        check_imgs([])

    with pytest.raises(
            ValueError,
            match=r"imgs\[1\] has type <class 'str'> whereas imgs\[0\] has "
            "type <class 'int'>. This is inconsistent."):
        check_imgs([0, "bla"])

    with pytest.raises(ValueError, match=empty_list_error % r"imgs\[0\]"):
        check_imgs([[]])

    with pytest.raises(
            ValueError,
            match=(r"imgs\[1\] has length 1 whereas imgs\[0\] has length 2."
                   " All subjects should have the same number of sessions.")):
        check_imgs([["a", "a"], ["a"]])

    with pytest.raises(ValueError,
                       match=array_type_error %
                       (r"imgs\[0\]\[0\]", r"<class 'str'>")):
        check_imgs([["bka"]])

    with pytest.raises(ValueError,
                       match=array_2axis_error % (r"imgs\[0\]\[0\]", 1)):
        check_imgs([[np.random.rand(5)]])

    with pytest.raises(ValueError,
                       match=array_2axis_error % (r"imgs\[0\]", 1)):
        check_imgs([np.random.rand(5)])

    with pytest.raises(ValueError,
                       match=(r"imgs\[0, 0\] is stored using type "
                              "<class 'numpy.float64'> which is not a str")):
        check_imgs(np.random.rand(5, 3))

    with pytest.raises(ValueError, match=array_2axis_error % (r"imgs", 1)):
        check_imgs(np.random.rand(5))

    with pytest.raises(
            ValueError,
            match=("The number of subjects should be greater than 1")):
        check_imgs([np.random.rand(5, 3)])

    with pytest.raises(
            ValueError,
            match=("Subject 1 Session 0 does not have the same number "
                   "of timeframes as Subject 0 Session 0")):
        check_imgs([np.random.rand(10, 5), np.random.rand(10, 10)])

    with pytest.raises(
            ValueError,
            match=("Subject 1 Session 0 does not have the same number "
                   "of voxels as Subject 0 Session 0")):
        check_imgs([np.random.rand(10, 5), np.random.rand(20, 5)])

    with pytest.raises(
            ValueError,
            match=("Total number of timeframes is shorter than number "
                   r"of components \(5 < 8\)")):
        check_imgs([np.random.rand(10, 5),
                    np.random.rand(10, 5)],
                   n_components=8)

    with pytest.raises(
            ValueError,
            match=("Number of voxels in the atlas is not the same as "
                   r"the number of voxels in input data \(11 != 10\)")):
        check_imgs([np.random.rand(10, 5),
                    np.random.rand(10, 5)],
                   n_components=3,
                   atlas_shape=(8, 11))


def test_check_shared():
    n_subjects = 2
    n_sessions = 2
    input_shapes = np.zeros((n_subjects, n_sessions, 2))
    input_shapes[0, 0, 0] = 10
    input_shapes[0, 0, 1] = 3
    input_shapes[0, 1, 0] = 10
    input_shapes[0, 1, 1] = 2
    input_shapes[1, 0, 0] = 10
    input_shapes[1, 0, 1] = 3
    input_shapes[1, 1, 0] = 10
    input_shapes[1, 1, 1] = 2

    shared_list_list = [[
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[1, 2], [4, 5]]),
    ], [
        np.array([[2, 3, 4], [5, 6, 7]]),
        np.array([[2, 3], [5, 6]]),
    ]]

    shared_list_subjects = [
        np.array([[1, 2, 3, 1, 2], [4, 5, 6, 4, 5]]),
        np.array([[2, 3, 4, 2, 3], [5, 6, 7, 5, 6]])
    ]

    shared_list_sessions = [
        np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
        np.array([[1.5, 2.5], [4.5, 5.5]]),
    ]

    shared_array = np.array([[1.5, 2.5, 3.5, 1.5, 2.5],
                             [4.5, 5.5, 6.5, 4.5, 5.5]])

    with pytest.raises(ValueError,
                       match=(r"shared_response should be either a list or an "
                              "array but is of type <class 'str'>")):
        check_shared_response("bla")

    with pytest.raises(
            ValueError,
            match=(r"shared_response is a list but shared_response\[0\] "
                   "is neither a list or an array. This is invalid.")):
        check_shared_response(["bla", "bli"])

    with pytest.raises(
            ValueError,
            match=(r"shared_response\[0\] is a list but shared_response\[1\] "
                   "is not a list this is incompatible")):
        check_shared_response(
            [[np.random.rand(2, 2)], np.array([1])], aggregate=None)

    with pytest.raises(ValueError,
                       match=(r"shared_response\[1\] has len 1 whereas "
                              r"shared_response\[0\] has len 2. They should "
                              "have same len")):
        check_shared_response([[np.random.rand(2, 2),
                                np.random.rand(2, 2)], [np.random.rand(2, 2)]],
                              aggregate=None)

    with pytest.raises(
            ValueError,
            match=('Number of timeframes in input images during session 0 '
                   'does not match the number of timeframes during session '
                   r'0 of shared_response \(2 != 3\)')):
        check_shared_response(
            [np.random.rand(2, 2), np.random.rand(2, 2)],
            aggregate="mean",
            input_shapes=input_shapes)

    with pytest.raises(ValueError,
                       match=("Number of components in shared_response "
                              "during session 0 is different than "
                              "the number of components of the "
                              r"model \(2 != 4\)")):
        check_shared_response(np.random.rand(2, 10), n_components=4)

    with pytest.raises(ValueError,
                       match=("self.aggregate has value 'mean' but shared "
                              "response is a list of list. "
                              "This is incompatible")):
        added_session, reshaped_shared = check_shared_response(
            shared_list_list,
            aggregate="mean",
            n_components=2,
            input_shapes=input_shapes)

    added_session, reshaped_shared = check_shared_response(
        shared_list_subjects,
        aggregate=None,
        n_components=2,
        input_shapes=input_shapes)
    assert added_session
    assert_array_almost_equal(np.array(reshaped_shared),
                              shared_array.reshape(1, 2, 5))

    added_session, reshaped_shared = check_shared_response(
        shared_list_sessions,
        aggregate="mean",
        n_components=2,
        input_shapes=input_shapes)
    assert not added_session
    for j in range(len(reshaped_shared)):
        assert_array_almost_equal(reshaped_shared[j], shared_list_sessions[j])

    added_session, reshaped_shared = check_shared_response(
        shared_array,
        aggregate="mean",
        n_components=2,
        input_shapes=input_shapes)
    assert added_session
    assert_array_almost_equal(np.array(reshaped_shared),
                              shared_array.reshape(1, 2, 5))


def test_reduce_data_dummyatlases():
    n_jobs = 1
    with tempfile.TemporaryDirectory() as datadir:
        for n_timeframes in ([25, 24], [25, 25]):
            # We authorize different timeframes for different sessions
            # but they should be the same across subject
            n_voxels = 10
            n_subjects = 2
            n_components = 3  # number of components used for SRM model
            n_sessions = len(n_timeframes)

            np.random.seed(0)
            paths, _, _ = generate_data(n_voxels, n_timeframes, n_subjects,
                                        n_components, datadir)

            # test atlas that reduces nothing
            atlas = np.arange(1, n_voxels + 1)
            data = reduce_data(paths,
                               atlas=atlas,
                               n_jobs=n_jobs,
                               low_ram=False)
            for i in range(n_subjects):
                for j in range(n_sessions):
                    assert_array_almost_equal(data[i, j].T,
                                              np.load(paths[i, j]))

            # test atlas that reduces everything
            atlas = np.ones(n_voxels)
            data = reduce_data(paths,
                               atlas=atlas,
                               n_jobs=n_jobs,
                               low_ram=False)
            for i in range(n_subjects):
                for j in range(n_sessions):
                    assert_array_almost_equal(
                        data[i, j].T.flatten(),
                        np.mean(np.load(paths[i, j]), axis=0))


def test_reduce_data_outputshapes():
    n_jobs = 1
    with tempfile.TemporaryDirectory() as datadir:
        for n_timeframes in ([25, 24], [25, 25]):
            # We authorize different timeframes for different sessions
            # but they should be the same across subject
            n_voxels = 10
            n_subjects = 2
            n_components = 3  # number of components used for SRM model
            n_supervoxels = 5  # number of components of the atlas
            n_sessions = len(n_timeframes)

            np.random.seed(0)
            paths, _, _ = generate_data(n_voxels, n_timeframes, n_subjects,
                                        n_components, datadir)

            # Test if reduced data has the good shape
            # probabilistic atlas
            atlas = np.random.rand(n_supervoxels, n_voxels)
            data = reduce_data(paths,
                               atlas=atlas,
                               n_jobs=n_jobs,
                               low_ram=False,
                               temp_dir=None)

            for subject in range(n_subjects):
                for session in range(n_sessions):
                    assert data[subject, session].shape == (
                        n_timeframes[session], n_supervoxels)

            # deterministic atlas
            det_atlas = np.round(np.random.rand(n_voxels) * n_supervoxels)
            n_unique = len(np.unique(det_atlas)[1:])
            while n_unique != n_supervoxels:
                det_atlas = np.round(np.random.rand(n_voxels) * n_supervoxels)
                n_unique = len(np.unique(det_atlas)[1:])

            data = reduce_data(paths,
                               atlas=det_atlas,
                               n_jobs=n_jobs,
                               low_ram=True,
                               temp_dir=datadir)

            for subject in range(n_subjects):
                for session in range(n_sessions):
                    assert (np.load(data[subject, session]).shape == (
                        n_timeframes[session], n_supervoxels))


def test_reduced_data_srm():
    n_jobs = 1
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)

        # We authorize different timeframes for different sessions but
        # they should be the same across subject
        n_voxels = 10
        n_timeframes = [25, 24]
        n_subjects = 5
        n_components = 3  # number of components used for SRM model
        n_sessions = len(n_timeframes)

        paths, W, S = generate_data(n_voxels, n_timeframes, n_subjects,
                                    n_components, datadir, 0)

        atlas = np.arange(1, n_voxels + 1)
        data = reduce_data(paths,
                           atlas=atlas,
                           n_jobs=n_jobs,
                           low_ram=False,
                           temp_dir=None)

        # Test if shared response has the good shape
        shared_response_list = \
            _reduced_space_compute_shared_response(
                data,
                reduced_basis_list=None,
                n_components=n_components
            )
        assert len(shared_response_list) == n_sessions

        for session in range(n_sessions):
            assert (shared_response_list[session].shape == (
                n_timeframes[session], n_components))

        # Test basis from shared response
        for i, sessions in enumerate(paths):
            basis = _compute_basis_subject_online(
                sessions, [S[k].T for k in range(len(S))])
            # test shape
            assert basis.shape == (n_components, n_voxels)
            # test orthogonality
            assert np.allclose(basis.dot(basis.T), np.eye(n_components))
            # test correctness
            assert_array_almost_equal(basis, W[i], 2)

        # Test reduced_data_shared_response
        shared_response_list = _reduced_space_compute_shared_response(
            data, reduced_basis_list=W, n_components=n_components)
        for session in range(n_sessions):
            S_real = np.mean(
                [data[i, session].dot(W[i].T) for i in range(n_subjects)],
                axis=0)
            assert_array_almost_equal(shared_response_list[session], S_real)
            assert_array_almost_equal(shared_response_list[session],
                                      S[session].T)

        shared_response_list = fast_srm(data, n_components=n_components)

        for i, sessions in enumerate(paths):
            basis = _compute_basis_subject_online(sessions,
                                                  shared_response_list)
            for j, session in enumerate(sessions):
                assert_array_almost_equal(shared_response_list[j].dot(basis),
                                          np.load(paths[i, j]).T, 3)


def test_compute_and_save():
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)
        n_voxels = 10
        n_timeframes = [25, 24]
        n_subjects = 5
        n_components = 3  # number of components used for SRM model

        paths, W, S = generate_data(n_voxels, n_timeframes, n_subjects,
                                    n_components, datadir, 0)

        for m, subjects in enumerate(paths.T):
            for subject in subjects:
                _compute_and_save_corr_mat(subject, S[m].T, datadir)

        for i, sessions in enumerate(paths):
            basis = _compute_and_save_subject_basis(i, sessions, datadir)

            assert_array_almost_equal(np.load(basis), W[i], 3)


def test_fastsrm_class():
    n_jobs = 1
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)

        # We authorize different timeframes for different sessions
        # but they should be the same across subject
        n_voxels = 10
        n_timeframes = [25, 24]
        n_subjects = 5
        n_components = 3  # number of components used for SRM model

        np.random.seed(0)
        paths, W, S = generate_data(n_voxels, n_timeframes, n_subjects,
                                    n_components, datadir, 0)

        atlas = np.arange(1, n_voxels + 1)
        srm = FastSRM(atlas=atlas,
                      n_components=n_components,
                      n_iter=10,
                      temp_dir=datadir,
                      low_ram=True,
                      verbose=True,
                      n_jobs=n_jobs)

        # Raises an error because model is not fitted yet
        with pytest.raises(NotFittedError):
            srm.transform(paths)

        srm.fit(paths)

        # An error can occur if temporary directory already exists
        with pytest.raises(ValueError,
                           match=("Path %s already exists. When a model "
                                  "is used, filesystem should be "
                                  r"cleaned by using the .clean\(\) "
                                  "method" % re.escape(srm.temp_dir))):
            # Error can occur if the filesystem is uncleaned
            create_temp_dir(srm.temp_dir)
            create_temp_dir(srm.temp_dir)

        shared_response = srm.transform(paths)

        # Raise error when wrong index
        with pytest.raises(ValueError,
                           match=("subjects_indexes should be either "
                                  "a list, an array or None but "
                                  "received type <class 'int'>")):
            srm.transform(paths, subjects_indexes=1000)

        with pytest.raises(ValueError,
                           match=("subjects_indexes should be either "
                                  "a list, an array or None but "
                                  "received type <class 'int'>")):
            srm.inverse_transform(shared_response, subjects_indexes=1000)

        with pytest.raises(ValueError,
                           match=("sessions_indexes should be either "
                                  "a list, an array or None but "
                                  "received type <class 'int'>")):
            srm.inverse_transform(shared_response, sessions_indexes=1000)

        with pytest.raises(ValueError,
                           match=("Input data imgs has len 5 whereas "
                                  "subject_indexes has len 1. "
                                  "The number of basis used to compute "
                                  "the shared response should be equal to "
                                  "the number of subjects in imgs")):
            srm.transform(paths, subjects_indexes=[0])

        with pytest.raises(ValueError,
                           match=("Index 1 of subjects_indexes has value 8 "
                                  "whereas value should be between 0 and 4")):
            srm.transform(paths[:2], subjects_indexes=[0, 8])

        with pytest.raises(ValueError,
                           match=("Index 1 of sessions_indexes has value 8 "
                                  "whereas value should be between 0 and 1")):
            srm.inverse_transform(shared_response, sessions_indexes=[0, 8])

        # Check behavior of .clean
        assert os.path.exists(srm.temp_dir)
        srm.clean()
        assert not os.path.exists(srm.temp_dir)


n_voxels = 10
n_subjects = 5
n_components = 3  # number of components used for SRM model


def apply_aggregate(shared_response, aggregate, input_format):
    if aggregate is None:
        if input_format == "list_of_array":
            return [np.mean(shared_response, axis=0)]
        else:
            return [
                np.mean([
                    shared_response[i][j] for i in range(len(shared_response))
                ],
                        axis=0) for j in range(len(shared_response[0]))
            ]
    else:
        if input_format == "list_of_array":
            return [shared_response]
        else:
            return shared_response


def apply_input_format(X, input_format):
    if input_format == "array":
        n_sessions = len(X[0])
        XX = [[np.load(X[i, j]) for j in range(len(X[i]))]
              for i in range(len(X))]
    elif input_format == "list_of_array":
        XX = [[x] for x in X]
        n_sessions = 1
    else:
        XX = X
        n_sessions = len(X[0])
    return XX, n_sessions


@pytest.mark.parametrize(
    "input_format, low_ram, tempdir, atlas, n_jobs, n_timeframes, aggregate",
    [("array", True, True, None, 1, [25, 25], "mean"),
     ("list_of_list", False, False, np.arange(1, n_voxels + 1), 1, [25, 24
                                                                    ], None),
     ("list_of_array", True, False, np.eye(n_voxels), 1, [25, 25], None),
     ("list_of_array", False, True, None, 1, [25, 24], "mean")])
def test_fastsrm_class_correctness(input_format, low_ram, tempdir, atlas,
                                   n_jobs, n_timeframes, aggregate):
    with tempfile.TemporaryDirectory() as datadir:
        np.random.seed(0)
        X, W, S = generate_data(n_voxels, n_timeframes, n_subjects,
                                n_components, datadir, 0, input_format)

        XX, n_sessions = apply_input_format(X, input_format)

        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        srm = FastSRM(atlas=atlas,
                      n_components=n_components,
                      n_iter=10,
                      temp_dir=temp_dir,
                      low_ram=low_ram,
                      verbose=True,
                      n_jobs=n_jobs,
                      aggregate=aggregate,
                      seed=0)

        # Check that there is no difference between fit_transform
        # and fit then transform

        srm.fit(X)
        basis = [safe_load(b) for b in srm.basis_list]
        shared_response_raw = srm.transform(X)
        shared_response = apply_aggregate(shared_response_raw, aggregate,
                                          input_format)
        shared_response_fittransform = apply_aggregate(srm.fit_transform(X),
                                                       aggregate, input_format)

        for j in range(n_sessions):
            assert_array_almost_equal(shared_response_fittransform[j],
                                      shared_response[j])

        # Check that the decomposition works
        for i in range(n_subjects):
            for j in range(n_sessions):
                assert_array_almost_equal(shared_response[j].T.dot(basis[i]),
                                          XX[i][j].T, 3)

        # Check that if we use all subjects but one if gives almost the
        # same shared response
        shared_response_partial_raw = srm.transform(X[1:5],
                                                    subjects_indexes=list(
                                                        range(1, 5)))

        shared_response_partial = apply_aggregate(shared_response_partial_raw,
                                                  aggregate, input_format)
        for j in range(n_sessions):
            assert_array_almost_equal(shared_response_partial[j],
                                      shared_response[j], 3)

        # Check that if we perform add 2 times the same subject we
        # obtain the same decomposition
        srm.add_subjects(X[:1], shared_response_raw)
        assert_array_almost_equal(safe_load(srm.basis_list[0]),
                                  safe_load(srm.basis_list[-1]))


@pytest.mark.parametrize(
    "input_format, low_ram, tempdir, atlas, n_jobs, n_timeframes, aggregate",
    [("array", True, True, None, 1, [25, 25], "mean"),
     ("list_of_list", False, False, np.arange(1, n_voxels + 1), 1, [25, 24
                                                                    ], None),
     ("list_of_array", True, False, np.eye(n_voxels), 1, [25, 25], None),
     ("list_of_array", False, True, None, 1, [25, 24], "mean")])
def test_class_srm_inverse_transform(input_format, low_ram, tempdir, atlas,
                                     n_jobs, n_timeframes, aggregate):

    with tempfile.TemporaryDirectory() as datadir:
        X, W, S = generate_data(n_voxels, n_timeframes, n_subjects,
                                n_components, datadir, 0, input_format)

        if tempdir:
            temp_dir = datadir
        else:
            temp_dir = None

        srm = FastSRM(atlas=atlas,
                      n_components=n_components,
                      n_iter=10,
                      temp_dir=temp_dir,
                      low_ram=low_ram,
                      verbose=True,
                      n_jobs=n_jobs,
                      aggregate=aggregate,
                      seed=0)

        # Check that there is no difference between fit_transform
        # and fit then transform

        srm.fit(X)
        shared_response_raw = srm.transform(X)
        # Check inverse transform
        if input_format == "list_of_array":
            reconstructed_data = srm.inverse_transform(shared_response_raw,
                                                       subjects_indexes=[0, 2])
            for i, ii in enumerate([0, 2]):
                assert_array_almost_equal(reconstructed_data[i], X[ii], 3)

            reconstructed_data = srm.inverse_transform(shared_response_raw,
                                                       subjects_indexes=None)
            for i in range(len(X)):
                assert_array_almost_equal(reconstructed_data[i], X[i], 3)
        else:
            reconstructed_data = srm.inverse_transform(shared_response_raw,
                                                       sessions_indexes=[1],
                                                       subjects_indexes=[0, 2])
            for i, ii in enumerate([0, 2]):
                for j, jj in enumerate([1]):
                    assert_array_almost_equal(reconstructed_data[i][j],
                                              safe_load(X[ii][jj]), 3)

            reconstructed_data = srm.inverse_transform(shared_response_raw,
                                                       subjects_indexes=None,
                                                       sessions_indexes=None)

            for i in range(len(X)):
                for j in range(len(X[i])):
                    assert_array_almost_equal(reconstructed_data[i][j],
                                              safe_load(X[i][j]), 3)


def test_fastsrm_identity():
    # In this function we test whether fastsrm and DetSRM have
    # identical behavior when atlas=None

    # We authorize different timeframes for different sessions
    # but they should be the same across subject
    n_voxels = 8
    n_timeframes = [4, 5, 6]
    n_subjects = 2
    n_components = 3  # number of components used for SRM model

    np.random.seed(0)
    paths, W, S = generate_data(n_voxels,
                                n_timeframes,
                                n_subjects,
                                n_components,
                                None,
                                input_format="list_of_array")

    # Test if generated data has the good shape
    for subject in range(n_subjects):
        assert paths[subject].shape == (n_voxels, np.sum([n_timeframes]))

    srm = DetSRM(n_iter=11, features=3, rand_seed=0)
    srm.fit(paths)
    shared = srm.transform(paths)

    fastsrm = FastSRM(atlas=None,
                      n_components=3,
                      verbose=True,
                      seed=0,
                      n_jobs=1,
                      n_iter=10)
    fastsrm.fit(paths)
    shared_fast = fastsrm.transform(paths)

    assert_array_almost_equal(shared_fast, np.mean(shared, axis=0))

    for i in range(n_subjects):
        assert_array_almost_equal(safe_load(fastsrm.basis_list[i]),
                                  srm.w_[i].T)


def load_and_concat(paths):
    """
    Take list of path and yields input data for ProbSRM
    Parameters
    ----------
    paths
    Returns
    -------
    X
    """
    X = []
    for i in range(len(paths)):
        X_i = np.concatenate(
            [np.load(paths[i, j]) for j in range(len(paths[i]))], axis=1)
        X.append(X_i)
    return X


def test_consistency_paths_data():
    with tempfile.TemporaryDirectory() as datadir:
        # In this function we test that input format
        # does not change the results

        n_voxels = 8
        n_timeframes = [4, 5, 6]
        n_subjects = 2
        n_components = 3  # number of components used for SRM model

        np.random.seed(0)
        paths, W, S = generate_data(n_voxels,
                                    n_timeframes,
                                    n_subjects,
                                    n_components,
                                    datadir,
                                    input_format="array")

        print()
        print("shape", paths.shape)

        fastsrm = FastSRM(
            n_components=3,
            atlas=None,
            verbose=True,
            seed=0,
            n_jobs=1,
            n_iter=10,
        )

        fastsrm.fit(paths)
        b0 = fastsrm.basis_list[0]

        fastsrm.fit(load_and_concat(paths))
        b1 = fastsrm.basis_list[0]

        assert_array_almost_equal(b0, b1)
