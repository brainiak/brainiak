"""Fast Shared Response Model (FastSRM)

The implementation is based on the following publications:

.. [Richard2019] "Fast Shared Response Model for fMRI data"
    H. Richard, L. Martin, A. Pinho, J. Pillow, B. Thirion, 2019
    https://arxiv.org/pdf/1909.12537.pdf
"""

# Author: Hugo Richard

import hashlib
import logging
import os

import numpy as np
import scipy
from joblib import Parallel, delayed

from brainiak.funcalign.srm import DetSRM
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import uuid

__all__ = [
    "FastSRM",
]

logger = logging.getLogger(__name__)


def get_shape(path):
    """Get shape of saved np array
    Parameters
    ----------
    path: str
        path to np array
    """
    f = open(path, "rb")
    version = np.lib.format.read_magic(f)
    shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
    f.close()
    return shape


def safe_load(data):
    """If data is an array returns data else returns np.load(data)"""
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.load(data)


def safe_encode(img):
    if isinstance(img, np.ndarray):
        name = hashlib.md5(img.tostring()).hexdigest()
    else:
        name = hashlib.md5(img.encode()).hexdigest()
    return name


def assert_non_empty_list(input_list, list_name):
    """
    Check that input list is not empty
    Parameters
    ----------
    input_list: list
    list_name: str
        Name of the list
    """
    if len(input_list) == 0:
        raise ValueError("%s is a list of length 0 which is not valid" %
                         list_name)


def assert_array_2axis(array, name_array):
    """Check that input is an np array with 2 axes

    Parameters
    ----------
    array: np array
    name_array: str
        Name of the array
    """

    if not isinstance(array, np.ndarray):
        raise ValueError("%s should be of type "
                         "np.ndarray but is of type %s" %
                         (name_array, type(array)))

    if len(array.shape) != 2:
        raise ValueError("%s must have exactly 2 axes "
                         "but has %i axes" % (name_array, len(array.shape)))


def assert_valid_index(indexes, max_value, name_indexes):
    """
    Check that indexes are between 0 and max_value and number
    of indexes is less than max_value
    """
    for i, ind_i in enumerate(indexes):
        if ind_i < 0 or ind_i >= max_value:
            raise ValueError("Index %i of %s has value %i "
                             "whereas value should be between 0 and %i" %
                             (i, name_indexes, ind_i, max_value - 1))


def _check_imgs_list(imgs):
    """
    Checks that imgs is a non empty list of elements of the same type

    Parameters
    ----------

    imgs : list
    """
    # Check the list is non empty
    assert_non_empty_list(imgs, "imgs")

    # Check that all input have same type
    for i in range(len(imgs)):
        if not isinstance(imgs[i], type(imgs[0])):
            raise ValueError("imgs[%i] has type %s whereas "
                             "imgs[%i] has type %s. "
                             "This is inconsistent." %
                             (i, type(imgs[i]), 0, type(imgs[0])))


def _check_imgs_list_list(imgs):
    """
    Check input images if they are list of list of arrays

    Parameters
    ----------

    imgs : list of list of array of shape [n_voxels, n_components]
            imgs is a list of list of arrays where element i, j of
            the array is a numpy array of shape [n_voxels, n_timeframes] that
            contains the data of subject i collected during session j.
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

    Returns
    -------
    shapes: array
        Shape of input images
    """
    n_subjects = len(imgs)

    # Check that the number of session is not 0
    assert_non_empty_list(imgs[0], "imgs[%i]" % 0)

    # Check that the number of sessions is the same for all subjects
    n_sessions = None
    for i in range(len(imgs)):
        if n_sessions is None:
            n_sessions = len(imgs[i])
        if n_sessions != len(imgs[i]):
            raise ValueError("imgs[%i] has length %i whereas imgs[%i] "
                             "has length %i. All subjects should have "
                             "the same number of sessions." %
                             (i, len(imgs[i]), 0, len(imgs[0])))

    shapes = np.zeros((n_subjects, n_sessions, 2))
    # Run array-level checks
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            assert_array_2axis(imgs[i][j], "imgs[%i][%i]" % (i, j))
            shapes[i, j, :] = imgs[i][j].shape

    return shapes


def _check_imgs_list_array(imgs):
    """
    Check input images if they are list of arrays.
    In this case returned images are a list of list of arrays
    where element i,j of the array is a numpy array of
    shape [n_voxels, n_timeframes] that contains the data of subject i
    collected during session j.

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions]
            imgs is a list of arrays where element i of the array is
            a numpy array of shape [n_voxels, n_timeframes] that contains the
            data of subject i (number of sessions is implicitly 1)
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

    Returns
    -------
    shapes: array
        Shape of input images
    new_imgs: list of list of array of shape [n_voxels, n_components]
    """
    n_subjects = len(imgs)
    n_sessions = 1
    shapes = np.zeros((n_subjects, n_sessions, 2))
    new_imgs = []
    for i in range(len(imgs)):
        assert_array_2axis(imgs[i], "imgs[%i]" % i)
        shapes[i, 0, :] = imgs[i].shape
        new_imgs.append([imgs[i]])

    return new_imgs, shapes


def _check_imgs_array(imgs):
    """Check input image if it is an array

    Parameters
    ----------
    imgs : array of str, shape=[n_subjects, n_sessions]
            Element i, j of the array is a path to the data of subject i
            collected during session j.
            Data are loaded with numpy.load and expected
            shape is [n_voxels, n_timeframes]
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

    Returns
    -------
    shapes : array
        Shape of input images
    """
    assert_array_2axis(imgs, "imgs")
    n_subjects, n_sessions = imgs.shape

    shapes = np.zeros((n_subjects, n_sessions, 2))
    for i in range(n_subjects):
        for j in range(n_sessions):
            if not (isinstance(imgs[i, j], str) or isinstance(
                    imgs[i, j], np.str_) or isinstance(imgs[i, j], str)):
                raise ValueError("imgs[%i, %i] is stored using "
                                 "type %s which is not a str" %
                                 (i, j, type(imgs[i, j])))
            shapes[i, j, :] = get_shape(imgs[i, j])
    return shapes


def _check_shapes_components(n_components, n_timeframes):
    """Check that n_timeframes is greater than number of components"""


def _check_shapes_atlas_compatibility(n_voxels,
                                      n_timeframes,
                                      n_components=None,
                                      atlas_shape=None):
    if n_components is not None:
        if np.sum(n_timeframes) < n_components:
            raise ValueError("Total number of timeframes is shorter than "
                             "number of components (%i < %i)" %
                             (np.sum(n_timeframes), n_components))

    if atlas_shape is not None:
        n_supervoxels, n_atlas_voxels = atlas_shape
        if n_atlas_voxels != n_voxels:
            raise ValueError(
                "Number of voxels in the atlas is not the same "
                "as the number of voxels in input data (%i != %i)" %
                (n_atlas_voxels, n_voxels))


def _check_shapes(shapes,
                  n_components=None,
                  atlas_shape=None,
                  ignore_nsubjects=False):
    """Check that number of voxels is the same for each subjects. Number of
    timeframes can vary between sessions but must be consistent across
    subjects

    Parameters
    ----------
    shapes : array of shape (n_subjects, n_sessions, 2)
        Array of shapes of input images
    """
    n_subjects, n_sessions, _ = shapes.shape

    if n_subjects <= 1 and not ignore_nsubjects:
        raise ValueError("The number of subjects should be greater than 1")

    n_timeframes_list = [None] * n_sessions
    n_voxels = None
    for n in range(n_subjects):
        for m in range(n_sessions):
            if n_timeframes_list[m] is None:
                n_timeframes_list[m] = shapes[n, m, 1]

            if n_voxels is None:
                n_voxels = shapes[m, n, 0]

            if n_timeframes_list[m] != shapes[n, m, 1]:
                raise ValueError("Subject %i Session %i does not have the "
                                 "same number of timeframes "
                                 "as Subject %i Session %i" % (n, m, 0, m))

            if n_voxels != shapes[n, m, 0]:
                raise ValueError("Subject %i Session %i"
                                 " does not have the same number of voxels as "
                                 "Subject %i Session %i." % (n, m, 0, 0))

    _check_shapes_atlas_compatibility(n_voxels, np.sum(n_timeframes_list),
                                      n_components, atlas_shape)


def check_atlas(atlas, n_components=None):
    """ Check input atlas

    Parameters
    ----------
    atlas :  array, shape=[n_supervoxels, n_voxels] or array, shape=[n_voxels]
     or str or None
        Probabilistic or deterministic atlas on which to project the data
        Deterministic atlas is an array of shape [n_voxels,] where values
        range from 1 to n_supervoxels. Voxels labelled 0 will be ignored.
        If atlas is a str the corresponding array is loaded with numpy.load
        and expected shape is (n_voxels,) for a deterministic atlas and
        (n_supervoxels, n_voxels) for a probabilistic atlas.

    n_components : int
        Number of timecourses of the shared coordinates

    Returns
    -------
    shape : array or None
        atlas shape
    """
    if atlas is None:
        return None

    if not (isinstance(atlas, np.ndarray) or isinstance(atlas, str)
            or isinstance(atlas, np.str_) or isinstance(atlas, str)):
        raise ValueError("Atlas is stored using "
                         "type %s which is neither np.ndarray or str" %
                         type(atlas))

    if isinstance(atlas, np.ndarray):
        shape = atlas.shape
    else:
        shape = get_shape(atlas)

    if len(shape) == 1:
        # We have a deterministic atlas
        atlas_array = safe_load(atlas)
        n_voxels = atlas_array.shape[0]
        n_supervoxels = len(np.unique(atlas_array)) - 1
        shape = (n_supervoxels, n_voxels)
    elif len(shape) != 2:
        raise ValueError(
            "Atlas has %i axes. It should have either 1 or 2 axes." %
            len(shape))

    n_supervoxels, n_voxels = shape

    if n_supervoxels > n_voxels:
        raise ValueError("Number of regions in the atlas is bigger than "
                         "the number of voxels (%i > %i)" %
                         (n_supervoxels, n_voxels))

    if n_components is not None:
        if n_supervoxels < n_components:
            raise ValueError("Number of regions in the atlas is "
                             "lower than the number of components "
                             "(%i < %i)" % (n_supervoxels, n_components))
    return shape


def check_imgs(imgs,
               n_components=None,
               atlas_shape=None,
               ignore_nsubjects=False):
    """
    Check input images

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions]
            Element i, j of the array is a path to the data of subject i
            collected during session j.
            Data are loaded with numpy.load and expected
            shape is [n_voxels, n_timeframes]
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

            imgs can also be a list of list of arrays where element i, j of
            the array is a numpy array of shape [n_voxels, n_timeframes] that
            contains the data of subject i collected during session j.

            imgs can also be a list of arrays where element i of the array is
            a numpy array of shape [n_voxels, n_timeframes] that contains the
            data of subject i (number of sessions is implicitly 1)

    Returns
    -------
    reshaped_input: bool
        True if input had to be reshaped to match the
        n_subjects, n_sessions input
    new_imgs: list of list of array or np array
        input imgs reshaped if it is a list of arrays so that it becomes a
        list of list of arrays
    shapes: array
        Shape of input images
    """
    reshaped_input = False
    new_imgs = imgs
    if isinstance(imgs, list):
        _check_imgs_list(imgs)
        if isinstance(imgs[0], list):
            shapes = _check_imgs_list_list(imgs)
        elif isinstance(imgs[0], np.ndarray):
            new_imgs, shapes = _check_imgs_list_array(imgs)
            reshaped_input = True
        else:
            raise ValueError(
                "Since imgs is a list, it should be a list of list "
                "of arrays or a list of arrays but imgs[0] has type %s" %
                type(imgs[0]))
    elif isinstance(imgs, np.ndarray):
        shapes = _check_imgs_array(imgs)
    else:
        raise ValueError(
            "Input imgs should either be a list or an array but has type %s" %
            type(imgs))

    _check_shapes(shapes, n_components, atlas_shape, ignore_nsubjects)

    return reshaped_input, new_imgs, shapes


def check_indexes(indexes, name):
    if not (indexes is None or isinstance(indexes, list)
            or isinstance(indexes, np.ndarray)):
        raise ValueError(
            "%s should be either a list, an array or None but received type %s"
            % (name, type(indexes)))


def _check_shared_response_list_of_list(shared_response, n_components,
                                        input_shapes):

    # Check that shared_response is indeed a list of list of arrays
    n_subjects = len(shared_response)
    n_sessions = None
    for i in range(len(shared_response)):
        if not isinstance(shared_response[i], list):
            raise ValueError("shared_response[0] is a list but "
                             "shared_response[%i] is not a list "
                             "this is incompatible." % i)
        assert_non_empty_list(shared_response[i], "shared_response[%i]" % i)
        if n_sessions is None:
            n_sessions = len(shared_response[i])
        elif n_sessions != len(shared_response[i]):
            raise ValueError(
                "shared_response[%i] has len %i whereas "
                "shared_response[0] has len %i. They should "
                "have same length" %
                (i, len(shared_response[i]), len(shared_response[0])))
        for j in range(len(shared_response[i])):
            assert_array_2axis(shared_response[i][j],
                               "shared_response[%i][%i]" % (i, j))

    return _check_shared_response_list_sessions([
        np.mean([shared_response[i][j] for i in range(n_subjects)], axis=0)
        for j in range(n_sessions)
    ], n_components, input_shapes)


def _check_shared_response_list_sessions(shared_response, n_components,
                                         input_shapes):
    for j in range(len(shared_response)):
        assert_array_2axis(shared_response[j], "shared_response[%i]" % j)
        if input_shapes is not None:
            if shared_response[j].shape[1] != input_shapes[0][j][1]:
                raise ValueError(
                    "Number of timeframes in input images during "
                    "session %i does not match the number of "
                    "timeframes during session %i "
                    "of shared_response (%i != %i)" %
                    (j, j, shared_response[j].shape[1], input_shapes[0, j, 1]))
        if n_components is not None:
            if shared_response[j].shape[0] != n_components:
                raise ValueError(
                    "Number of components in "
                    "shared_response during session %i is "
                    "different than "
                    "the number of components of the model (%i != %i)" %
                    (j, shared_response[j].shape[0], n_components))
    return shared_response


def _check_shared_response_list_subjects(shared_response, n_components,
                                         input_shapes):
    for i in range(len(shared_response)):
        assert_array_2axis(shared_response[i], "shared_response[%i]" % i)

    return _check_shared_response_array(np.mean(shared_response, axis=0),
                                        n_components, input_shapes)


def _check_shared_response_array(shared_response, n_components, input_shapes):
    assert_array_2axis(shared_response, "shared_response")
    if input_shapes is None:
        new_input_shapes = None
    else:
        n_subjects, n_sessions, _ = input_shapes.shape
        new_input_shapes = np.zeros((n_subjects, 1, 2))
        new_input_shapes[:, 0, 0] = input_shapes[:, 0, 0]
        new_input_shapes[:, 0, 1] = np.sum(input_shapes[:, :, 1], axis=1)
    return _check_shared_response_list_sessions([shared_response],
                                                n_components, new_input_shapes)


def check_shared_response(shared_response,
                          aggregate="mean",
                          n_components=None,
                          input_shapes=None):
    """
    Check that shared response has valid input and turn it into
    a session-wise shared response

    Returns
    -------
    added_session: bool
        True if an artificial sessions was added to match the list of
        session input type for shared_response
    reshaped_shared_response: list of arrays
        shared response (reshaped to match the list of session input)
    """
    # Depending on aggregate and shape of input we infer what to do
    if isinstance(shared_response, list):
        assert_non_empty_list(shared_response, "shared_response")
        if isinstance(shared_response[0], list):
            if aggregate == "mean":
                raise ValueError("self.aggregate has value 'mean' but "
                                 "shared response is a list of list. This is "
                                 "incompatible")
            return False, _check_shared_response_list_of_list(
                shared_response, n_components, input_shapes)
        elif isinstance(shared_response[0], np.ndarray):
            if aggregate == "mean":
                return False, _check_shared_response_list_sessions(
                    shared_response, n_components, input_shapes)
            else:
                return True, _check_shared_response_list_subjects(
                    shared_response, n_components, input_shapes)
        else:
            raise ValueError("shared_response is a list but "
                             "shared_response[0] is neither a list "
                             "or an array. This is invalid.")
    elif isinstance(shared_response, np.ndarray):
        return True, _check_shared_response_array(shared_response,
                                                  n_components, input_shapes)
    else:
        raise ValueError("shared_response should be either "
                         "a list or an array but is of type %s" %
                         type(shared_response))


def create_temp_dir(temp_dir):
    """
    This check whether temp_dir exists and creates dir otherwise
    """
    if temp_dir is None:
        return None

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        raise ValueError("Path %s already exists. "
                         "When a model is used, filesystem should be cleaned "
                         "by using the .clean() method" % temp_dir)


def reduce_data_single(subject_index,
                       session_index,
                       img,
                       atlas=None,
                       inv_atlas=None,
                       low_ram=False,
                       temp_dir=None):
    """Reduce data using given atlas

    Parameters
    ----------
    subject_index : int

    session_index : int

    img : str or array
        path to data.
        Data are loaded with numpy.load and expected shape is
         (n_voxels, n_timeframes)
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

        img can also be an array of shape (n_voxels, n_timeframes)

    atlas :  array, shape=[n_supervoxels, n_voxels] or [n_voxels] or None
        Probabilistic or deterministic atlas on which to project the data
        Deterministic atlas is an array of shape [n_voxels,] where values
        range from 1 to n_supervoxels. Voxels labelled 0 will be ignored.

    inv_atlas : array, shape=[n_voxels, n_supervoxels] or None
        Pseudo inverse of the atlas (only for probabilistic atlases)

    temp_dir : str or None
        path to dir where temporary results are stored
        if None temporary results will be stored in memory. This
        can results in memory errors when the number of subjects
        and / or sessions is large

    low_ram : bool
        if True and temp_dir is not None, reduced_data will be saved on disk
        this increases the number of IO but reduces memory complexity when the
         number
        of subject and number of sessions are large

    Returns
    -------

    reduced_data : array, shape=[n_timeframes, n_supervoxels]
        reduced data
    """
    # Here we return to the conventions of the paper
    data = safe_load(img).T

    n_timeframes, n_voxels = data.shape

    # Here we check that input is normalized
    if (np.max(np.abs(np.mean(data, axis=0))) > 1e-6
            or np.max(np.abs(np.var(data, axis=0) - 1))) > 1e-6:
        ValueError("Data in imgs[%i, %i] does not have 0 mean and unit \
        variance. If you are using NiftiMasker to mask your data \
        (nilearn) please use standardize=True." %
                   (subject_index, session_index))

    if inv_atlas is None and atlas is not None:
        atlas_values = np.unique(atlas)
        if 0 in atlas_values:
            atlas_values = atlas_values[1:]

        reduced_data = np.array(
            [np.mean(data[:, atlas == c], axis=1) for c in atlas_values]).T
    elif inv_atlas is not None and atlas is None:
        # this means that it is a probabilistic atlas
        reduced_data = data.dot(inv_atlas)
    else:
        reduced_data = data

    if low_ram:
        name = safe_encode(img)
        path = os.path.join(temp_dir, "reduced_data_" + name)
        np.save(path, reduced_data)
        return path + ".npy"
    else:
        return reduced_data


def reduce_data(imgs, atlas, n_jobs=1, low_ram=False, temp_dir=None):
    """Reduce data using given atlas.
    Work done in parallel across subjects.

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions]
        Element i, j of the array is a path to the data of subject i
        collected during session j.
        Data are loaded with numpy.load and expected shape is
        [n_timeframes, n_voxels]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

        imgs can also be a list of list of arrays where element i, j of
        the array is a numpy array of shape [n_voxels, n_timeframes] that
        contains the data of subject i collected during session j.

        imgs can also be a list of arrays where element i of the array is
        a numpy array of shape [n_voxels, n_timeframes] that contains the
        data of subject i (number of sessions is implicitly 1)

    atlas :  array, shape=[n_supervoxels, n_voxels] or array, shape=[n_voxels]
        or None
        Probabilistic or deterministic atlas on which to project the data
        Deterministic atlas is an array of shape [n_voxels,] where values
        range from 1 to n_supervoxels. Voxels labelled 0 will be ignored.

    n_jobs : integer, optional, default=1
        The number of CPUs to use to do the computation.
         -1 means all CPUs, -2 all CPUs but one, and so on.

    temp_dir : str or None
        path to dir where temporary results are stored
        if None temporary results will be stored in memory. This
        can results in memory errors when the number of subjects
        and / or sessions is large

    low_ram : bool
        if True and temp_dir is not None, reduced_data will be saved on disk
        this increases the number of IO but reduces memory complexity when
         the number of subject and/or sessions is large

    Returns
    -------

    reduced_data_list : array of str, shape=[n_subjects, n_sessions]
    or array, shape=[n_subjects, n_sessions, n_timeframes, n_supervoxels]
        Element i, j of the array is a path to the data of subject i collected
         during session j.
        Data are loaded with numpy.load and expected shape is
        [n_timeframes, n_supervoxels]
        or Element i, j of the array is the data in array of
        shape=[n_timeframes, n_supervoxels]
        n_timeframes and n_supervoxels
         are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1
    """
    if atlas is None:
        A = None
        A_inv = None
    else:
        loaded_atlas = safe_load(atlas)
        if len(loaded_atlas.shape) == 2:
            A = None
            A_inv = loaded_atlas.T.dot(
                np.linalg.inv(loaded_atlas.dot(loaded_atlas.T)))
        else:
            A = loaded_atlas
            A_inv = None

    n_subjects = len(imgs)
    n_sessions = len(imgs[0])

    reduced_data_list = Parallel(n_jobs=n_jobs)(
        delayed(reduce_data_single)(i,
                                    j,
                                    imgs[i][j],
                                    atlas=A,
                                    inv_atlas=A_inv,
                                    low_ram=low_ram,
                                    temp_dir=temp_dir)
        for i in range(n_subjects) for j in range(n_sessions))

    # Check if reduced_data_list is ragged, in this case, we need to create a
    # numpy array with dtype=object, numpy no longer does this automatically
    try:
        reduced_data_list = np.array(reduced_data_list)
    except ValueError as ex:
        if "setting an array element with a sequence" in str(ex):
            reduced_data_list = np.array(reduced_data_list, dtype=object)
        else:
            raise ex

    if low_ram:
        reduced_data_list = np.reshape(reduced_data_list,
                                       (n_subjects, n_sessions))
    else:
        if len(np.array(reduced_data_list).shape) == 1:
            reduced_data_list = np.reshape(reduced_data_list,
                                           (n_subjects, n_sessions))
        else:
            n_timeframes, n_supervoxels = np.array(reduced_data_list).shape[1:]
            reduced_data_list = np.reshape(
                reduced_data_list,
                (n_subjects, n_sessions, n_timeframes, n_supervoxels))

    return reduced_data_list


def _reduced_space_compute_shared_response(reduced_data_list,
                                           reduced_basis_list,
                                           n_components=50):
    """Compute shared response with basis fixed in reduced space

    Parameters
    ----------

    reduced_data_list : array of str, shape=[n_subjects, n_sessions]
    or array, shape=[n_subjects, n_sessions, n_timeframes, n_supervoxels]
        Element i, j of the array is a path to the data of subject i
        collected during session j.
        Data are loaded with numpy.load and expected shape is
        [n_timeframes, n_supervoxels]
        or Element i, j of the array is the data in array of
        shape=[n_timeframes, n_supervoxels]
        n_timeframes and n_supervoxels are
        assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    reduced_basis_list : None or list of array, element i has
    shape=[n_components, n_supervoxels]
        each subject's reduced basis
        if None the basis will be generated on the fly

    n_components : int or None
        number of components

    Returns
    -------

    shared_response_list : list of array, element i has
    shape=[n_timeframes, n_components]
        shared response, element i is the shared response during session i

    """
    n_subjects, n_sessions = reduced_data_list.shape[:2]

    s = [None] * n_sessions

    # This is just to check that all subjects have same number of
    # timeframes in a given session
    for n in range(n_subjects):
        for m in range(n_sessions):
            data_nm = safe_load(reduced_data_list[n][m])
            n_timeframes, n_supervoxels = data_nm.shape

            if reduced_basis_list is None:
                reduced_basis_list = []
                for subject in range(n_subjects):
                    q = np.eye(n_components, n_supervoxels)
                    reduced_basis_list.append(q)

            basis_n = reduced_basis_list[n]
            if s[m] is None:
                s[m] = data_nm.dot(basis_n.T)
            else:
                s[m] = s[m] + data_nm.dot(basis_n.T)

    for m in range(n_sessions):
        s[m] = s[m] / float(n_subjects)

    return s


def _compute_and_save_corr_mat(img, shared_response, temp_dir):
    """computes correlation matrix and stores it

    Parameters
    ----------
    img : str
        path to data.
        Data are loaded with numpy.load and expected shape is
        [n_timeframes, n_voxels]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    shared_response : array, shape=[n_timeframes, n_components]
        shared response
    """
    data = safe_load(img).T
    name = safe_encode(img)
    path = os.path.join(temp_dir, "corr_mat_" + name)
    np.save(path, shared_response.T.dot(data))


def _compute_and_save_subject_basis(subject_number, sessions, temp_dir):
    """computes correlation matrix for all sessions

    Parameters
    ----------

    subject_number: int
        Number that identifies the subject. Basis will be stored in
         [temp_dir]/basis_[subject_number].npy

    sessions : array of str
        Element i of the array is a path to the data collected during
        session i.
        Data are loaded with numpy.load and expected shape is
        [n_timeframes, n_voxels]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance

    temp_dir : str or None
        path to dir where temporary results are stored
        if None temporary results will be stored in memory. This
        can results in memory errors when the number of subjects
        and / or sessions is large

    Returns
    -------

    basis: array, shape=[n_component, n_voxels] or str
        basis of subject [subject_number] or path to this basis
    """
    corr_mat = None
    for session in sessions:
        name = safe_encode(session)
        path = os.path.join(temp_dir, "corr_mat_" + name + ".npy")
        if corr_mat is None:
            corr_mat = np.load(path)
        else:
            corr_mat += np.load(path)
        os.remove(path)
    basis_i = _compute_subject_basis(corr_mat)
    path = os.path.join(temp_dir, "basis_%i" % subject_number)
    np.save(path, basis_i)
    return path + ".npy"


def _compute_subject_basis(corr_mat):
    """From correlation matrix between shared response and subject data,
    Finds subject's basis

    Parameters
    ----------

    corr_mat: array, shape=[n_component, n_voxels]
    or shape=[n_components, n_supervoxels]
        correlation matrix between shared response and subject data or
         subject reduced data
        element k, v is given by S.T.dot(X_i) where S is the shared response
         and X_i the data of subject i.

    Returns
    -------

    basis: array, shape=[n_components, n_voxels]
    or shape=[n_components, n_supervoxels]
        basis of subject or reduced_basis of subject
    """
    # The perturbation is only here to be
    # consistent with current implementation
    # of DetSRM.
    perturbation = np.zeros(corr_mat.shape)
    np.fill_diagonal(perturbation, 0.001)
    U, _, V = scipy.linalg.svd(corr_mat + perturbation, full_matrices=False)
    return U.dot(V)


def fast_srm(reduced_data_list,
             n_iter=10,
             n_components=None,
             low_ram=False,
             seed=0):
    """Computes shared response and basis in reduced space

    Parameters
    ----------

    reduced_data_list : array, shape=[n_subjects, n_sessions]
    or array, shape=[n_subjects, n_sessions, n_timeframes, n_supervoxels]
        Element i, j of the array is a path to the data of subject i
        collected during session j.
        Data are loaded with numpy.load and expected
        shape is [n_timeframes, n_supervoxels]
        or Element i, j of the array is the data in array of
        shape=[n_timeframes, n_supervoxels]
        n_timeframes and n_supervoxels are
         assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    n_iter : int
        Number of iterations performed

    n_components : int or None
        number of components

    Returns
    -------

    shared_response_list : list of array, element i has
     shape=[n_timeframes, n_components]
        shared response, element i is the shared response during session i
    """
    if low_ram:
        return lowram_srm(reduced_data_list, n_iter, n_components)
    else:
        # We need to switch data to DetSRM format
        # Indeed in DetSRM all sessions are concatenated.
        # Whereas in FastSRM multiple sessions are supported.

        n_subjects, n_sessions = reduced_data_list.shape[:2]
        # We store the correspondence between timeframes and session
        timeframes_slices = []
        current_j = 0
        for j in range(n_sessions):
            timeframes_slices.append(
                slice(current_j, current_j + len(reduced_data_list[0, j])))
            current_j += len(reduced_data_list[0][j])
        # Now we can concatenate everything
        X = [
            np.concatenate(reduced_data_list[i], axis=0).T
            for i in range(n_subjects)
        ]

        srm = DetSRM(n_iter=n_iter, features=n_components, rand_seed=seed)
        srm.fit(X)

        # SRM gives a list of data projected in shared space
        # we get the shared response by averaging those
        concatenated_s = np.mean(srm.transform(X), axis=0).T

        # Let us return the shared response sliced by sessions
        return [concatenated_s[i] for i in timeframes_slices]


def lowram_srm(reduced_data_list, n_iter=10, n_components=None):
    """Computes shared response and basis in reduced space

    Parameters
    ----------

    reduced_data_list : array of str, shape=[n_subjects, n_sessions]
        Element i, j of the array is a path to the data of subject i
        collected during session j.
        Data are loaded with numpy.load and expected
        shape is [n_timeframes, n_supervoxels]
        n_timeframes and n_supervoxels are
         assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    n_iter : int
        Number of iterations performed

    n_components : int or None
        number of components

    Returns
    -------

    shared_response_list : list of array, element i has
     shape=[n_timeframes, n_components]
        shared response, element i is the shared response during session i
    """

    n_subjects, n_sessions = reduced_data_list.shape[:2]
    shared_response = _reduced_space_compute_shared_response(
        reduced_data_list, None, n_components)

    reduced_basis = [None] * n_subjects
    for _ in range(n_iter):
        for n in range(n_subjects):
            cov = None
            for m in range(n_sessions):
                data_nm = np.load(reduced_data_list[n, m])
                if cov is None:
                    cov = shared_response[m].T.dot(data_nm)
                else:
                    cov += shared_response[m].T.dot(data_nm)
            reduced_basis[n] = _compute_subject_basis(cov)

        shared_response = _reduced_space_compute_shared_response(
            reduced_data_list, reduced_basis, n_components)

    return shared_response


def _compute_basis_subject_online(sessions, shared_response_list):
    """Computes subject's basis with shared response fixed

    Parameters
    ----------

    sessions : array of str
        Element i of the array is a path to the data
        collected during session i.
        Data are loaded with numpy.load and expected shape is
         [n_timeframes, n_voxels]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    shared_response_list : list of array, element i has
    shape=[n_timeframes, n_components]
        shared response, element i is the shared response during session i

    Returns
    -------

    basis: array, shape=[n_components, n_voxels]
        basis
    """

    basis_i = None
    i = 0
    for session in sessions:
        data = safe_load(session).T
        if basis_i is None:
            basis_i = shared_response_list[i].T.dot(data)
        else:
            basis_i += shared_response_list[i].T.dot(data)
        i += 1
        del data
    return _compute_subject_basis(basis_i)


def _compute_shared_response_online_single(subjects, basis_list, temp_dir,
                                           subjects_indexes, aggregate):
    """Computes shared response during one session with basis fixed

    Parameters
    ----------

    subjects : array of str
        Element i of the array is a path to the data of subject i.
        Data are loaded with numpy.load and expected shape is
         [n_timeframes, n_voxels]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    basis_list : None or list of array, element i has
    shape=[n_components, n_voxels]
        basis of all subjects, element i is the basis of subject i

    temp_dir : None or str
        path to basis folder where file basis_%i.npy contains the basis of
        subject i

    subjects_indexes : list of int or None
        list of indexes corresponding to the subjects to use to compute
        shared response

    aggregate: str or None, default="mean"
        if "mean": returns the mean shared response S from all subjects
        if None: returns the subject-specific response in shared space S_i

    Returns
    -------

    shared_response : array, shape=[n_timeframes, n_components] or list
        shared response
    """
    n = 0
    if aggregate == "mean":
        shared_response = None
    if aggregate is None:
        shared_response = []

    for k, i in enumerate(subjects_indexes):
        subject = subjects[k]
        # Transpose to be consistent with paper
        data = safe_load(subject).T
        if temp_dir is None:
            basis_i = basis_list[i]
        else:
            basis_i = np.load(os.path.join(temp_dir, "basis_%i.npy" % i))

        if aggregate == "mean":
            if shared_response is None:
                shared_response = data.dot(basis_i.T)
            else:
                shared_response += data.dot(basis_i.T)
            n += 1

        if aggregate is None:
            shared_response.append(data.dot(basis_i.T))

    if aggregate is None:
        return shared_response

    if aggregate == "mean":
        return shared_response / float(n)


def _compute_shared_response_online(imgs, basis_list, temp_dir, n_jobs,
                                    subjects_indexes, aggregate):
    """Computes shared response with basis fixed

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions]
        Element i, j of the array is a path to the data of subject i
        collected during session j.
        Data are loaded with numpy.load and expected shape is
        [n_timeframes, n_voxels]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

        imgs can also be a list of list of arrays where element i, j of
        the array is a numpy array of shape [n_voxels, n_timeframes] that
        contains the data of subject i collected during session j.

    basis_list : None or list of array, element i has
    shape=[n_components, n_voxels]
        basis of all subjects, element i is the basis of subject i

    temp_dir : None or str
        path to basis folder where file basis_%i.npy contains the basis of
        subject i

    n_jobs : integer, optional, default=1
            The number of CPUs to use to do the computation.
             -1 means all CPUs, -2 all CPUs but one, and so on.

    subjects_indexes : list or None
        list of indexes corresponding to the subjects to use to compute
        shared response

    aggregate: str or None, default="mean"
        if "mean": returns the mean shared response S from all subjects
        if None: returns the subject-specific response in shared space S_i

    Returns
    -------

    shared_response_list : list of array or list of list of array
        shared response, element i is the shared response during session i
        or element i, j is the shared response of subject i during session j
    """

    n_subjects = len(subjects_indexes)
    n_sessions = len(imgs[0])

    shared_response_list = Parallel(n_jobs=n_jobs)(
        delayed(_compute_shared_response_online_single)
        ([imgs[i][j] for i in range(n_subjects)], basis_list, temp_dir,
         subjects_indexes, aggregate) for j in range(n_sessions))

    if aggregate is None:
        shared_response_list = [[
            shared_response_list[j][i].T for j in range(n_sessions)
        ] for i in range(n_subjects)]

    if aggregate == "mean":
        shared_response_list = [
            shared_response_list[j].T for j in range(n_sessions)
        ]

    return shared_response_list


class FastSRM(BaseEstimator, TransformerMixin):
    """SRM decomposition using a very low amount of memory and \
computational power thanks to the use of an atlas \
as described in [Richard2019]_.

    Given multi-subject data, factorize it as a shared response S \
among all subjects and an orthogonal transform (basis) W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Parameters
    ----------

    atlas :  array, shape=[n_supervoxels, n_voxels] or array,\
shape=[n_voxels] or str or None, default=None
        Probabilistic or deterministic atlas on which to project the data. \
Deterministic atlas is an array of shape [n_voxels,] \
where values range from 1 \
to n_supervoxels. Voxels labelled 0 will be ignored. If atlas is a str the \
corresponding array is loaded with numpy.load and expected shape \
is (n_voxels,) for a deterministic atlas and \
(n_supervoxels, n_voxels) for a probabilistic atlas.

    n_components : int
        Number of timecourses of the shared coordinates

    n_iter : int
        Number of iterations to perform

    temp_dir : str or None
        Path to dir where temporary results are stored. If None \
temporary results will be stored in memory. This can results in memory \
errors when the number of subjects and/or sessions is large

    low_ram : bool
        If True and temp_dir is not None, reduced_data will be saved on \
disk. This increases the number of IO but reduces memory complexity when \
the number of subject and/or sessions is large

    seed : int
        Seed used for random sampling.

    n_jobs : int, optional, default=1
        The number of CPUs to use to do the computation. \
-1 means all CPUs, -2 all CPUs but one, and so on.

    verbose : bool or "warn"
        If True, logs are enabled. If False, logs are disabled. \
If "warn" only warnings are printed.

    aggregate: str or None, default="mean"
        If "mean", shared_response is the mean shared response \
from all subjects. If None, shared_response contains all \
subject-specific responses in shared space

    Attributes
    ----------

    `basis_list`: list of array, element i has \
shape=[n_components, n_voxels] or list of str
        - if basis is a list of array, element i is the basis of subject i
        - if basis is a list of str, element i is the path to the basis \
of subject i that is loaded with np.load yielding an array of \
shape [n_components, n_voxels].

        Note that any call to the clean method erases this attribute

    Note
    -----

        **References:**
        H. Richard, L. Martin, A. Pinho, J. Pillow, B. Thirion, 2019: \
Fast shared response model for fMRI data (https://arxiv.org/pdf/1909.12537.pdf)

    """
    def __init__(self,
                 atlas=None,
                 n_components=20,
                 n_iter=100,
                 temp_dir=None,
                 low_ram=False,
                 seed=None,
                 n_jobs=1,
                 verbose="warn",
                 aggregate="mean"):

        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_components = n_components
        self.n_iter = n_iter
        self.atlas = atlas

        if aggregate is not None and aggregate != "mean":
            raise ValueError("aggregate can have only value mean or None")

        self.aggregate = aggregate

        self.basis_list = None

        if temp_dir is None:
            if self.verbose == "warn" or self.verbose is True:
                logger.warning("temp_dir has value None. "
                               "All basis (spatial maps) and reconstructed "
                               "data will therefore be kept in memory."
                               "This can lead to memory errors when the "
                               "number of subjects "
                               "and/or sessions is large.")
            self.temp_dir = None
            self.low_ram = False

        if temp_dir is not None:
            self.temp_dir = os.path.join(temp_dir,
                                         "fastsrm" + str(uuid.uuid4()))
            self.low_ram = low_ram

    def clean(self):
        """This erases temporary files and basis_list attribute to \
free memory. This method should be called when fitted model \
is not needed anymore.
        """
        if self.temp_dir is not None:
            if os.path.exists(self.temp_dir):
                for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                os.rmdir(self.temp_dir)

        if self.basis_list is not None:
            self.basis_list is None

    def fit(self, imgs):
        """Computes basis across subjects from input imgs

        Parameters
        ----------
        imgs : array of str, shape=[n_subjects, n_sessions] or \
list of list of arrays or list of arrays
            Element i, j of the array is a path to the data of subject i \
collected during session j. Data are loaded with numpy.load and expected \
shape is [n_voxels, n_timeframes] n_timeframes and n_voxels are assumed \
to be the same across subjects n_timeframes can vary across sessions. \
Each voxel's timecourse is assumed to have mean 0 and variance 1

            imgs can also be a list of list of arrays where element i, j \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i collected during session j.

            imgs can also be a list of arrays where element i \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i (number of sessions is implicitly 1)

        Returns
        -------
        self : object
           Returns the instance itself. Contains attributes listed \
at the object level.
        """
        atlas_shape = check_atlas(self.atlas, self.n_components)
        reshaped_input, imgs, shapes = check_imgs(
            imgs, n_components=self.n_components, atlas_shape=atlas_shape)
        self.clean()
        create_temp_dir(self.temp_dir)

        if self.verbose is True:
            logger.info("[FastSRM.fit] Reducing data")

        reduced_data = reduce_data(imgs,
                                   atlas=self.atlas,
                                   n_jobs=self.n_jobs,
                                   low_ram=self.low_ram,
                                   temp_dir=self.temp_dir)

        if self.verbose is True:
            logger.info("[FastSRM.fit] Finds shared "
                        "response using reduced data")
        shared_response_list = fast_srm(reduced_data,
                                        n_iter=self.n_iter,
                                        n_components=self.n_components,
                                        low_ram=self.low_ram,
                                        seed=self.seed)
        if self.verbose is True:
            logger.info("[FastSRM.fit] Finds basis using "
                        "full data and shared response")

        if self.n_jobs == 1:
            basis = []
            for i, sessions in enumerate(imgs):
                basis_i = _compute_basis_subject_online(
                    sessions, shared_response_list)
                if self.temp_dir is None:
                    basis.append(basis_i)
                else:
                    path = os.path.join(self.temp_dir, "basis_%i" % i)
                    np.save(path, basis_i)
                    basis.append(path + ".npy")
                del basis_i
        else:
            if self.temp_dir is None:
                basis = Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_basis_subject_online)(
                        sessions, shared_response_list) for sessions in imgs)
            else:
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_and_save_corr_mat)(
                        imgs[i][j], shared_response_list[j], self.temp_dir)
                    for j in range(len(imgs[0])) for i in range(len(imgs)))

                basis = Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_and_save_subject_basis)(i, sessions,
                                                             self.temp_dir)
                    for i, sessions in enumerate(imgs))

        self.basis_list = basis
        return self

    def fit_transform(self, imgs, subjects_indexes=None):
        """Computes basis across subjects and shared response from input imgs
        return shared response.

        Parameters
        ----------
        imgs : array of str, shape=[n_subjects, n_sessions] or \
list of list of arrays or list of arrays
            Element i, j of the array is a path to the data of subject i \
collected during session j. Data are loaded with numpy.load and expected \
shape is [n_voxels, n_timeframes] n_timeframes and n_voxels are assumed \
to be the same across subjects n_timeframes can vary across sessions. \
Each voxel's timecourse is assumed to have mean 0 and variance 1

            imgs can also be a list of list of arrays where element i, j \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i collected during session j.

            imgs can also be a list of arrays where element i \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i (number of sessions is implicitly 1)

        subjects_indexes : list or None:
            if None imgs[i] will be transformed using basis_list[i]. \
Otherwise imgs[i] will be transformed using basis_list[subjects_index[i]]

        Returns
        --------
        shared_response : list of arrays, list of list of arrays or array
            - if imgs is a list of array and self.aggregate="mean": shared \
response is an array of shape (n_components, n_timeframes)
            - if imgs is a list of array and self.aggregate=None: shared \
response is a list of array, element i is the projection of data of \
subject i in shared space.
            - if imgs is an array or a list of list of array and \
self.aggregate="mean": shared response is a list of array, \
element j is the shared response during session j
            - if imgs is an array or a list of list of array and \
self.aggregate=None: shared response is a list of list of array, \
element i, j is the projection of data of subject i collected \
during session j in shared space.
        """
        self.fit(imgs)
        return self.transform(imgs, subjects_indexes=subjects_indexes)

    def transform(self, imgs, subjects_indexes=None):
        """From data in imgs and basis from training data,
        computes shared response.

        Parameters
        ----------
        imgs : array of str, shape=[n_subjects, n_sessions] or \
list of list of arrays or list of arrays
            Element i, j of the array is a path to the data of subject i \
collected during session j. Data are loaded with numpy.load and expected \
shape is [n_voxels, n_timeframes] n_timeframes and n_voxels are assumed \
to be the same across subjects n_timeframes can vary across sessions. \
Each voxel's timecourse is assumed to have mean 0 and variance 1

            imgs can also be a list of list of arrays where element i, j \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i collected during session j.

            imgs can also be a list of arrays where element i \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i (number of sessions is implicitly 1)

        subjects_indexes : list or None:
            if None imgs[i] will be transformed using basis_list[i]. \
Otherwise imgs[i] will be transformed using basis[subjects_index[i]]

        Returns
        --------
        shared_response : list of arrays, list of list of arrays or array
            - if imgs is a list of array and self.aggregate="mean": shared \
response is an array of shape (n_components, n_timeframes)
            - if imgs is a list of array and self.aggregate=None: shared \
response is a list of array, element i is the projection of data of \
subject i in shared space.
            - if imgs is an array or a list of list of array and \
self.aggregate="mean": shared response is a list of array, \
element j is the shared response during session j
            - if imgs is an array or a list of list of array and \
self.aggregate=None: shared response is a list of list of array, \
element i, j is the projection of data of subject i collected \
during session j in shared space.
         """
        aggregate = self.aggregate
        if self.basis_list is None:
            raise NotFittedError("The model fit has not been run yet.")

        atlas_shape = check_atlas(self.atlas, self.n_components)
        reshaped_input, imgs, shapes = check_imgs(
            imgs,
            n_components=self.n_components,
            atlas_shape=atlas_shape,
            ignore_nsubjects=True)
        check_indexes(subjects_indexes, "subjects_indexes")
        if subjects_indexes is None:
            subjects_indexes = np.arange(len(imgs))
        else:
            subjects_indexes = np.array(subjects_indexes)

        # Transform specific checks
        if len(subjects_indexes) < len(imgs):
            raise ValueError("Input data imgs has len %i whereas "
                             "subject_indexes has len %i. "
                             "The number of basis used to compute "
                             "the shared response should be equal "
                             "to the number of subjects in imgs" %
                             (len(imgs), len(subjects_indexes)))

        assert_valid_index(subjects_indexes, len(self.basis_list),
                           "subjects_indexes")

        shared_response = _compute_shared_response_online(
            imgs, self.basis_list, self.temp_dir, self.n_jobs,
            subjects_indexes, aggregate)

        # If shared response has only 1 session we need to reshape it
        if reshaped_input:
            if aggregate == "mean":
                shared_response = shared_response[0]
            if aggregate is None:
                shared_response = [
                    shared_response[i][0] for i in range(len(subjects_indexes))
                ]

        return shared_response

    def inverse_transform(
            self,
            shared_response,
            subjects_indexes=None,
            sessions_indexes=None,
    ):
        """From shared response and basis from training data
        reconstruct subject's data

        Parameters
        ----------

        shared_response : list of arrays, list of list of arrays or array
            - if imgs is a list of array and self.aggregate="mean": shared \
response is an array of shape (n_components, n_timeframes)
            - if imgs is a list of array and self.aggregate=None: shared \
response is a list of array, element i is the projection of data of \
subject i in shared space.
            - if imgs is an array or a list of list of array and \
self.aggregate="mean": shared response is a list of array, \
element j is the shared response during session j
            - if imgs is an array or a list of list of array and \
self.aggregate=None: shared response is a list of list of array, \
element i, j is the projection of data of subject i collected \
during session j in shared space.

        subjects_indexes : list or None
            if None reconstructs data of all subjects used during train. \
Otherwise reconstructs data of subjects specified by subjects_indexes.

        sessions_indexes : list or None
            if None reconstructs data of all sessions. \
Otherwise uses reconstructs data of sessions specified by sessions_indexes.

        Returns
        -------
        reconstructed_data: list of list of arrays or list of arrays
            - if reconstructed_data is a list of list : element i, j is \
the reconstructed data for subject subjects_indexes[i] and \
session sessions_indexes[j] as an np array of shape n_voxels, \
n_timeframes
            - if reconstructed_data is a list : element i is the \
reconstructed data for subject \
subject_indexes[i] as an np array of shape n_voxels, n_timeframes

        """
        added_session, shared = check_shared_response(
            shared_response, self.aggregate, n_components=self.n_components)
        n_subjects = len(self.basis_list)
        n_sessions = len(shared)

        for j in range(n_sessions):
            assert_array_2axis(shared[j], "shared_response[%i]" % j)

        check_indexes(subjects_indexes, "subjects_indexes")
        check_indexes(sessions_indexes, "sessions_indexes")

        if subjects_indexes is None:
            subjects_indexes = np.arange(n_subjects)
        else:
            subjects_indexes = np.array(subjects_indexes)

        assert_valid_index(subjects_indexes, n_subjects, "subjects_indexes")

        if sessions_indexes is None:
            sessions_indexes = np.arange(len(shared))
        else:
            sessions_indexes = np.array(sessions_indexes)

        assert_valid_index(sessions_indexes, n_sessions, "sessions_indexes")

        data = []
        for i in subjects_indexes:
            data_ = []
            basis_i = safe_load(self.basis_list[i])
            if added_session:
                data.append(basis_i.T.dot(shared[0]))
            else:
                for j in sessions_indexes:
                    data_.append(basis_i.T.dot(shared[j]))
                data.append(data_)
        return data

    def add_subjects(self, imgs, shared_response):
        """ Add subjects to the current fit. Each new basis will be \
appended at the end of the list of basis (which can \
be accessed using self.basis)

        Parameters
        ----------

        imgs : array of str, shape=[n_subjects, n_sessions] or \
list of list of arrays or list of arrays
            Element i, j of the array is a path to the data of subject i \
collected during session j. Data are loaded with numpy.load and expected \
shape is [n_voxels, n_timeframes] n_timeframes and n_voxels are assumed \
to be the same across subjects n_timeframes can vary across sessions. \
Each voxel's timecourse is assumed to have mean 0 and variance 1

            imgs can also be a list of list of arrays where element i, j \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i collected during session j.

            imgs can also be a list of arrays where element i \
of the array is a numpy array of shape [n_voxels, n_timeframes] \
that contains the data of subject i (number of sessions is implicitly 1)

        shared_response : list of arrays, list of list of arrays or array
            - if imgs is a list of array and self.aggregate="mean": shared \
response is an array of shape (n_components, n_timeframes)
            - if imgs is a list of array and self.aggregate=None: shared \
response is a list of array, element i is the projection of data of \
subject i in shared space.
            - if imgs is an array or a list of list of array and \
self.aggregate="mean": shared response is a list of array, \
element j is the shared response during session j
            - if imgs is an array or a list of list of array and \
self.aggregate=None: shared response is a list of list of array, \
element i, j is the projection of data of subject i collected \
during session j in shared space.
        """
        atlas_shape = check_atlas(self.atlas, self.n_components)
        reshaped_input, imgs, shapes = check_imgs(
            imgs,
            n_components=self.n_components,
            atlas_shape=atlas_shape,
            ignore_nsubjects=True)

        _, shared_response_list = check_shared_response(
            shared_response,
            n_components=self.n_components,
            aggregate=self.aggregate,
            input_shapes=shapes)

        # we need to transpose shared_response_list to be consistent with
        # other functions
        shared_response_list = [
            shared_response_list[j].T for j in range(len(shared_response_list))
        ]

        if self.n_jobs == 1:
            basis = []
            for i, sessions in enumerate(imgs):
                basis_i = _compute_basis_subject_online(
                    sessions, shared_response_list)
                if self.temp_dir is None:
                    basis.append(basis_i)
                else:
                    path = os.path.join(
                        self.temp_dir, "basis_%i" % (len(self.basis_list) + i))
                    np.save(path, basis_i)
                    basis.append(path + ".npy")
                del basis_i
        else:
            if self.temp_dir is None:
                basis = Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_basis_subject_online)(
                        sessions, shared_response_list) for sessions in imgs)
            else:
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_and_save_corr_mat)(
                        imgs[i][j], shared_response_list[j], self.temp_dir)
                    for j in range(len(imgs[0])) for i in range(len(imgs)))

                basis = Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_and_save_subject_basis)(
                        len(self.basis_list) + i, sessions, self.temp_dir)
                    for i, sessions in enumerate(imgs))

        self.basis_list += basis
