"""Fast Shared Response Model (FastSRM)

The implementations are based on the following publications:

.. [Chen2015] "A Reduced-Dimension fMRI Shared Response Model",
   P.-H. Chen, J. Chen, Y. Yeshurun-Dishon, U. Hasson, J. Haxby, P. Ramadge
   Advances in Neural Information Processing Systems (NIPS), 2015.
   http://papers.nips.cc/paper/5855-a-reduced-dimension-fmri-shared-response-model

.. [Anderson2016] "Enabling Factor Analysis on Thousand-Subject Neuroimaging
   Datasets",
   Michael J. Anderson, Mihai Capotă, Javier S. Turek, Xia Zhu, Theodore L.
   Willke, Yida Wang, Po-Hsuan Chen, Jeremy R. Manning, Peter J. Ramadge,
   Kenneth A. Norman,
   IEEE International Conference on Big Data, 2016.
   https://doi.org/10.1109/BigData.2016.7840719

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


def is_low_ram(reduced_data):
    """
    Depending on type of reduced_data infer if we are in low-ram mode or not
    Parameters
    ----------
    reduced_data : str or array, shape=[n_timeframes, n_supervoxels]
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
    """
    if type(reduced_data) == np.ndarray:
        low_ram = False
    elif (type(reduced_data) == str or type(reduced_data) == np.str_
          or type(reduced_data) == np.str):
        low_ram = True
    else:
        raise ValueError("Reduced data are stored using "
                         "type %s which is neither np.ndarray or str" %
                         type(reduced_data))
    return low_ram


def safe_load(data):
    """If data is an array returns data else returns np.load(data)"""
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.load(data)
    return data


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
        raise ValueError("%s must have exactly 2 axes"
                         "but has %i axes" % (name_array, len(array.shape)))


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
            raise ValueError("imgs[%i] has type %s whereas \
                imgs[%i] has type %s. This is inconsistent." %
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
            raise ValueError("imgs[%i] has length %i whereas imgs[%i] \
            has length %i. All subjects should have the same number \
            of sessions." % (i, len(imgs[i]), 0, len(imgs[0])))

    shapes = np.zeros((n_subjects, n_sessions, 2))
    # Run array-level checks
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            assert_array_2axis(imgs[i, j], "imgs[%i, %i]" % (i, j))
            shapes[i, j, :] = imgs[i, j].shape

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
                    imgs[i, j], np.str_) or isinstance(imgs[i, j], np.str)):
                raise ValueError("imgs[i, j] is stored using "
                                 "type %s which is not a str" %
                                 type(imgs[i, j]))
            shapes[i, j, :] = get_shape(imgs[i, j])
    return shapes


def _check_shapes_atlas(n_components, n_voxels, atlas_shape):
    """Check if n_voxel in the atlas is consistent with number of voxels in
    the data, that number of supervoxels is lower than number of voxels
    but greater than number of components

    Parameters
    ----------
    n_components : int
    n_voxels : int
        number of voxels in the data
    atlas_shape: tuple"""
    if atlas_shape is not None:
        n_supervoxels, n_atlas_voxels = atlas_shape
        if n_atlas_voxels != n_voxels:
            raise ValueError("Number of voxels in the atlas is not the same \
            as the number of voxels in input data (imgs)")

        if n_supervoxels > n_voxels:
            raise ValueError("Number of regions in the atlas should be less \
            than the number of voxels")

        if n_components is not None:
            if n_supervoxels < n_components:
                raise ValueError("Number of regions in the atlas should \
                be bigger than the number of components")


def _check_shapes_components(n_components, n_timeframes):
    """Check that n_timeframes is greater than number of components"""
    if n_components is not None:
        if n_timeframes < n_components:
            raise ValueError("Number of timeframes %i is shorter than "
                             "number of components %i" %
                             (n_timeframes, n_components))


def _check_shapes(shapes, n_components=None, atlas_shape=None):
    """Check that number of voxels is the same for each subjects. Number of
    timeframes can vary between sessions but must be consistent across
    subjects

    Parameters
    ----------
    shapes : array of shape (n_subjects, n_sessions, 2)
        Array of shapes of input images
    """
    n_subjects, n_sessions, _ = shapes.shape

    if n_subjects <= 1:
        raise ValueError("The number of subjects should be greater than 1")

    n_timeframes_list = [None] * n_sessions
    n_voxels = None
    for n in range(n_subjects):
        for m in range(n_sessions):
            if n_timeframes_list[m] is None:
                n_timeframes_list[m] = shapes[m, n, 1]

            if n_voxels is None:
                n_voxels = shapes[m, n, 0]

            if n_timeframes_list[m] != shapes[m, n, 1]:
                raise ValueError("Subject %i Session %i does not have the "
                                 "same number of timeframes "
                                 "as Subject %i Session %i" % (n, m, 0, m))

            if n_voxels != shapes[m, n, 0]:
                raise ValueError("Subject %i Session %i"
                                 " does not have the same number of voxels as "
                                 "Subject %i Session %i." % (n, m, 0, 0))

    _check_shapes_components(n_components, np.sum(n_timeframes_list))
    _check_shapes_atlas(n_components, n_voxels, atlas_shape)


def check_atlas(atlas):
    """ Check input atlas

    Parameters
    ----------
    atlas :  array, shape=[n_supervoxels, n_voxels] or array, shape=[n_voxels]
     or str
        Probabilistic or deterministic atlas on which to project the data
        Deterministic atlas is an array of shape [n_voxels,] where values
        range from 1 to n_supervoxels. Voxels labelled 0 will be ignored.
        If atlas is a str the corresponding array is loaded with numpy.load
        and expected shape is (n_voxels,) for a deterministic atlas and
        (n_supervoxels, n_voxels) for a probabilistic atlas.

    Returns
    -------
    shape : array
        atlas shape
    """

    if not (isinstance(atlas, np.ndarray) or isinstance(atlas, str)
            or isinstance(atlas, np.str_) or isinstance(atlas, np.str)):
        raise ValueError("Atlas is stored using "
                         "type %s which is neither np.ndarray or str" %
                         type(atlas))

    if isinstance(atlas, np.ndarray):
        return atlas.shape
    else:
        shape = get_shape(atlas)
        if len(shape) == 1:
            # We have a deterministic atlas
            n_voxels = atlas.shape[0]
            n_supervoxels = len(np.unique(np.load(atlas))) - 1
            return (n_supervoxels, n_voxels)
        elif len(shape) == 2:
            return shape
        else:
            raise ValueError("Atlas has %i axes. It should have either 0 or 1 \
            axes." % len(shape))


def check_imgs(imgs, n_components=None, atlas_shape=None):
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
                "since imgs is a list it should be a list of list of array or \
                a list of array but imgs[0] as type %s" % type(imgs[0]))
    elif isinstance(imgs, np.ndarray):
        shapes = _check_imgs_array(imgs)
    else:
        raise ValueError(
            "imgs should either be a list of an array but has type" %
            type(imgs))

    _check_shapes(shapes, n_components, atlas_shape)

    return reshaped_input, new_imgs, shapes


def check_reduced_data(reduced_data_list,
                       n_components=None,
                       return_low_ram=False):

    if type(reduced_data_list) != np.ndarray:
        raise ValueError("reduced data must have type np.ndarray but"
                         "has type %s" % type(reduced_data_list))

    low_ram = is_low_ram(reduced_data_list[0, 0])
    n_subjects, n_sessions = reduced_data_list.shape[:2]

    # Let us check that reduced data have compatible shapes
    n_timeframes_list = [None] * n_sessions
    n_supervoxels = None
    for n in range(n_subjects):
        for m in range(n_sessions):
            data_nm = safe_load(reduced_data_list[n, m], low_ram)

            if n_timeframes_list[m] is None:
                n_timeframes_list[m] = data_nm.shape[0]

            if n_supervoxels is None:
                n_supervoxels = data_nm.shape[1]

            if n_timeframes_list[m] != data_nm.shape[0]:
                raise ValueError("Subject %i Session %i does not have the "
                                 "same number of timeframes "
                                 "as Subject %i Session %i" % (n, m, 0, m))

            if n_supervoxels != data_nm.shape[1]:
                raise ValueError(
                    "Reduced data from Subject %i Session %i"
                    " does not have the same number of supervoxels as "
                    "Subject %i Session %i." % (n, m, 0, 0))

    n_timeframes = np.sum(n_timeframes_list)
    if n_components is not None:
        check_n_components(n_supervoxels, n_components, n_timeframes)

    if return_low_ram:
        return low_ram


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
                         "by using the .clean() method")


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
    else:
        # this means that it is a probabilistic atlas
        reduced_data = data.dot(inv_atlas)

    if low_ram:
        name = hashlib.md5(img.encode()).hexdigest()
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
                                    imgs[i, j],
                                    atlas=A,
                                    inv_atlas=A_inv,
                                    low_ram=low_ram,
                                    temp_dir=temp_dir)
        for i in range(n_subjects) for j in range(n_sessions))

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
    low_ram = is_low_ram(reduced_data_list[0, 0])

    s = [None] * n_sessions

    # This is just to check that all subjects have same number of
    # timeframes in a given session
    for n in range(n_subjects):
        for m in range(n_sessions):
            if low_ram:
                data_nm = np.load(reduced_data_list[n, m])
            else:
                data_nm = reduced_data_list[n, m]

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
    data = np.load(img)
    name = hashlib.md5(img.encode()).hexdigest()
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
        name = hashlib.md5(session.encode()).hexdigest()
        path = os.path.join(temp_dir, "corr_mat_" + name + ".npy")
        if corr_mat is None:
            corr_mat = np.load(path)
        else:
            corr_mat += np.load(path)
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
    U, _, V = scipy.linalg.svd(corr_mat, full_matrices=False)
    return U.dot(V)


def fast_srm(reduced_data_list, n_iter=10, n_components=None, low_ram=False):
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
        n_subjects, n_sessions = reduced_data_list.shape[:2]
        # We store the correspondence between timeframes and session
        timeframes_slices = []
        current_j = 0
        for j in range(n_sessions):
            timeframes_slices.append(
                slice(current_j, current_j + len(reduced_data_list[0, j])))
            current_j = len(reduced_data_list[0][j])
        # Now we can concatenate everything
        X = [
            np.concatenate(reduced_data_list[i], axis=0).T
            for i in range(n_subjects)
        ]

        srm = DetSRM(n_iter=n_iter, features=n_components)
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
        data = np.load(session)
        if basis_i is None:
            basis_i = shared_response_list[i].T.dot(data)
        else:
            basis_i += shared_response_list[i].T.dot(data)
        i += 1
        del data
    return _compute_subject_basis(basis_i)


def _compute_shared_response_online_single(subjects, basis_list, temp_dir,
                                           subjects_indexes):
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

    Returns
    -------

    shared_response : array, shape=[n_timeframes, n_components]
        shared response
    """
    n = 0
    shared_response = None
    for k, i in enumerate(subjects_indexes):
        subject = subjects[k]
        data = np.load(subject)
        if temp_dir is None:
            basis_i = basis_list[i]
        else:
            basis_i = np.load(os.path.join(temp_dir, "basis_%i.npy" % i))

        if shared_response is None:
            shared_response = data.dot(basis_i.T)
        else:
            shared_response += data.dot(basis_i.T)

        n += 1
    return shared_response / float(n)


def _compute_shared_response_online(imgs, basis_list, temp_dir, n_jobs,
                                    subjects_indexes):
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

        imgs can also be a list of arrays where element i of the array is
        a numpy array of shape [n_voxels, n_timeframes] that contains the
        data of subject i (number of sessions is implicitly 1)

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

    Returns
    -------

    shared_response_list : list of array, element i has
    shape=[n_timeframes, n_components]
        shared response, element i is the shared response during session i
    """
    shared_response_list = Parallel(n_jobs=n_jobs)(
        delayed(_compute_shared_response_online_single)(
            subjects, basis_list, temp_dir, subjects_indexes)
        for subjects in imgs.T)

    return shared_response_list


class FastSRM(BaseEstimator, TransformerMixin):
    """SRM decomposition using a very low amount of memory and
    computational power

    Given multi-subject data, factorize it as a shared response S among all
    subjects and an orthogonal transform (basis) W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Parameters
    ----------

    atlas :  array, shape=[n_supervoxels, n_voxels] or array, shape=[n_voxels]
     or str
        Probabilistic or deterministic atlas on which to project the data
        Deterministic atlas is an array of shape [n_voxels,] where values
        range from 1 to n_supervoxels. Voxels labelled 0 will be ignored.
        If atlas is a str the corresponding array is loaded with numpy.load
        and expected shape is (n_voxels,) for a deterministic atlas and
        (n_supervoxels, n_voxels) for a probabilistic atlas.

    n_components : int
        Number of timecourses of the shared coordinates

    n_iter : int
        Number of iterations to perform

    temp_dir : str or None
        path to dir where temporary results are stored
        if None temporary results will be stored in memory. This
        can results in memory errors when the number of subjects
        and / or sessions is large

    low_ram : bool
        if True and temp_dir is not None, reduced_data will be saved on disk
        this increases the number of IO but reduces memory complexity when
        the number of subject and / or sessions is large

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    n_jobs : int, optional, default=1
        The number of CPUs to use to do the computation.
         -1 means all CPUs, -2 all CPUs but one, and so on.

    verbose : bool or "warn"
        if True, logs are enabled.
        if False, logs are disabled.
        if "warn" only warnings are printed.

    Attributes
    ----------

    `basis_list`: list of array, element i has shape=[n_voxels, n_components]
     or list of str
        - if basis is a list of array, element i is the basis of subject i
        - if basis is a list of str, element i is the path to the basis
            of subject i that is loaded with np.load yielding an array of
            shape [n_voxels, n_components].
        Note that any call to clean erases this attribute

    Notes
    -----
    **References:**
    H. Richard, L. Martin, A. Pinho, J. Pillow, B. Thirion, 2019: Fast
    shared response model for fMRI data (https://arxiv.org/pdf/1909.12537.pdf)
    """

    def __init__(
            self,
            atlas,
            n_components=20,
            n_iter=100,
            temp_dir=None,
            low_ram=False,
            random_state=None,
            n_jobs=1,
            verbose="warn",
    ):

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_components = n_components
        self.n_iter = n_iter
        self.atlas = atlas

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
            self.temp_dir = os.path.join(temp_dir, "fastsrm")
            self.low_ram = low_ram

    def clean(self):
        # TODO: Check that this does erase fastsrm file
        """This erases temporary files and basis_list attribute to free memory.
        This method should be called when fitted model is not needed anymore.
        """
        if self.temp_dir is not None:
            if os.path.exists(self.temp_dir):
                for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))

        if self.basis_list is not None:
            self.basis_list is None

    def fit(self, imgs):
        """Computes basis across subjects from input imgs

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
        self : object
           Returns the instance itself. Contains attributes listed
           at the object level.
        """
        atlas_shape = check_atlas(self.atlas)
        reshaped_input, imgs, shapes = check_imgs(
            imgs, n_components=self.n_components, atlas_shape=atlas_shape)
        self.clean()
        create_temp_dir(self.temp_dir)

        if self.verbose is True:
            n_subjects, n_sessions = imgs.shape
            logger.info(
                "Fitting using %i subjects and %i sessions per subject" %
                (n_subjects, n_sessions))
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
                                        low_ram=self.low_ram)

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
                        subject, shared_response_list[m], self.temp_dir)
                    for m, subjects in enumerate(imgs.T)
                    for subject in subjects)

                basis = Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_and_save_subject_basis)(i, sessions,
                                                             self.temp_dir)
                    for i, sessions in enumerate(imgs))

        self.basis_list = basis
        return self

    def fit_transform(self, imgs, subjects_indexes=None, aggregate="mean"):
        """Computes basis across subjects and shared response from input imgs
        return shared response.

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

        aggregate: str or None, default="mean"
            if "mean": returns the mean shared response S from all subjects
            if None: returns the subject-specific response in shared space S_i

        Returns
        --------
        shared_response_list : list of array, element i has
         shape=[n_timeframes, n_components]
            shared response, element i is the shared response during session i
        """
        self.fit(imgs)
        return self.transform(imgs,
                              subjects_indexes=subjects_indexes,
                              aggregate=aggregate)

    def transform(self, imgs, subjects_indexes=None, aggregate="mean"):
        """From data in imgs and basis from training data,
        computes shared response.

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

        subjects_indexes : list or None:
            if None imgs[i] will be transformed using basis[i]
            otherwise imgs[i] will be transformed using
            basis[subjects_index[i]]

        aggregate: str or None, default="mean"
            if "mean": returns the mean shared response S from all subjects
            if None: returns the subject-specific response in shared space S_i

        Returns
        -------
        shared_response_list : list of array, element i has
        shape=[n_components, n_timeframes]
            shared response, element i is the shared response during session i
        """
        if self.basis_list is None:
            raise NotFittedError("The model fit has not been run yet.")

        if subjects_indexes is None:
            subjects_indexes = np.arange(len(imgs))
        else:
            subjects_indexes = np.array(subjects_indexes)

        shared_response = _compute_shared_response_online(
            imgs, self.basis_list, self.temp_dir, self.n_jobs,
            subjects_indexes)

        return shared_response

    def inverse_transform(self,
                          shared_response_list,
                          subjects_indexes=None,
                          sessions_indexes=None):
        """From shared response and basis from training data
        reconstruct subject's data

        Parameters
        ----------

        shared_response_list : list of array, element i has
        shape=[n_components, n_timeframes]
            shared response, element i is the shared response during session i

        subjects_indexes : list or None
            if None reconstructs data of all subjects' used during train
            otherwise reconstructs data using subject's specified by
            subjects_indexes

        sessions_indexes : list or None
            if None reconstructs data using all sessions
            otherwise uses only specified sessions

        Returns
        -------
        reconstructed_data: array
        shape=[len(subjects_indexes), len(sessions_indexes),
        n_timeframes, n_voxels]
            Reconstructed data for chosen subjects and sessions
        """
        n_subjects = len(self.basis_list)

        if subjects_indexes is None:
            subjects_indexes = np.arange(n_subjects)
        else:
            subjects_indexes = np.array(subjects_indexes)

        if sessions_indexes is None:
            sessions_indexes = np.arange(len(shared_response_list))
        else:
            sessions_indexes = np.array(sessions_indexes)

        data = []
        for i in subjects_indexes:
            data_ = []
            if self.temp_dir is None:
                basis_i = self.basis_list[i]
            else:
                basis_i = np.load(
                    os.path.join(self.temp_dir, "basis_%i.npy" % i))

            for j in sessions_indexes:
                data_.append(shared_response_list[j].dot(basis_i))

            data.append(np.array(data_))
        return np.array(data)
