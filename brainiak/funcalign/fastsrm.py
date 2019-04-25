"""Fast Shared Response Model (FastSRM)
"""

# Author: Hugo Richard (INRIA - Parietal)
# under the supervision of Jonathan Pillow (Princeton Neuroscience Institute)
# and Bertrand Thirion (Inria - Parietal)
# building upon code and work of Po-Hsuan Chen (Princeton Neuroscience
# Institute) and Javier Turek (Intel Labs)

import logging

import numpy as np
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed
import os
import glob
import hashlib

__all__ = [
    "FastSRM",
]

logger = logging.getLogger(__name__)


def reduce_data_single(img, atlas=None, inv_atlas=None, low_ram=False,
                       temp_dir=None):
    """Reduce data using given atlas

    Parameters
    ----------

    img : str
        path to data.
        Data are loaded with numpy.load and expected shape is
         [n_timeframes, n_voxels]
        n_timeframes and n_voxels are assumed to be the same across subjects
        n_timeframes can vary across sessions
        Each voxel's timecourse is assumed to have mean 0 and variance 1

    atlas :  array, shape=[n_supervoxels, n_voxels] or None
    or None or array, shape=[n_voxels]
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
    if atlas is None and inv_atlas is None:
        raise ValueError("An atlas or the pseudo inverse of"
                         " a probabilistic atlas should be provided")

    if inv_atlas is None and atlas is not None:
        n_voxels = atlas.shape[0]
        atlas_values = np.unique(atlas)
        if 0 in atlas_values:
            atlas_values = atlas_values[1:]
        data = np.load(img)

        if data.shape[1] != n_voxels:
            raise ValueError("%s have %i voxels and"
                             " the atlas has %i voxels."
                             "This is incompatible." % (img, data.shape[1],
                                                          n_voxels))

        reduced_data = np.array([np.mean(data[:, atlas == c], axis=1)
                                 for c in atlas_values]).T
    else:
        # this means that it is a probabilistic atlas
        assert len(inv_atlas.shape) == 2

        n_voxels = inv_atlas.shape[0]
        data = np.load(img)

        if data.shape[1] != n_voxels:
            raise ValueError("%s have %i voxels and the atlas"
                             " has %i voxels."
                             "This is incompatible." % (img, data.shape[1],
                                                          n_voxels))

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
    if type(atlas) != np.ndarray:
        raise ValueError("atlas should be of type np.ndarray but has type %s"
                         % (type(atlas)))

    if len(atlas.shape) == 2:
        A = None
        A_inv = atlas.T.dot(np.linalg.inv(atlas.dot(atlas.T)))
    else:
        if len(atlas.shape) != 1:
            raise ValueError("atlas should have shape of length 1 "
                             "(deterministic) or 2 (probabilistic) but input"
                             "atlas has shape of length %i" % len(atlas.shape))
        A = atlas
        A_inv = None

    n_subjects, n_sessions = imgs.shape

    reduced_data_list = Parallel(n_jobs=n_jobs)(
        delayed(reduce_data_single)(
            img,
            atlas=A,
            inv_atlas=A_inv,
            low_ram=low_ram,
            temp_dir=temp_dir
        ) for img in imgs.flatten())

    if low_ram:
        reduced_data_list = np.reshape(reduced_data_list,
                                       (n_subjects, n_sessions))
    else:
        if len(np.array(reduced_data_list).shape) == 1:
            reduced_data_list = np.reshape(reduced_data_list,
                                           (n_subjects, n_sessions))
        else:
            n_timeframes, n_supervoxels = np.array(reduced_data_list).shape[1:]
            reduced_data_list = np.reshape(reduced_data_list,
                                           (n_subjects,
                                            n_sessions,
                                            n_timeframes,
                                            n_supervoxels))

    return reduced_data_list


def check_shapes(n_supervoxels, n_components, n_timeframes):
    """
    Check assumptions about input parameters

    Parameters
    ----------

    n_supervoxels: int

    n_components: int

    n_timeframes: int

    """

    if n_supervoxels < n_components:
        raise ValueError("The number of regions in the atlas "
                         "%i is smaller than "
                         "the number of components %i of fastSRM"
                         % (n_supervoxels, n_components))

    if n_timeframes < n_components:
        raise ValueError("Number of timeframes %i is shorter than "
                         "number of components %i" % (n_timeframes,
                                                      n_components))


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
    elif (type(reduced_data) == str or
          type(reduced_data) == np.str_ or
          type(reduced_data) == np.str):
        low_ram = True
    else:
        raise ValueError("Reduced data are stored using "
                         "type %s which is neither np.ndarray or str"
                         % type(reduced_data))
    return low_ram


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
    list_n_timeframes = [None] * n_sessions
    for n in range(n_subjects):
        for m in range(n_sessions):
            if low_ram:
                data_nm = np.load(reduced_data_list[n, m])
            else:
                data_nm = reduced_data_list[n, m]

            n_timeframes, n_supervoxels = data_nm.shape

            check_shapes(n_supervoxels, n_components, n_timeframes)

            if list_n_timeframes[m] is None:
                list_n_timeframes[m] = n_timeframes
            elif list_n_timeframes[m] != n_timeframes:
                raise ValueError("Subject %i Session %i does not have the "
                                 "same number of timeframes "
                                 "as Subject %i Session %i" % (n, m, 0, m))

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


def fast_srm(reduced_data_list, n_iter=10, n_components=None):
    """Computes shared response and basis in reduced space

    Parameters
    ----------

    reduced_data_list : array of str, shape=[n_subjects, n_sessions]
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

    if type(reduced_data_list) != np.ndarray:
        raise ValueError("reduced data must have type np.ndarray but"
                         "has type %s" % type(reduced_data_list))

    low_ram = is_low_ram(reduced_data_list[0, 0])
    n_subjects, n_sessions = reduced_data_list.shape[:2]
    shared_response = _reduced_space_compute_shared_response(
        reduced_data_list,
        None,
        n_components
    )

    reduced_basis = [None] * n_subjects
    for _ in range(n_iter):
        for n in range(n_subjects):
            cov = None
            for m in range(n_sessions):
                if low_ram:
                    data_nm = np.load(reduced_data_list[n, m])
                else:
                    data_nm = reduced_data_list[n, m]
                if cov is None:
                    cov = shared_response[m].T.dot(data_nm)
                else:
                    cov += shared_response[m].T.dot(data_nm)
            reduced_basis[n] = _compute_subject_basis(cov)

        shared_response = _reduced_space_compute_shared_response(
            reduced_data_list,
            reduced_basis,
            n_components
        )

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


def _compute_shared_response_online_single(subjects, basis_list,
                                           temp_dir, subjects_indexes):
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
            subjects,
            basis_list,
            temp_dir,
            subjects_indexes
        ) for subjects in imgs.T)

    return shared_response_list


def check_imgs(imgs):
    """
    Check input images

    Parameters
    ----------

    imgs : array of str, shape=[n_subjects, n_sessions]
            Element i, j of the array is a path to the data of subject i
            collected during session j.
            Data are loaded with numpy.load and expected
            shape is [n_timeframes, n_voxels]
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1
    """
    if type(imgs) != np.ndarray:
        raise ValueError("imgs should be of type "
                         "np.ndarray but is of type %s"
                         % type(imgs))

    if len(imgs.shape) != 2:
        raise ValueError("imgs should be an array of shape "
                         "[n_subjects, n_sessions] "
                         "but its shape is of size %i"
                         % len(imgs.shape))

    n_subjects, n_sessions = imgs.shape

    if n_subjects <= 1:
        raise ValueError("The number of subjects should be greater than 1")


class FastSRM(BaseEstimator, TransformerMixin):
    """SRM decomposition using a very low amount of memory and
    computational power

    Given multi-subject data, factorize it as a shared response S among all
    subjects and an orthogonal transform (basis) W per subject:

    .. math:: X_i \\approx W_i S, \\forall i=1 \\dots N

    Parameters
    ----------

    atlas :  array, shape=[n_supervoxels, n_voxels] or array, shape=[n_voxels]
        Probabilistic or deterministic atlas on which to project the data
        Deterministic atlas is an array of shape [n_voxels,] where values
        range from 1 to n_supervoxels. Voxels labelled 0 will be ignored.

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

    `basis_list`: list of array, element i has shape=[n_components, n_voxels]
     or list of str
        basis of all subjects, element i is the basis of subject i
        or path to basis of all subjects, element i is the path to the
        basis of subject i
    """
    def __init__(self,
                 atlas,
                 n_components=20,
                 n_iter=100,
                 temp_dir=None,
                 low_ram=False,
                 random_state=None,
                 n_jobs=1,
                 verbose="warn",):

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
            if not os.path.exists(os.path.join(temp_dir, "fastsrm")):
                os.mkdir(os.path.join(temp_dir, "fastsrm"))
            self.temp_dir = os.path.join(temp_dir, "fastsrm")

            # Remove files in temp folder
            paths = glob.glob(os.path.join(self.temp_dir, "*.npy"))
            for path in paths:
                os.remove(path)

            self.low_ram = low_ram

    def fit(self, imgs):
        """Computes basis across subjects from input imgs

        Parameters
        ----------

        imgs : array of str, shape=[n_subjects, n_sessions]
            Element i, j of the array is a path to the data of subject i
            collected during session j.
            Data are loaded with numpy.load and expected
            shape is [n_timeframes, n_voxels]
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

        Returns
        -------
        self : object
           Returns the instance itself. Contains attributes listed
           at the object level.
        """

        if self.temp_dir is not None:
            # Remove former basis in temp folder
            paths = glob.glob(os.path.join(self.temp_dir, "*.npy"))
            for path in paths:
                os.remove(path)

        check_imgs(imgs)

        if self.verbose is True:
            n_subjects, n_sessions = imgs.shape
            logger.info("Fitting using %i subjects and %i sessions per subject"
                        % (n_subjects, n_sessions))
            logger.info("[FastSRM.fit] Reducing data")

        reduced_data = reduce_data(
            imgs,
            atlas=self.atlas,
            n_jobs=self.n_jobs,
            low_ram=self.low_ram,
            temp_dir=self.temp_dir
        )

        if self.verbose is True:
            logger.info("[FastSRM.fit] Finds shared "
                        "response using reduced data")

        shared_response_list = fast_srm(
            reduced_data,
            n_iter=self.n_iter,
            n_components=self.n_components,
        )

        if self.verbose is True:
            logger.info("[FastSRM.fit] Finds basis using "
                        "full data and shared response")

        if self.n_jobs == 1:
            basis = []
            for i, sessions in enumerate(imgs):
                basis_i = _compute_basis_subject_online(
                    sessions,
                    shared_response_list
                )
                if self.temp_dir is None:
                    basis.append(basis_i)
                else:
                    path = os.path.join(self.temp_dir, "basis_%i" % i)
                    np.save(path, basis_i)
                    basis.append(path + ".npy")
                del basis_i
        else:
            if self.temp_dir is None:
                basis = Parallel(
                    n_jobs=self.n_jobs
                )(delayed(
                    _compute_basis_subject_online
                )(
                    sessions,
                    shared_response_list
                ) for sessions in imgs)
            else:
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_and_save_corr_mat)(
                        subject,
                        shared_response_list[m],
                        self.temp_dir
                    )
                    for m, subjects in enumerate(imgs.T)
                    for subject in subjects
                )

                basis = Parallel(
                    n_jobs=self.n_jobs
                )(delayed(
                    _compute_and_save_subject_basis
                )(
                    i,
                    sessions,
                    self.temp_dir
                )
                  for i, sessions in enumerate(imgs)
                  )

        self.basis_list = basis
        return self

    def fit_transform(self, imgs, **fit_params):
        """Computes basis across subjects and shared response from input imgs
        return shared response.

        Parameters
        ----------
        imgs : array of str, shape=[n_subjects, n_sessions]
            Element i, j of the array is a path to the data of subject i
            collected during session j.
            Data are loaded with numpy.load and expected
            shape is [n_timeframes, n_voxels]
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1

        Returns
        --------
        shared_response_list : list of array, element i has
         shape=[n_timeframes, n_components]
            shared response, element i is the shared response during session i
        """
        self.fit(imgs)
        return self.transform(imgs)

    def transform(self, imgs, subjects_indexes=None):
        """From data in imgs and basis from training data,
        computes shared response.

        Parameters
        ----------

        imgs : array of str, shape=[n_subjects, n_sessions]
            Element i, j of the array is a path to the data of subject i
            collected during session j.
            Data are loaded with numpy.load and expected
            shape is [n_timeframes, n_voxels]
            n_timeframes and n_voxels are assumed to be the same across
            subjects
            n_timeframes can vary across sessions
            Each voxel's timecourse is assumed to have mean 0 and variance 1


        subjects_indexes : list or None:
            if None imgs[i] will be transformed using basis[i]
            otherwise imgs[i] will be transformed using
            basis[subjects_index[i]]

        Returns
        -------
        shared_response_list : list of array, element i has
        shape=[n_timeframes, n_components]
            shared response, element i is the shared response during session i
        """
        if self.basis_list is None:
            raise NotFittedError("The model fit has not been run yet.")

        if subjects_indexes is None:
            subjects_indexes = np.arange(len(imgs))
        else:
            subjects_indexes = np.array(subjects_indexes)

        shared_response = _compute_shared_response_online(
            imgs,
            self.basis_list,
            self.temp_dir,
            self.n_jobs,
            subjects_indexes
        )

        return shared_response

    def inverse_transform(self, shared_response_list, subjects_indexes=None,
                          sessions_indexes=None):
        """From shared response and basis from training data
        reconstruct subject's data

        Parameters
        ----------

        shared_response_list : list of array, element i has
        shape=[n_timeframes, n_components]
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
                basis_i = np.load(os.path.join(self.temp_dir,
                                               "basis_%i.npy" % i))

            for j in sessions_indexes:
                data_.append(shared_response_list[j].dot(basis_i))

            data.append(np.array(data_))
        return np.array(data)
