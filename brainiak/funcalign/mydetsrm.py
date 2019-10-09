import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from scipy.sparse import diags
import scipy


def hyperalign(X, Y, scale=False, primal=None):
    """
    Hyperalign X with Y using R and sc such that
    frobenius norm ||sc RX - Y||^2 is minimized and
    R is an orthogonal matrix
    sc is a scalar
    Parameters
    ----------
    X: (n_features, n_timeframes) nd array
        source data
    Y: (n_features, n_timeframes) nd array
        target data
    scale: bool
        If scale is true, computes a floating scaling parameter sc such that:
        ||sc * RX - Y||^2 is minimized and
        - R is an orthogonal matrix
        - sc is a scalar
        If scale is false sc is set to 1
    primal: bool or None, optional,
         Whether the SVD is done on the YX^T (primal) or Y^TX (dual)
         if None primal is used iff n_features <= n_timeframes

    Returns
    ----------
    R: (n_features, n_features) nd array
        transformation matrix
    sc: int
        scaling parameter
    """
    if np.linalg.norm(X) == 0 or np.linalg.norm(Y) == 0:
        return diags(np.ones(X.shape[1])).tocsr(), 1

    if primal is None:
        primal = X.shape[1] >= X.shape[0]

    if primal:
        A = Y.dot(X.T)
        if A.shape[0] == A.shape[1]:
            A += +1.e-18 * np.eye(A.shape[0])
        U, s, V = scipy.linalg.svd(A, full_matrices=0)
        R = U.dot(V)
    else:  # "dual" mode
        Uy, sy, Vy = scipy.linalg.svd(Y, full_matrices=0)
        Ux, sx, Vx = scipy.linalg.svd(X, full_matrices=0)
        A = np.diag(sy).dot(Vy).dot(Vx.T).dot(np.diag(sx))
        U, s, V = scipy.linalg.svd(A)
        R = Uy.dot(U).dot(V).dot(Ux.T)
    """
    if X.shape[0] > 10000:
        R = diags(np.ones(X.shape[0])).tocsr()
        s = np.sum(Y * X, 1)
    """
    if scale:
        sc = s.sum() / (np.linalg.norm(X)**2)
    else:
        sc = 1
    return R, sc


def create_orthogonal_matrix(rows, cols, random_state=None):
    """
    Creates matrix W with orthogonal columns:
    W.T.dot(W) = I
    Parameters
    ----------
    rows: int
        number of rows
    cols: int
        number of columns
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.
    Returns
    ---------
    Matrix W of shape (rows, cols) such that W.T.dot(W) = np.eye(cols)
    """
    v = rows
    k = cols
    if random_state is None:
        rnd_matrix = np.random.rand(v, k)
    else:
        rnd_matrix = random_state.rand(v, k)
    q, r = np.linalg.qr(rnd_matrix)
    return q


def _compute_shared_response(compressed_data, basis, scale):
    """
    Computes the shared response S using subject basis and scaling
    the basis refers to sc_i * W_i
    the scale refers to sc_i
    """
    s = None
    for m in range(len(basis)):
        data_m = compressed_data[m]
        if s is None:
            s = basis[m].T.dot(data_m)
        else:
            s = s + basis[m].T.dot(data_m)
    s /= np.sum(scale**2)
    return s


def fast_srm(reduced_data,
             random_state=None,
             max_iter=10,
             tol=1e-6,
             use_scaling=False,
             n_components=None):
    """
    Computes shared response and basis in reduced space
    the basis refers to sc_i * W_i
    the scale refers to sc_i

    Parameters
    ----------
    reduced_data: list of n_subjects np array of shape n_voxels, n_timeframes
        The reduced data
    random_state: RandomState
    max_iter: int
    tol: int
    use_scaling: bool
        If True the scaling procedure is used
    n_components: int or None
        number of components if n_voxels != n_components
    Returns
    -------
    scale: np array of shape n_subjects
    shared_response: np array of shape n_components, n_timeframes
    basis: list of n_subjects arrays of shape n_voxels, n_components
    """

    n_subjects = len(reduced_data)
    basis = []
    scale = []
    random_state = check_random_state(random_state)
    for subject in range(n_subjects):
        n_voxels, n_timeframes = reduced_data[subject].shape
        if n_components is None:
            n_components = n_voxels
        q = create_orthogonal_matrix(n_voxels,
                                     n_components,
                                     random_state=random_state)
        basis.append(q)
        scale.append(1.)
    scale = np.array(scale)

    shared_response = _compute_shared_response(reduced_data, basis, scale)
    for n_iter in range(max_iter):
        for i in range(n_subjects):
            X_i = reduced_data[i]
            R, sc = hyperalign(shared_response, X_i, scale=use_scaling)
            basis[i] = sc * R

        shared_response = _compute_shared_response(reduced_data, basis, scale)

        if np.sum([
                np.linalg.norm(reduced_data[i] - basis[i].dot(shared_response))
                for i in range(n_subjects)
        ],
                  axis=0) < tol:
            break

    return scale, basis, shared_response


class MyDetSRM(BaseEstimator, TransformerMixin):
    """My Deterministic Shared Response Model (DetSRM)
    """
    def __init__(self, n_iter=10, features=50, rand_seed=0):
        self.n_iter = n_iter
        self.features = features
        self.rand_seed = rand_seed
        return

    def fit(self, X, y=None):
        """Compute the Deterministic Shared Response Model

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used
        """
        _, self.w_, self.s_ = fast_srm(X,
                                       self.rand_seed,
                                       max_iter=self.n_iter,
                                       n_components=self.features)

        return self

    def transform(self, X, y=None):
        """Use the model to transform data to the Shared Response subspace

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels_i, samples_i]
            Each element in the list contains the fMRI data of one subject.

        y : not used


        Returns
        -------
        s : list of 2D arrays, element i has shape=[features_i, samples_i]
            Shared responses from input data (X)
        """

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        s = [None] * len(X)
        for subject in range(len(X)):
            s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    def _objective_function(self, data, w, s):
        """Calculate the objective function

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        w : list of 2D arrays, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        s : array, shape=[features, samples]
            The shared response

        Returns
        -------

        objective : float
            The objective function value.
        """
        subjects = len(data)
        objective = 0.0
        for m in range(subjects):
            objective += \
                np.linalg.norm(data[m] - w[m].dot(s), 'fro') ** 2

        return objective * 0.5 / data[0].shape[1]

    def _compute_shared_response(self, data, w):
        """ Compute the shared response S

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.

        w : list of 2D arrays, element i has shape=[voxels_i, features]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        Returns
        -------

        s : array, shape=[features, samples]
            The shared response for the subjects data with the mappings in w.
        """
        s = np.zeros((w[0].shape[1], data[0].shape[1]))
        for m in range(len(w)):
            s = s + w[m].T.dot(data[m])
        s /= len(w)

        return s

    @staticmethod
    def _update_transform_subject(Xi, S):
        """Updates the mappings `W_i` for one subject.

        Parameters
        ----------

        Xi : array, shape=[voxels, timepoints]
            The fMRI data :math:`X_i` for aligning the subject.

        S : array, shape=[features, timepoints]
            The shared response.

        Returns
        -------

        Wi : array, shape=[voxels, features]
            The orthogonal transform (mapping) :math:`W_i` for the subject.
        """
        A = Xi.dot(S.T)
        # Solve the Procrustes problem
        U, _, V = np.linalg.svd(A, full_matrices=False)
        return U.dot(V)

    def transform_subject(self, X):
        """Transform a new subject using the existing model.
        The subject is assumed to have recieved equivalent stimulation

        Parameters
        ----------

        X : 2D array, shape=[voxels, timepoints]
            The fMRI data of the new subject.

        Returns
        -------

        w : 2D array, shape=[voxels, features]
            Orthogonal mapping `W_{new}` for new subject
        """

        # Check the number of TRs in the subject
        if X.shape[1] != self.s_.shape[1]:
            raise ValueError("The number of timepoints(TRs) does not match the"
                             "one in the model.")

        w = self._update_transform_subject(X, self.s_)

        return w
