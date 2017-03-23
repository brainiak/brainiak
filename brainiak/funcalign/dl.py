#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Multi-subject Dictionary Learning

The implementation is based on the following publication:

.. [Varoquaux2011] "Multi-subject Dictionary Learning to Segment an Atlas of
   Brain Spontaneous Activity",
   G. Varoquaux, A. Gramfort, F. Pedregosa, V. Michel, .B. Thirion
   Information Processing in Medical Imaging: 22nd International Conference, IPMI 2011
   http://rd.springer.com/chapter/10.1007/978-3-642-22092-0_46

"""

# Authors: Javier Turek (Intel Labs), 2017

import logging

import numpy as np
import scipy
import scipy.sparse
from sklearn.decomposition import FastICA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import NotFittedError

__all__ = [
    "MSDL"
]

logger = logging.getLogger(__name__)


class MSDL(BaseEstimator, TransformerMixin):
    """Multi-subject Dictionary Learning

    Given multi-subject data, factorize it as spatial maps V_i and loadings
    U_i per subject:

    .. math:: X_i \\approx U_i V_i^T + E_i, \\forall i=1 \\dots N

    E_i is assumed to be white Gaussian Noise N(0,\\sigmaI)
    U_i is assumed to be Gaussian N(0,\\Sigma_U) (same covariance for all subjects)
    V_i = V + F_i, where F_i is Gaussian N(0,\\xsiI)
    V is a shared template across all subjects and it is assumed to have unit column norm


    Parameters
    ----------

    n_iter : int, default: 10
        Number of iterations to run the algorithm.

    factors : int, default: 10
        Number of factors to decompose the data.

    rand_seed : int, default: 0
        Seed for initializing the random number generator.

    mu : float, default: 1.0
        Value of the subject-global spatial map matching term.

    lam : float, default: 1.0
        Value of the regularization parameter.

    kappa : float, default: 0.5
        Step size for the FISTA algorithm. Should be a value in (0.0, 1.0).

    fista_iter : int, default: 20
        Number of iterations to run the FISTA algorithm to update the spatial map template.


    Attributes
    ----------

    Us_ : list of 2D arrays, element i has shape=[samples, factors]
        The loadings :math:`U_s` for each subject.

    Vs_ : list of 2D arrays, element i has shape=[voxels, factors]
        The spatial maps :math:`V_s` for each subject.

    V_ : 2D array, shape=[voxels, factors]
        The spatial map template :math:`V`.

    Note
    ----

        The number of voxels should be the same between subjects.

        The Multi-Subject Dictionary Learning is approximated using a
        Block Coordinate Descent (BCD) algorithm proposed in [Varoquaux2011]_.

        This is a single node version.

        The run-time complexity is :math:`O(S (V T K + (V + T) K^2 + K^3 + K I_{V}))` with
        I - the number of iterations of FISTA, V - the voxels from a subject,
        T - the number of time samples, K - the number of features/factors (typically, :math:`V \\gg T \\gg K`),
        and S - the number of subjects.
    """

    def __init__(self, n_iter=10, factors=10, rand_seed=0, mu=1.0, lam=1.0, kappa=0.5, fista_iter=20):
        self.n_iter = n_iter
        self.factors = factors
        self.rand_seed = rand_seed
        self.mu = mu
        self.lam = lam
        self.fista_iter = fista_iter
        self.kappa = kappa
        return

    def fit(self, X, y=None, R=None):
        """Compute the Multi-Subject Dictionary Learning decomposition

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used

        R : list of 2D arrays, element i has shape=[voxels, 3]
            Each row in the list contains the scanner coordinate of each voxel
            of fMRI data of all subjects.

        """
        logger.info('Starting MSDL')

        # Check the number of subjects
        if len(X) <= 1:
            raise ValueError("There are not enough subjects "
                             "({0:d}) to train the model.".format(len(X)))

        # Check for input data sizes
        if X[0].shape[1] < self.factors:
            raise ValueError(
                "There are not enough samples to train the model with "
                "{0:d} features.".format(self.factors))

        # Check if all subjects have same number of voxels
        number_voxels = X[0].shape[0]
        number_subjects = len(X)
        for subject in range(number_subjects):
            assert_all_finite(X[subject])
            if X[subject].shape[0] != number_voxels:
                raise ValueError("Different number of voxels between subjects"
                                 ".")

        # Check that we have the position for each voxel
        if R is None:
            raise TypeError("Cannot find parameter R.")
        if R.shape[0] != number_voxels:
            raise ValueError("Wrong number of voxels in R ({0:d})."
                             .format(len(X)))

        if self.kappa < 0.0 or self.kappa >= 1.0:
            raise ValueError("Kappa should be in range (0,1).")

        if self.lam <= 0.0 or self.mu <= 0.0:
            raise ValueError("Lambda and Mu regularization parameters should positive.")

        # Prepare the laplacian operator for this data
        self.L_ = self._create_laplacian_operator(R)
        self.max_eigval_L_ = scipy.sparse.linalg.svds(self.L_, k=1, which='LM', return_singular_vectors=False)
        self.max_eigval_L_ =  (self.max_eigval_L_[0]**2)
        logger.info("Lipschitz Constant of L ",self.max_eigval_L_)

        # Run MSDL
        self.Us_, self.Vs_, self.V_ = self._msdl(X)

        return self

    def transform(self, X, y=None):
        """Use the model to transform data to the low-dimentional subspace

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used

        Returns
        -------
        Us : list of 2D arrays, element i has shape=[factors, samples]
            Loadings for each subject new given data.
        """

        # Check if the model exist
        if hasattr(self, 'Vs_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.Vs_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        Us = [None] * len(X)
        Vu, Vsig, Vv = np.linalg.svd(self.V_, full_matrices=False)
        for subject in range(len(X)):
            U = X[subject].T.dot(Vu).dot(np.diag(Vsig / (Vsig**2).dot(Vv)))
            Us[subject] = self._update_us(X[subject], self.Vs_[subject], U)

        return Us

    @staticmethod
    def _create_laplacian_operator(R):
        """ Pre-computes the 3D-Laplacian operator

        Parameters
        ----------
        R : list of 2D arrays, element i has shape=[voxels, 3]
            Each row in the list contains the scanner coordinate of each voxel
            of fMRI data of all subjects.

        Returns
        -------
        2D sparse array, shape=[voxels,voxels]
            The laplacian matrix.
        """
        nmax = R.max(axis=0)+1
        voxels = R.shape[0]
        cube = -np.ones((nmax))
        cube[R[:,0],R[:,1],R[:,2]] = np.arange(voxels)

        data = np.zeros(7*voxels)
        row_ind = np.zeros(7*voxels)
        col_ind = np.zeros(7*voxels)
        kernel = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
        total_values = 0
        for v in range(voxels):
            offset = 0
            positions = kernel + R[v,:]
            for i in range(positions.shape[0]):
                if np.any(positions[i,:] < 0) or np.any(positions[i,:]>= nmax)\
                    or cube[positions[i,0],positions[i,1],positions[i,2]]<0:
                    continue
                data[total_values+offset] = -1.0
                row_ind[total_values+offset] = v
                col_ind[total_values+offset] = cube[positions[i,0],positions[i,1],positions[i,2]]
                offset += 1

            data[total_values+offset] = offset
            row_ind[total_values+offset] = v
            col_ind[total_values+offset] = v
            total_values += offset+1

        L = scipy.sparse.csr_matrix((data[:total_values], (row_ind[:total_values], col_ind[:total_values])), shape=(voxels, voxels))
        return L

    def _laplacian(self, x):
        """Computes the inner product with a 3D-laplacian operator.

        Parameters
        ----------
        x : array, shape=[voxels,] or [voxels, n]
            An array with one or more volumes represented as a column vectorized
            set of voxels.

        Returns
        -------
        array (same shape as x) with the result of applying the 3D laplacian
        operator on each column of x.
        """
        return self.L_.dot(x)

    def _objective_function(self, data, Us, Vs, V):
        """Calculate the objective function of MSDL

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.

        Us : list of 2D arrays, element i has shape=[samples, factors]
            The loadings :math:`U_s` for each subject.

        Vs : list of 2D arrays, element i has shape=[voxels, factors]
            The spatial maps :math:`V_s` for each subject.

        V : 2D array, shape=[voxels, factors]
            The template spatial map :math:`V`.

        Returns
        -------

        objective : float
            The objective function value.
        """
        subjects = len(data)
        objective = 0.0
        for s in range(subjects):
            objective += np.linalg.norm(data[s].T - Us[s].dot(Vs[s].T), 'fro')**2 + self.mu * np.linalg.norm(Vs[s] - V, 'fro')**2
        objective /= 2
        objective += self.lam * (np.sum(np.abs(V)) + 0.5 * np.sum(V * self._laplacian(V)))
        return objective

    @staticmethod
    def _update_us(data, Vs, Us):
        """Dictionary (loadings) update for a subject

        Parameters
        ----------
        data : array, shape=[voxels, samples]
            The fMRI data of subject s.

        Vs : array, shape=[voxels, factors]
            The spatial map :math:`V_s` for subject s.

        Us : array, shape=[samples, factors]
            The current loadings :math:`U_s` for subject s.

        Returns
        -------
        Us : array of shape=[samples, factors]
            The updated loadings :math:`U_s` for subject s.
        """
        factors = Vs.shape[1]
        A = Vs.T.dot(Vs)
        B = data.T.dot(Vs)
        for l in range(factors):
            dir = Us[:, l] + (B[:,l] - Us.dot(A[:, l])) / A[l,l]
            Us[:, l] = dir / np.amax((np.linalg.norm(dir, ord=2), 1.0))
        return Us

    def _update_vs(self, data, V, Us):
        """ Spatial map update

        Parameters
        ----------
        data : array, shape=[voxels, samples]
            The fMRI data of subject s.

        V : array, shape=[voxels, factors]
            The spatial map template :math:`V`.

        Us : array, shape=[samples, factors]
            The current loadings :math:`U_s` for subject s.

        Returns
        -------
        Vsi : array of shape=[voxels, factors]
            The updated spatial map :math:`V_s` for subject s.
        """
        factors = self.factors
        A = Us.T.dot(Us) + self.mu * np.eye(factors)
        Vsi = V + np.linalg.solve(A, Us.T.dot(data.T - Us.dot(V.T))).T
        return Vsi

    @staticmethod
    def _shrink(v, offset):
        """Computes soft shrinkage on the elements of an array

        Parameters
        ----------

        v : array
            An array with input values

        offset : float
            Offset for applying the shrinkage function


        Returns
        -------

        The array after applying the element-wise soft-thresholding function.
        """
        return np.sign(v) * np.max(np.abs(v) - offset, 0)

    def _prox(self, v):
        """Computes the proximal operator of a set of vectors

        Parameters
        ----------

        v : array, shape=[voxels,]  or [voxels,n]
            One or more column-vectors containing each a volume


        Returns
        -------

        v_star : (shape=same as input) with the proximal operator applied to the
        input vectors in v
        """
        v_star = v.copy()
        z = v_star
        tau = 1.0
        kappa = self.kappa/(1 + self.gamma * self.max_eigval_L_)
        obj_fun_prev = 0
        for l in range(self.fista_iter):
            v0 = v_star
            v_star = self._shrink(z - kappa * (z - v + self.gamma * self._laplacian(z)), kappa * self.gamma)

            # Check for convergence
            obj_fun = np.sum((v_star-v)**2,0) + self.gamma * (np.sum(np.abs(v_star),0) + 0.5 * np.sum(v * self._laplacian(v),0))
            if l > 0 and obj_fun > obj_fun_prev:
                return v0
            obj_fun_prev = obj_fun

            tau0 = tau
            tau = (1.0 + np.sqrt(1.0 + 4.0*tau0*tau0))/2.0
            z = v_star + (tau0-1)/tau*(v_star - v0)
        return v_star

    def _update_v(self, data, Vs):
        """ Spatial map template update

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.

        Vs : list of 2D arrays, element i has shape=[voxels, factors]
            The spatial maps :math:`V_s` for each subject.


        Returns
        -------

        V : 2D array, shape=[voxels, factors]
            The updated spatial map template :math:`V`.
        """
        subjects = len(data)
        meanVs = np.zeros(Vs[0].shape)
        for s in range(subjects):
            meanVs += Vs[s]
        meanVs /= subjects
        V = np.zeros(Vs[0].shape)
        for l in range(Vs[0].shape[1]):
            V[:,l] = self._prox(meanVs[:,l])
        return V

    def _msdl(self, data):
        """Expectation-Maximization algorithm for fitting the probabilistic SRM.

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.


        Returns
        -------

        Us : list of 2D arrays, element i has shape=[samples, factors]
            The loadings :math:`U_s` for each subject.

        Vs : list of 2D arrays, element i has shape=[voxels, factors]
            The spatial maps :math:`V_s` for each subject.

        V : 2D array, shape=[voxels, factors]
            The spatial map template :math:`V`.
        """
        subjects = len(data)
        np.random.seed(self.rand_seed)

        # Initialization step: initialize the outputs.
        Vs = [None] * subjects
        Us = [None] * subjects

        V = self._init_template(data, self.factors)

        Vu, Vsig, Vv = np.linalg.svd(V, full_matrices=False)
        for i in range(subjects):
            Vs[i] = V.copy()
            Us[i] = data[i].T.dot(Vu).dot(np.diag(Vsig / (Vsig**2)).dot(Vv))
            logger.info('SVD initialization %f ' % np.linalg.norm(np.linalg.solve(V.T.dot(V), V.T.dot(data[i])).T - Us[i]))

        if logger.isEnabledFor(logging.INFO):
            # Calculate the current objective function value
            objective = self._objective_function(data, Us, Vs, V)
            logger.info('Objective function %f' % objective)

        # Main loop of the algorithm
        for iteration in range(self.n_iter):
            logger.info('Iteration %d' % (iteration + 1))

            self.gamma = self.lam/self.mu/subjects
            logger.info('Gamma set to %f' % self.gamma)

            # Update each subject's decomposition:
            for i in range(subjects):
                logger.info('Subject %d' % (i))
                Us[i] = self._update_us(data[i], Vs[i], Us[i])
                logger.info('After Us update %d' % self._objective_function(data, Us, Vs, V))
                Vs[i] = self._update_vs(data[i], V, Us[i])
                logger.info('After Vs update %d' %self._objective_function(data, Us, Vs, V))

            # Update the spatial maps template:
            V = self._update_v(data, Vs)
            logger.info('After V update %d' %self._objective_function(data, Us, Vs, V))

            if logger.isEnabledFor(logging.INFO):
                # Calculate the current objective function value
                objective = self._objective_function(data, Us, Vs, V)
                logger.info('Objective function %f' % objective)

        return Us, Vs, V

    def _init_template(self, data, factors):
        """Initialize the template spatial map (V) for the MSDL with ICA.

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.

        factors : int
            The number of factors in the model.


        Returns
        -------

        V : array, shape=[voxels, factors]
            The initialized template spatial map :math:`V`.
        """
        subjects = len(data)
        voxels = data[0].shape[0]
        fica = FastICA(n_components=factors, whiten=True, max_iter=200, random_state=self.rand_seed)
        samples = 0
        for i in range(subjects):
            samples += data[i].shape[1]

        data_stacked = np.empty((voxels, samples))
        samples = 0
        for i in range(subjects):
            data_stacked[:, samples:(samples+data[i].shape[1])] = data[i]
            samples += data[i].shape[1]

        V = fica.fit_transform(data_stacked)

        return V
