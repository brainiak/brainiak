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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import NotFittedError

__all__ = [
    "MDL"
]

logger = logging.getLogger(__name__)

def _init_template(data, factors):
    """Initialize the template spatial map (V) for the MSDL with random orthogonal matrices.

    Parameters
    ----------

    data : list of 2D arrays, element i has shape=[voxels, samples]
        Each element in the list contains the fMRI data of one subject.

    factors : int
        The number of factors in the model.


    Returns
    -------

    v : array, shape=[voxels, factors]
        The initialized template spatial map :math:`V`.

    Note
    ----

        This function assumes that the numpy random number generator was
        initialized.

        Not thread safe.
    """
    #v = np.empty(data[0].shape[0], factors)
    #subjects = len(data)
    V = np.random.random((data[0].shape[0], factors))
    # Set Wi to a random orthogonal voxels by features matrix
    # for subject in range(subjects):
    #     voxels[subject] = data[subject].shape[0]
    #     rnd_matrix = np.random.random((voxels[subject], features))
    #     q, r = np.linalg.qr(rnd_matrix)
    #     w.append(q)

    return V


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


    Attributes
    ----------

    w_ : list of array, element i has shape=[voxels_i, features]
        The orthogonal transforms (mappings) for each subject.

    s_ : array, shape=[features, samples]
        The shared response.

    Note
    ----

        The number of voxels should be the same between subjects. #TODO: However, the
        number of samples may be different across subjects.

        The Multi-Subject Dictionary Learning is approximated using a
        Block Coordinate Descent (BCD) algorithm proposed in [Varoquaux2011]_.

        This is a single node version.

        TODO: The run-time complexity is :math:`O(I (V T K + V K^2))` and the memory
        complexity is :math:`O(V T)` with I - the number of iterations, V - the
        sum of voxels from all subjects, T - the number of samples, K - the
        number of features (typically, :math:`V \\gg T \\gg K`), and N - the
        number of subjects.
    """

    def __init__(self, n_iter=10, factors=10, rand_seed=0, mu=1.0, lam=1.0, gamma=1.0, fista_iter=100):
        self.n_iter = n_iter
        self.factors = factors
        self.rand_seed = rand_seed
        self.mu = mu
        self.lam = lam
        self.gamma = gamma
        self.fista_iter = fista_iter
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

        # Prepare the laplacian operator for this data
        self.L_ = self._create_laplacian_operator(R)
        self.max_singularval_L_ = scipy.sparse.linalg.svds(self.L_, 1, return_singular_vectors=False)
        self.max_singularval_L_ = self.max_singularval_L_[0]
        print(self.max_singularval_L_)

        # Run MSDL
        self.Us_, self.Vs_, self.V_ = self._msdl(X)

        return self

    def transform(self, X, y=None):
        """Use the model to transform data to the Shared Response subspace

        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.

        y : not used

        Returns
        -------
        s : list of 2D arrays, element i has shape=[features_i, samples_i]
            Shared responses from input data (X)
        """

        # Check if the model exist
        if hasattr(self, 'w_') is False:
            raise NotFittedError("The model fit has not been run yet.")

        # Check the number of subjects
        if len(X) != len(self.w_):
            raise ValueError("The number of subjects does not match the one"
                             " in the model.")

        s = [None] * len(X)
        for subject in range(len(X)):
            s[subject] = self.w_[subject].T.dot(X[subject])

        return s

    def _create_laplacian_operator(self, R):
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
                data[total_values+offset] = 1.0
                row_ind[total_values+offset] = v
                col_ind[total_values+offset] = cube[positions[i,0],positions[i,1],positions[i,2]]
                offset += 1

            data[total_values+offset] = -offset
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

    def _update_us(self, data, Vs, Us):
        factors = Vs.shape[1]
        #TODO: we should make sure that we use the updated atoms to update the dictionary
        # Need to Check Online dictionary learning work.
        v_2_squared = np.sum(Vs**2, axis=0)
        for l in range(factors):
            #TODO: there was an error in the original paper
            #TODO: we need to check with the original OnlineDL paper from Julien Mairal
            Us[:,l] += (data.T.dot(Vs[:,l]) - Us.dot(Vs.T.dot(Vs[:,l])))/v_2_squared[l]
        return Us

    def _update_vs(self, data, V, Us):
        factors = self.factors
        A = Us.T.dot(Us) + self.mu * np.eye(factors)
        Vsi = V + np.linalg.solve(A, Us.T.dot(data.T - Us.dot(V.T))).T
        return Vsi

    def _update_v(self, data, Vs):
        subjects = len(data)
        meanVs = np.zeros(Vs[0].shape)
        for s in range(subjects):
            meanVs += Vs[s]
        #TODO: need to pass the parameters for the proximal operator
        return self._prox(meanVs/subjects)

    def _shrink(self, v, offset):
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
        v* (shape=same as input) with the proximal operator applied to the
        input vectors in v

        """
        v_star = v
        z = v_star
        tau = 1.0
        kappa = 0.2/(1 + self.gamma * self.max_singularval_L_)
        k = 10
        for l in range(self.fista_iter):
            v0 = v_star
            v_star = self._shrink(z - kappa*(z - v + self.gamma * self._laplacian(z)), kappa*self.gamma)
            tau0 = tau
            tau = (1 + np.sqrt(1 + 4*tau*tau))/2
            z = v_star + (tau0-1)/tau*(v_star - v0)
        return v_star

    def _msdl(self, data):
        """Expectation-Maximization algorithm for fitting the probabilistic SRM.

        Parameters
        ----------

        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.


        Returns
        -------

        Us : list of array, element i has shape=[voxels, factors]
            The orthogonal transforms (mappings) :math:`W_i` for each subject.

        Vs : list of array, element i has shape=[factors, samples]
            The shared response :math:`V_i` for each subject.

        V  : array, shape=[factors, samples]
            The spatial map template.
        """

        subjects = len(data)
        print(subjects)
        np.random.seed(self.rand_seed)

        # Initialization step: initialize the outputs.
        Vs = [None] * subjects
        Us = [None] * subjects

        V = _init_template(data, self.factors)
        print(V.shape)
        print(data[0].shape)

        Vu, Vsig, Vv = np.linalg.svd(V, full_matrices=False)
        print(Vu.shape)
        print(Vsig.shape)
        print(Vv.shape)
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
            print(iteration)
            print(self._objective_function(data, Us, Vs, V))

            # Update each subject's decomposition:
            for i in range(subjects):
                Us[i] = self._update_us(data[i], Vs[i], Us[i])
                Vs[i] = self._update_vs(data[i], V, Us[i])

            # Update the spatial maps template:
            V = self._update_v(data, Vs)

            if logger.isEnabledFor(logging.INFO):
                # Calculate the current objective function value
                objective = self._objective_function(data, Us, Vs, V)
                logger.info('Objective function %f' % objective)

        return Us, Vs, V
