import numpy as np


"""
Some utility functions that can be used by different algorithms
"""


def from_tri_2_sym(tri, dim):
    """
    Parameters
    ----------

    tri: 1D array
        Contains elements of upper triangular matrix

    dim : int
        The dimension of target matrix.


    Returns
    -------

    symm : 2D array
        Symmetric matrix in shape=[dim, dim]
    """
    symm = np.zeros((dim, dim))
    symm[np.triu_indices(dim)] = tri
    return symm


def from_sym_2_tri(symm):
    """
    Parameters
    ----------

    symm : 2D array
          Symmetric matrix


    Returns
    -------

    tri: 1D array
          Contains elements of upper triangular matrix
    """

    inds = np.triu_indices_from(symm)
    tri = symm[inds]
    return tri


def fast_inv(a):
    """
    Parameters
    ----------

    a : 2D array


    Returns
    -------

    inva: 2D array
         inverse of input matrix a
    """

    identity = np.identity(a.shape[1], dtype=a.dtype)
    inva = np.linalg.solve(a, identity)
    return inva
