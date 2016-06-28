import numpy as np
import logging

"""
Some utility functions that can be used by different algorithms
"""


def from_tri_2_sym(tri, dim):
    """convert a upper triangular matrix in 1D format
       to 2D symmetric matrix


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
    """convert a 2D symmetric matrix to an upper
       triangular matrix in 1D format


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
    """to invert a 2D matrix
    Parameters
    ----------

    a : 2D array


    Returns
    -------

    inva: 2D array
         inverse of input matrix a


    Raises
    -------

    LinAlgError
        If a is singular or not square
    """
    if a.ndim != 2:
        raise TypeError("Input matrix should be 2D array")
    identity = np.identity(a.shape[1], dtype=a.dtype)
    inva = None
    try:
        inva = np.linalg.solve(a, identity)
        return inva
    except np.linalg.linalg.LinAlgError:
        logging.exception('Error from np.linalg.solve')
        raise
