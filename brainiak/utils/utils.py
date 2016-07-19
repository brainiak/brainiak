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
