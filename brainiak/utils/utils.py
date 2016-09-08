#  Copyright 2016 Intel Corporation, Princeton University
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
import re
import warnings

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


class read_design:
    """ A class which has the ability of reading in design matrix in .1D file,
        generated by AFNI's 3dDeconvolve.
        In future, we will make it read SPM's design matrix file as well.
    """
    def __init__(self, fname=None, include_orth=False, include_pols=False):
        if fname is None:
            # fname is the name of the file to read in the design matrix
            self.design = np.zeros([0, 0])
            self.n_col = 0
            # number of columns (conditions) in the design matrix
            self.column_types = np.ones(0)
            self.n_basis = 0
            self.n_stim = 0
            self.n_orth = 0
            self.StimLabels = []
        else:
            isAFNI = re.match(r'.+[.](1D|1d|txt)$', fname)
            # We assume all AFNI 1D files have extension of 1D or 1d or txt
            if isAFNI:
                self.readAFNI(fname=fname)

        self.include_orth = include_orth
        self.include_pols = include_pols
        self.cols_used = np.where(self.column_types == 1)[0]
        if self.include_orth:
            self.cols_used = np.sort(
                np.append(self.cols_used, np.where(self.column_types == 0)[0]))
        if self.include_pols:
            self.cols_used = np.sort(np.append(
                self.cols_used, np.where(self.column_types == -1)[0]))
        self.design_used = self.design[:, self.cols_used]
        if not self.include_pols:
            # baseline is not included, then we add a column of all 1's
            self.design_used = np.insert(self.design_used, 0, 1, axis=1)
        self.n_TR = np.size(self.design_used, axis=0)

    def readAFNI(self, fname):
        # Read design file written by AFNI
        self.n_basis = 0
        self.n_stim = 0
        self.n_orth = 0
        self.StimLabels = []
        self.design = np.loadtxt(fname, ndmin=2)
        with open(fname) as f:
            all_text = f.read()

        find_n_column = re.compile(
            r'^#[ ]+ni_type[ ]+=[ ]+"(?P<n_col>\d+)[*]', re.MULTILINE)
        n_col_found = find_n_column.search(all_text)
        if n_col_found:
            self.n_col = int(n_col_found.group('n_col'))
            if self.n_col != np.size(self.design, axis=1):
                warnings.warn(
                    'The number of columns in the design matrix'
                    + 'does not match the header information')
                self.n_col = np.size(self.design, axis=1)
        else:
            self.n_col = np.size(self.design, axis=1)

        self.column_types = np.ones(self.n_col)
        # default that all columns are conditions of interest

        find_ColumnGroups = re.compile(
            r'^#[ ]+ColumnGroups[ ]+=[ ]+"(?P<CGtext>.+)"', re.MULTILINE)
        CG_found = find_ColumnGroups.search(all_text)
        if CG_found:
            CG_text = re.split(',', CG_found.group('CGtext'))
            curr_idx = 0
            for CG in CG_text:
                split_by_at = re.split('@', CG)
                if len(split_by_at) == 2:
                    # the first tells the number of columns in this condition
                    # the second tells the condition type
                    n_this_cond = int(split_by_at[0])
                    self.column_types[curr_idx:curr_idx + n_this_cond] = \
                        int(split_by_at[1])
                    curr_idx += n_this_cond
                elif len(split_by_at) == 1 and \
                        not re.search('..', split_by_at[0]):
                    # Just a number, and not the type like '1..4'
                    self.column_types[curr_idx] = int(split_by_at[0])
                    curr_idx += 1
                else:  # must be a single stimulus condition
                    split_by_dots = re.split('\..', CG)
                    n_this_cond = int(split_by_dots[1])
                    self.column_types[curr_idx:curr_idx + n_this_cond] = 1
                    curr_idx += n_this_cond
            self.n_basis = np.sum(self.column_types == -1)
            self.n_stim = np.sum(self.column_types > 0)
            self.n_orth = np.sum(self.column_types == 0)

        find_StimLabels = re.compile(
            r'^#[ ]+StimLabels[ ]+=[ ]+"(?P<SLtext>.+)"', re.MULTILINE)
        StimLabels_found = find_StimLabels.search(all_text)
        if StimLabels_found:
            self.StimLabels = \
                re.split(r'[ ;]+', StimLabels_found.group('SLtext'))
        else:
            self.StimLabels = []
