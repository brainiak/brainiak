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
"""Full Correlation Matrix Analysis (FCMA)

Correlation related high performance routines
"""

# Authors: Yida Wang
# (Intel Labs), 2017

import numpy as np
from . import cython_blas as blas
from scipy.stats.mstats import zscore
import math

def normalize_for_correlation(data, axis):
    """normalize the data before computing correlation

    The data will be z-scored and divided by sqrt(n)
    along the assigned axis

    Parameters
    ----------
    data: 2D array

    axis: int
        specify which dimension of the data should be normalized

    Returns
    -------
    data: 2D array
        the normalized data
    """
    shape = data.shape
    data = zscore(data, axis=axis, ddof=0)
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    data = np.nan_to_num(data)
    data = data / math.sqrt(shape[axis])
    return data

def compute_correlation(set1, set2):
    """compute correlation between two sets

    Correlate the rows of set1 with the rows of set2.
    If set1 == set2, it is auto-correlation computation
    resulting in a symmetric correlation matrix.
    The number of columns MUST agree between set1 and set2

    Parameters
    ----------
    set1: 2D array in shape [r1, c]
        MUST be continuous and row-major

    set2: 2D array in shape [r2, c]
        MUST be continuous and row-major

    Returns
    -------
    corr_data: 2D array in shape [r1, r2]
    """
    set1 = set1.astype(np.float32)
    set2 = set2.astype(np.float32)
    [r1, d1] = set1.shape
    [r2, d2] = set2.shape
    if d1 != d2:
        raise ValueError('Dimension discrepancy')
    # preprocess two components
    set1 = normalize_for_correlation(set1, 1)
    set2 = normalize_for_correlation(set2, 1)
    corr_data = np.empty((r1, r2), dtype=np.float32, order='C')
    # blas routine is column-major
    blas.compute_single_matrix_multiplication('T', 'N',
                                              r2, r1, d1,
                                              1.0,
                                              set2, d2,
                                              set1, d1,
                                              0.0,
                                              corr_data,
                                              r2)
    return corr_data
