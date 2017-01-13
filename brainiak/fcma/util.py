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
    shape = data.shape
    data = zscore(data, axis=axis, ddof=0)
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    data = np.nan_to_num(data)
    data = data / math.sqrt(shape[axis])
    return data

def compute_correlation(component1, component2):
    [r1, d1] = component1.shape
    [r2, d2] = component2.shape
    if d1 != d2:
        raise ValueError('Dimension discrepancy')
    # preprocess two components
    component1 = normalize_for_correlation(component1, 1)
    component2 = normalize_for_correlation(component2, 1)
    corr_data = np.empty((r1, r2), dtype=np.float32, order='C')
    # blas routine is column-major
    blas.compute_single_matrix_multiplication('T', 'N',
                                              r1, r2, d1,
                                              1.0,
                                              component2, d2,
                                              component1, d1,
                                              0.0,
                                              corr_data,
                                              r2)
    return corr_data
