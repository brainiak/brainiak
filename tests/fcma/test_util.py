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
from numpy.random import RandomState
from brainiak.fcma.util import compute_correlation

# specify the random state to fix the random numbers
prng = RandomState(1234567890)


def test_correlation_computation():
    row1 = 5
    col = 10
    row2 = 6
    mat1 = prng.rand(row1, col).astype(np.float32)
    mat2 = prng.rand(row2, col).astype(np.float32)
    corr = compute_correlation(mat1, mat1)
    expected_corr = np.corrcoef(mat1)
    assert np.allclose(corr, expected_corr, atol=1e-5), (
        "high performance correlation computation does not provide correct "
        "correlation results within the same set")
    corr = compute_correlation(mat1, mat2)
    mat = np.concatenate((mat1, mat2), axis=0)
    expected_corr = np.corrcoef(mat)[0:row1, row1:]
    assert np.allclose(corr, expected_corr, atol=1e-5), (
        "high performance correlation computation does not provide correct "
        "correlation results between two sets")


if __name__ == '__main__':
    test_correlation_computation()
