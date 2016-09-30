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

from brainiak.fcma.voxelselector import VoxelSelector
from scipy.stats.mstats import zscore
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from mpi4py import MPI
from numpy.random import RandomState

# specify the random state to fix the random numbers
prng = RandomState(1234567890)

def create_epoch():
    row = 12
    col = 5
    mat = prng.rand(row, col).astype(np.float32)
    mat = zscore(mat, axis=0, ddof=0)
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    mat = np.nan_to_num(mat)
    mat = mat / math.sqrt(mat.shape[0])
    return mat

def test_voxel_selection():
    fake_raw_data = [create_epoch() for i in range(8)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    # 2 subjects, 4 epochs per subject
    vs = VoxelSelector(fake_raw_data, 4, labels, 2, voxel_unit=1)
    # test scipy normalization
    fake_corr = prng.rand(1, 4, 5).astype(np.float32)
    fake_corr = vs._correlationNormalization(fake_corr)
    if MPI.COMM_WORLD.Get_rank() == 0:
        expected_fake_corr = [[[1.19203866, 0.18862808, -0.54350245,
                                -1.18334889, -0.16860008],
                               [-1.06594729, -1.08742261, -1.19447124,
                                1.14114654, -0.67860204],
                               [0.7839641, 1.53981364, 0.24948341,
                                0.82626557, 1.67902875],
                               [-0.91005552, -0.64101928, 1.48848987,
                                -0.78406328, -0.83182675]]]
        assert np.allclose(fake_corr, expected_fake_corr), \
            'within-subject normalization does not provide correct results'
    # for cross validation, use SVM with precomputed kernel
    # no shrinking, set C=1
    clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    results = vs.run(clf)
    if MPI.COMM_WORLD.Get_rank() == 0:
        output = [None] * len(results)
        for tuple in results:
            output[tuple[0]] = int(8*tuple[1])
        expected_output = [7, 4, 6, 4, 4]
        assert np.allclose(output, expected_output, atol=1), \
            'voxel selection via SVM does not provide correct results'
    # for cross validation, use logistic regression
    clf = LogisticRegression()
    results = vs.run(clf)
    if MPI.COMM_WORLD.Get_rank() == 0:
        output = [None] * len(results)
        for tuple in results:
            output[tuple[0]] = int(8*tuple[1])
        expected_output = [6, 3, 6, 4, 4]
        assert np.allclose(output, expected_output, atol=1), \
            'voxel selection via logistic regression does not provide correct results'

if __name__ == '__main__':
    test_voxel_selection()
