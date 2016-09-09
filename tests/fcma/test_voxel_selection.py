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
import numpy as np
import math
from mpi4py import MPI
import time


def create_epoch():
    row = 12
    col = 1000
    mat = np.random.rand(row, col).astype(np.float32)
    mat = zscore(mat, axis=0, ddof=0)
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    mat = np.nan_to_num(mat)
    mat = mat / math.sqrt(mat.shape[0])
    return mat

def test_voxel_selection():
    fake_raw_data = [create_epoch(), create_epoch(),
                     create_epoch(), create_epoch()]
    labels = [0, 1, 0, 1]
    # set master rank to be 0
    vs = VoxelSelector(fake_raw_data, 2, labels, 2, master_rank=0)
    # for cross validation, use SVM with precomputed kernel
    # no shrinking, set C=10
    clf = svm.SVC(kernel='precomputed', shrinking=False, C=10)
    results0 = vs.run(clf)
    # set master rank to be 1 and do it again
    vs = VoxelSelector(fake_raw_data, 2, labels, 2, master_rank=1)
    results1 = vs.run(clf)
    # test scipy normalization
    fake_corr = np.random.rand(1, 12, 100).astype(np.float32)
    fake_corr = vs._correlationNormalization(fake_corr)
    # make one process sleep a while to resolve file writing competition
    if MPI.COMM_WORLD.Get_rank() == 0:
        time.sleep(0.5)
    return results0, results1, fake_corr
