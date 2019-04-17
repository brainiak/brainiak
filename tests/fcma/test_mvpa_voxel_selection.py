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

from brainiak.fcma.mvpa_voxelselector import MVPAVoxelSelector
from brainiak.searchlight.searchlight import Searchlight
from sklearn import svm
import numpy as np
from mpi4py import MPI
from numpy.random import RandomState

# specify the random state to fix the random numbers
prng = RandomState(1234567890)


def test_mvpa_voxel_selection():
    data = prng.rand(5, 5, 5, 8).astype(np.float32)
    # all MPI processes read the mask; the mask file is small
    mask = np.ones([5, 5, 5], dtype=np.bool)
    mask[0, 0, :] = False
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    # 2 subjects, 4 epochs per subject
    sl = Searchlight(sl_rad=1)
    mvs = MVPAVoxelSelector(data, mask, labels, 2, sl)
    # for cross validation, use SVM with precomputed kernel

    clf = svm.SVC(kernel='rbf', C=10)
    result_volume, results = mvs.run(clf)
    if MPI.COMM_WORLD.Get_rank() == 0:
        output = []
        for tuple in results:
            if tuple[1] > 0:
                output.append(int(8*tuple[1]))
        expected_output = [6, 6, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4,
                           4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 1]
        assert np.allclose(output, expected_output, atol=1), \
            'voxel selection via SVM does not provide correct results'


if __name__ == '__main__':
    test_mvpa_voxel_selection()
