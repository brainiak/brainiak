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
from mpi4py import MPI
import sys

from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Diamond

def test_cube():
    sl = Searchlight(sl_rad=3)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    dim0, dim1, dim2 = (50,50,50)
    ntr = 30
    nsubj = 3
    mask = np.zeros((dim0,dim1,dim2), dtype=np.bool)
    data = [np.empty((dim0,dim1,dim2,ntr), dtype=np.object) if i % size == rank else None for i in range(0, nsubj)]

    # Put a spot in the mask
    mask[10:17,10:17,10:17] = True

    def sfn(l, msk, myrad, bcast_var):
        return 1.0

    sl.distribute(data, mask)
    global_outputs = sl.run_searchlight(sfn)

    if rank == 0:
        assert global_outputs[13,13,13] == 1.0
        global_outputs[13,13,13] = None

        for i in range(global_outputs.shape[0]):
            for j in range(global_outputs.shape[1]):
                for k in range(global_outputs.shape[2]):
                    assert global_outputs[i,j,k] == None

def test_diamond():
    sl = Searchlight(sl_rad=3)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    dim0, dim1, dim2 = (50,50,50)
    ntr = 30
    nsubj = 3
    mask = np.zeros((dim0,dim1,dim2), dtype=np.bool)
    data = [np.empty((dim0,dim1,dim2,ntr), dtype=np.object) if i % size == rank else None for i in range(0, nsubj)]

    # Put a spot in the mask
    mask[10:17,10:17,10:17] = Diamond(3)

    def sfn(l, msk, myrad, bcast_var):
        return 1.0

    sl.distribute(data, mask)
    global_outputs = sl.run_searchlight(sfn)

    if rank == 0:
        assert global_outputs[13,13,13] == 1.0
        global_outputs[13,13,13] = None

        for i in range(global_outputs.shape[0]):
            for j in range(global_outputs.shape[1]):
                for k in range(global_outputs.shape[2]):
                    assert global_outputs[i,j,k] == None


