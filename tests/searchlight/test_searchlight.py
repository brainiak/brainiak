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

from collections import namedtuple

import numpy as np
from mpi4py import MPI

from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Diamond

"""Distributed Searchlight Test
"""


def cube_sfn(l, msk, myrad, bcast_var):
    if np.all(msk) and np.any(msk):
        return 1.0
    return None


def test_searchlight_with_cube():
    sl = Searchlight(sl_rad=3)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    dim0, dim1, dim2 = (50, 50, 50)
    ntr = 30
    nsubj = 3
    mask = np.zeros((dim0, dim1, dim2), dtype=np.bool)
    data = [np.empty((dim0, dim1, dim2, ntr), dtype=np.object)
            if i % size == rank
            else None
            for i in range(0, nsubj)]

    # Put a spot in the mask
    mask[10:17, 10:17, 10:17] = True

    sl.distribute(data, mask)
    global_outputs = sl.run_searchlight(cube_sfn)

    if rank == 0:
        assert global_outputs[13, 13, 13] == 1.0
        global_outputs[13, 13, 13] = None

        for i in range(global_outputs.shape[0]):
            for j in range(global_outputs.shape[1]):
                for k in range(global_outputs.shape[2]):
                    assert global_outputs[i, j, k] is None


def diamond_sfn(l, msk, myrad, bcast_var):
    assert not np.any(msk[~Diamond(3).mask_])
    if np.all(msk[Diamond(3).mask_]):
        return 1.0
    return None


def test_searchlight_with_diamond():
    sl = Searchlight(sl_rad=3, shape=Diamond)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    dim0, dim1, dim2 = (50, 50, 50)
    ntr = 30
    nsubj = 3
    mask = np.zeros((dim0, dim1, dim2), dtype=np.bool)
    data = [np.empty((dim0, dim1, dim2, ntr), dtype=np.object)
            if i % size == rank
            else None
            for i in range(0, nsubj)]

    # Put a spot in the mask
    mask[10:17, 10:17, 10:17] = Diamond(3).mask_

    sl.distribute(data, mask)
    global_outputs = sl.run_searchlight(diamond_sfn)

    if rank == 0:
        assert global_outputs[13, 13, 13] == 1.0
        global_outputs[13, 13, 13] = None

        for i in range(global_outputs.shape[0]):
            for j in range(global_outputs.shape[1]):
                for k in range(global_outputs.shape[2]):
                    assert global_outputs[i, j, k] is None


MaskRadBcast = namedtuple("MaskRadBcast", "mask rad")


def test_instantiate():
    sl = Searchlight(sl_rad=5, max_blk_edge=10)
    assert sl


def voxel_test_sfn(l, msk, myrad, bcast):
    rad = bcast.rad
    # Check each point
    for subj in l:
        for _tr in range(subj.shape[3]):
            tr = subj[:, :, :, _tr]
            midpt = tr[rad, rad, rad]
            for d0 in range(tr.shape[0]):
                for d1 in range(tr.shape[1]):
                    for d2 in range(tr.shape[2]):
                        assert np.array_equal(tr[d0, d1, d2] - midpt,
                                              np.array([d0-rad, d1-rad,
                                                        d2-rad, 0]))

    # Determine midpoint
    midpt = l[0][rad, rad, rad, 0]
    midpt = (midpt[0], midpt[1], midpt[2])

    for d0 in range(msk.shape[0]):
        for d1 in range(msk.shape[1]):
            for d2 in range(msk.shape[2]):
                pt = (midpt[0] - rad + d0, midpt[1] - rad + d1,
                      midpt[2] - rad + d2)
                assert bcast.mask[pt] == msk[d0, d1, d2]

    # Return midpoint
    return midpt


def block_test_sfn(l, msk, myrad, bcast_var, extra_params):
    outmat = l[0][:, :, :, 0]
    outmat[~msk] = None
    return outmat[myrad:-myrad, myrad:-myrad, myrad:-myrad]


def test_correctness():  # noqa: C901
    def voxel_test(data, mask, max_blk_edge, rad):

        comm = MPI.COMM_WORLD
        rank = comm.rank

        (dim0, dim1, dim2) = mask.shape

        # Initialize dataset with known pattern
        for subj in data:
            if subj is not None:
                for tr in range(subj.shape[3]):
                    for d1 in range(dim0):
                        for d2 in range(dim1):
                            for d3 in range(dim2):
                                subj[d1, d2, d3, tr] = np.array(
                                    [d1, d2, d3, tr])

        sl = Searchlight(sl_rad=rad, max_blk_edge=max_blk_edge)
        sl.distribute(data, mask)
        sl.broadcast(MaskRadBcast(mask, rad))
        global_outputs = sl.run_searchlight(voxel_test_sfn)

        if rank == 0:
            for d0 in range(rad, global_outputs.shape[0]-rad):
                for d1 in range(rad, global_outputs.shape[1]-rad):
                    for d2 in range(rad, global_outputs.shape[2]-rad):
                        if mask[d0, d1, d2]:
                            assert np.array_equal(
                                np.array(global_outputs[d0, d1, d2]),
                                np.array([d0, d1, d2]))

    def block_test(data, mask, max_blk_edge, rad):

        comm = MPI.COMM_WORLD
        rank = comm.rank

        (dim0, dim1, dim2) = mask.shape

        # Initialize dataset with known pattern
        for subj in data:
            if subj is not None:
                for tr in range(subj.shape[3]):
                    for d1 in range(dim0):
                        for d2 in range(dim1):
                            for d3 in range(dim2):
                                subj[d1, d2, d3, tr] = np.array(
                                    [d1, d2, d3, tr])

        sl = Searchlight(sl_rad=rad, max_blk_edge=max_blk_edge)
        sl.distribute(data, mask)
        sl.broadcast(mask)
        global_outputs = sl.run_block_function(block_test_sfn)

        if rank == 0:
            for d0 in range(rad, global_outputs.shape[0]-rad):
                for d1 in range(rad, global_outputs.shape[1]-rad):
                    for d2 in range(rad, global_outputs.shape[2]-rad):
                        if mask[d0, d1, d2]:
                            assert np.array_equal(
                                np.array(global_outputs[d0, d1, d2]),
                                np.array([d0, d1, d2, 0]))

    # Create dataset
    def do_test(dim0, dim1, dim2, ntr, nsubj, max_blk_edge, rad):
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        mask = np.random.choice([True, False], (dim0, dim1, dim2))
        data = [np.empty((dim0, dim1, dim2, ntr), dtype=np.object)
                if i % size == rank
                else None
                for i in range(0, nsubj)]
        voxel_test(data, mask, max_blk_edge, rad)
        block_test(data, mask, max_blk_edge, rad)

    do_test(dim0=7, dim1=5, dim2=9, ntr=5, nsubj=1, max_blk_edge=4, rad=1)
    do_test(dim0=7, dim1=5, dim2=9, ntr=5, nsubj=5, max_blk_edge=4, rad=1)
    do_test(dim0=1, dim1=5, dim2=9, ntr=5, nsubj=5, max_blk_edge=4, rad=1)
    do_test(dim0=0, dim1=10, dim2=8, ntr=5, nsubj=5, max_blk_edge=4, rad=1)
    do_test(dim0=7, dim1=5, dim2=9, ntr=5, nsubj=1, max_blk_edge=4, rad=2)
    do_test(dim0=7, dim1=5, dim2=9, ntr=5, nsubj=1, max_blk_edge=4, rad=3)
