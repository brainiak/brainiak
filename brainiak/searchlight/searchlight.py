#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#               http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from mpi4py import MPI
import numpy as np

"""Distributed searchlight
Given a function and a list of volumes, this applies the function
to all voxels.
"""

# Author: Michael Anderson

__all__ = [
    "Searchlight",
]


class Searchlight:
    """ Class for searchlight.
    A searchlight is a computation that is applied in a sliding
    window to subsets of voxels across a volume. The radius of the
    searchlight specifies how many voxels are involved in each
    computation. A user-provided function is applied to all voxels
    within the radius, provided that the mask was non-zero at that
    point in the volume. The ouput of the searchlight analysis is
    another volume which contains the user-provided function's
    output at each point.

    Parameters
    ----------
    None

    Attributes
    ----------
    None
    """

    def __init__(self):
        pass

    def _get_subarray(self, data, idx, rad):
        """ Return a subarray with radius 'rad', centered around 'idx' with the
                same list structure as 'data'.

        Parameters
        ----------
        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        idx: Three-element tuple containing the current center of the
        searchlight

        rad: Odd positive integer indicating the radius of the searchlight
        cube.

        """

        def _list_slice(l):
            if(isinstance(l, list)):
                return [_list_slice(el) for el in l]
            else:
                return l[:, idx[0] - rad:idx[0] + rad + 1,
                         idx[1] - rad:idx[1] + rad + 1,
                         idx[2] - rad:idx[2] + rad + 1]

        return _list_slice(data)

    def _get_submask(self, mask, idx, rad):
        """ Return a subarray of the mask centered around 'idx'

        Parameters
        ----------
        mask: A 3D numpy.ndarray object with type np.bool indicating which
                    voxels are active in the searchlight operation.

        idx: Three-element tuple containing the current center of the
        searchlight

        rad: Odd positive integer indicating the radius of the searchlight
        cube.

        """

        return mask[idx[0] - rad:idx[0] + rad + 1,
                    idx[1] - rad:idx[1] + rad + 1,
                    idx[2] - rad:idx[2] + rad + 1]

    def _do_master(self, tasks, data, mask, fn, rad):
        """ Distribute tasks dynamically to ranks 1-size (master)

        Parameters
        ----------
        tasks: searchlight centers, a list of tuples

        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        mask: A 3D numpy.ndarray object with type np.bool indicating which
        voxels are active in the searchlight operation.

        fn: A user-provided function which is applied to the data in a
        searchlight.
            This user-specified function must accept 2 parameters:
                data: A list of 4D numpy.ndarray objects which has the same
                structure as the 'data' parameter to the searchlight, but
                only includes voxels within a 'rad' radius from the current
                searchlight center
                mask: A 3D numpy.ndarray object with type np.bool indicating
                which includes voxels within a 'rad' radius from the current
                searchlight center

        rad: Odd positive integer indicating the radius of the searchlight
        cube.

        """

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        results = []

        # If there are no workers, then do everything here
        if size == 1:
            for idx in tasks:
                results += [(idx, fn(self._get_subarray(data, idx, rad),
                                     self._get_submask(mask, idx, rad)))]
        # Assign tasks to workers
        else:
            num_active = 0

            # Send wait for requests from workers
            for idx in tasks:
                (dst_rank, result) = comm.recv(source=MPI.ANY_SOURCE)

                # Add result to output array
                if result is not None:
                    results += [result]
                    num_active -= 1

                # Send a valid task
                comm.send((idx, self._get_subarray(data, idx, rad),
                           self._get_submask(mask, idx, rad)), dest=dst_rank)
                num_active += 1

            # wait for all to finish
            while num_active > 0:
                (dst_rank, result) = comm.recv(source=MPI.ANY_SOURCE)

                # Add result to output array
                if result is not None:
                    results += [result]
                    num_active -= 1

            # Send each an empty task
            for r in range(1, size):
                comm.send((-1, [], []), dest=r)

        return results

    def _do_worker(self, tasks, data, mask, fn, rad):
        """ Distribute tasks dynamically to ranks 1-size (worker)

        Parameters
        ----------
        tasks: searchlight centers, a list of tuples

        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        mask: A 3D numpy.ndarray object with type np.bool indicating which
        voxels are active in the searchlight operation.

        fn: A user-provided function which is applied to the data in a
        searchlight.
            This user-specified function must accept 2 parameters:
                data: A list of 4D numpy.ndarray objects which has the same
                structure as the 'data' parameter to the searchlight, but
                only includes voxels within a 'rad' radius from the current
                searchlight center
                mask: A 3D numpy.ndarray object with type np.bool indicating
                which includes voxels within a 'rad' radius from the current
                searchlight center

        rad: Odd positive integer indicating the radius of the searchlight
        cube.

        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Request task (empty result)
        comm.send((rank, None), dest=0)

        # Infinite loop
        while True:

            # Receive task
            (idx, d, m) = comm.recv(source=0)

            # If task is empty then break loop
            if(idx == -1):
                break

            result = fn(d, m)

            # Send result
            comm.send((rank, (idx, result)), dest=0)

    def _dynamic_tasking(self, tasks, data, mask, fn, rad):
        """ Distribute tasks dynamically to ranks 1-size

        Parameters
        ----------
        tasks: searchlight centers, a list of tuples

        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        mask: A 3D numpy.ndarray object with type np.bool indicating which
        voxels are active in the searchlight operation.

        fn: A user-provided function which is applied to the data in a
        searchlight.
            This user-specified function must accept 2 parameters:
                data: A list of 4D numpy.ndarray objects which has the same
                structure as the 'data' parameter to the searchlight, but
                only includes voxels within a 'rad' radius from the current
                searchlight center
                mask: A 3D numpy.ndarray object with type np.bool indicating
                which includes voxels within a 'rad' radius from the current
                searchlight center

        rad: Odd positive integer indicating the radius of the searchlight
        cube.

        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        results = []

        if rank == 0:
            results = self._do_master(tasks, data, mask, fn, rad)

        if rank != 0:
            self._do_worker(tasks, data, mask, fn, rad)

        return results

    # Partition the mask and create tasks
    def run(self, data, mask, fn, rad=1):
        """ Run searchlight

        Applies a function to each voxel present in the mask.

        Parameters
        ---------
        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        mask: A 3D numpy.ndarray object with type np.bool indicating which
                    voxels are active in the searchlight operation.

        fn: A user-provided function which is applied to the data in a
        searchlight.
        This user-specified function must accept 2 parameters:
          data: A list of 4D numpy.ndarray objects which has the same
          structure as the 'data' parameter to the searchlight, but only
          includes voxels within a 'rad' radius from the current searchlight
          center
          mask: A 3D numpy.ndarray object with type np.bool indicating which
          includes voxels within a 'rad' radius from the current
          searchlight center

        rad: Odd positive integer indicating the radius of the searchlight
        cube.

        Returns
        ----------
        output: A 3D numpy ndarray containing the return value from each
        invocation of the user-specified function. The output array
        will be the same size as the input mask. The values around the
        boundary (from 0 to rad, and from dim-rad to dim) will be
        None. Also, points where the input mask is "false" will be
        None in the output array.
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        tasks = []
        if rank == 0:

            # Create tasks
            for i in range(rad, mask.shape[0] - rad):
                for j in range(rad, mask.shape[1] - rad):
                    for k in range(rad, mask.shape[2] - rad):
                        if mask[i, j, k]:
                            tasks += [(i, j, k)]

        # Run dynamic tasking
        outputs = self._dynamic_tasking(tasks, data, mask, fn, rad)

        # Create output volume
        output = None
        if rank == 0:
            output = np.empty(mask.shape, dtype=object)

            for o in outputs:
                output[o[0]] = o[1]

        return output
