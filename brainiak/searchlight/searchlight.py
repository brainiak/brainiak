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
from sklearn.base import TransformerMixin

"""Distributed searchlight

Given a function and a list of volumes, this applies the function
to all voxels.
"""

# Author: Michael Anderson

__all__ = [
    "Searchlight",
]


class Searchlight(TransformerMixin):
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

    rad: Positive integer indicating the radius of the searchlight
         cube.

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

    Attributes
    ----------
    None
    """

    def __init__(self, rad, fn):
        """ Constructor for searchlight.

        rad: Positive integer indicating the radius of the searchlight
                cube.

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
        """

        self.rad = rad
        self.fn = fn

    def _get_subarray(self, data, idx):
        """ Return a subarray with radius 'rad'

        The subarray is centered around 'idx' with the
        same list structure as 'data'.

        Parameters
        ----------
        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        idx: Three-element tuple containing the current center of the
        searchlight

        Returns
        -------
        A list of 4D numpy ndarrays (subarrays)

        """

        def _list_slice(l):
            if(isinstance(l, list)):
                return [_list_slice(el) for el in l]
            else:
                return l[:, idx[0] - self.rad:idx[0] + self.rad + 1,
                         idx[1] - self.rad:idx[1] + self.rad + 1,
                         idx[2] - self.rad:idx[2] + self.rad + 1]

        return _list_slice(data)

    def _get_submask(self, mask, idx):
        """ Return a subarray of the mask centered around 'idx'

        Parameters
        ----------
        mask: A 3D numpy.ndarray object with type np.bool indicating which
                    voxels are active in the searchlight operation.

        idx: Three-element tuple containing the current center of the
        searchlight

        Returns
        ------
        A 3D numpy ndarray (subarray)

        """

        return mask[idx[0] - self.rad:idx[0] + self.rad + 1,
                    idx[1] - self.rad:idx[1] + self.rad + 1,
                    idx[2] - self.rad:idx[2] + self.rad + 1]

    def _do_master(self, tasks, data, mask, bcast_var):
        """ Distribute tasks dynamically to ranks 1-size (master)

        Parameters
        ----------
        tasks: searchlight centers, a list of tuples

        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        mask: A 3D numpy.ndarray object with type np.bool indicating which
        voxels are active in the searchlight operation.

        bcast_var: An object of any pickle-able type which is included
        as a parameter to each invocation of the user-specified function

        Returns
        -------
        A 3D numpy ndarray containing the outputs of the function at
        each voxel.

        """

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        results = []

        # If there are no workers, then do everything here
        if size == 1:
            for idx in tasks:
                results += [(idx, self.fn(self._get_subarray(data, idx),
                                          self._get_submask(mask, idx),
                                          bcast_var))]
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
                comm.send((idx, self._get_subarray(data, idx),
                           self._get_submask(mask, idx)), dest=dst_rank)
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

    def _do_worker(self, tasks, data, mask, bcast_var):
        """ Distribute tasks dynamically to ranks 1-size (worker)

        Parameters
        ----------
        tasks: searchlight centers, a list of tuples

        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        mask: A 3D numpy.ndarray object with type np.bool indicating which
        voxels are active in the searchlight operation.

        bcast_var: An object of any pickle-able type which is included
        as a parameter to each invocation of the user-specified function

        Returns
        -------
        None

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

            result = self.fn(d, m, bcast_var)

            # Send result
            comm.send((rank, (idx, result)), dest=0)

    def _dynamic_tasking(self, tasks, data, mask, bcast_var):
        """ Distribute tasks dynamically to ranks 1-size

        Parameters
        ----------
        tasks: searchlight centers, a list of tuples

        data: A list of 4D numpy.ndarray objects which must have at least one
        element. The computation is applied to this data.

        mask: A 3D numpy.ndarray object with type np.bool indicating which
        voxels are active in the searchlight operation.

        bcast_var: An object of any pickle-able type which is included
        as a parameter to each invocation of the user-specified function

        Returns
        -------
        results: A 3D numpy.ndarray object with the outputs of each
        function invocation

        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        bcast_var = comm.bcast(bcast_var, root=0)

        results = []

        if rank == 0:
            results = self._do_master(tasks, data, mask, bcast_var)

        if rank != 0:
            self._do_worker(tasks, data, mask, bcast_var)

        return results

    def fit_transform(self, X, y=None):
        """ Run searchlight according to mask

        Applies a function to each voxel present in the mask.

        Parameters
        ---------
        X: A tuple containing two objects:

          data: A list of 4D numpy.ndarray objects which must have at least
          one element. The computation is applied to this data.

          mask: A 3D numpy.ndarray object with type np.bool indicating which
          voxels are active in the searchlight operation.

        y: An object of any pickle-able type which is included
        as a parameter to each invocation of the user-specified function

        Returns
        ----------
        X_new: A 3D numpy ndarray containing the return value from each
        invocation of the user-specified function. The output array
        will be the same size as the input mask. The values around the
        boundary (from 0 to rad, and from dim-rad to dim) will be
        None. Also, points where the input mask is "false" will be
        None in the output array.
        """

        (data, mask) = X
        bcast_var = y
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        tasks = []
        if rank == 0:

            # Create tasks
            for i in range(self.rad, mask.shape[0] - self.rad):
                for j in range(self.rad, mask.shape[1] - self.rad):
                    for k in range(self.rad, mask.shape[2] - self.rad):
                        if mask[i, j, k]:
                            tasks += [(i, j, k)]

        # Run dynamic tasking
        outputs = self._dynamic_tasking(tasks, data, mask, bcast_var)

        # Create output volume
        output = None
        if rank == 0:
            output = np.empty(mask.shape, dtype=object)

            for o in outputs:
                output[o[0]] = o[1]

        return output
