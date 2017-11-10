#    Copyright 2016 Intel Corporation
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#             http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from multiprocessing import Pool
import numpy as np
from mpi4py import MPI
from scipy.spatial.distance import cityblock

from ..utils.utils import usable_cpu_count

"""Distributed Searchlight
"""

__all__ = [
        "Searchlight",
]


class Shape:
    """Shape

    Searchlight shape which is contained in a cube sized
    (2*rad+1,2*rad+1,2*rad+1)

    Attributes
    ----------

    mask_ : a 3D boolean numpy array of size (2*rad+1,2*rad+1,2*rad+1)
            which is set to True within the boundaries of the desired shape
    """

    def __init__(self, rad):
        """Constructor

        Parameters
        ----------

        rad: radius, in voxels, of the sphere inscribed in the
             searchlight cube, not counting the center voxel

        """
        self.rad = rad


class Cube(Shape):
    """Cube

    Searchlight shape which is a cube of size (2*rad+1,2*rad+1,2*rad+1)
    """
    def __init__(self, rad):
        """Constructor

        Parameters
        ----------

        rad: radius, in voxels, of the sphere inscribed in the
             searchlight cube, not counting the center voxel

        """
        super().__init__(rad)
        self.rad = rad
        self.mask_ = np.ones((2*rad+1, 2*rad+1, 2*rad+1), dtype=np.bool)


class Diamond(Shape):
    """Diamond

    Searchlight shape which is a diamond
    inscribed in a cube of size (2*rad+1,2*rad+1,2*rad+1).
    Any location in the cube which has a Manhattan distance of less than rad
    from the center point is set to True.
    """
    def __init__(self, rad):
        """Constructor

        Parameters
        ----------

        rad: radius, in voxels, of the sphere inscribed in the
             searchlight cube, not counting the center voxel

        """
        super().__init__(rad)
        self.mask_ = np.zeros((2*rad+1, 2*rad+1, 2*rad+1), dtype=np.bool)
        for r1 in range(2*self.rad+1):
            for r2 in range(2*self.rad+1):
                for r3 in range(2*self.rad+1):
                    if(cityblock((r1, r2, r3),
                                 (self.rad, self.rad, self.rad)) <= self.rad):
                        self.mask_[r1, r2, r3] = True


class Searchlight:
    """Distributed Searchlight

    Run a user-defined function over each voxel in a multi-subject
    dataset.

    Optionally, users can define a block function which runs over
    larger portions of the volume called blocks.
    """
    def __init__(self, sl_rad=1, max_blk_edge=10, shape=Cube,
                 min_active_voxels_proportion=0):
        """Constructor

        Parameters
        ----------

        sl_rad: radius, in voxels, of the sphere inscribed in the
                   searchlight cube, not counting the center voxel

        max_blk_edge: max edge length, in voxels, of the 3D block

        shape: brainiak.searchlight.searchlight.Shape indicating the
        shape in voxels of the searchlight region

        min_active_voxels_proportion: float
            If a searchlight region does not have more than this minimum
            proportion of active voxels in the mask, it is not processed by the
            searchlight function. The mask used for the test is the
            intersection of the global (brain) mask and the `Shape` mask. The
            seed (central) voxel of the searchlight region is taken into
            consideration.
        """
        self.sl_rad = sl_rad
        self.max_blk_edge = max_blk_edge
        self.min_active_voxels_proportion = min_active_voxels_proportion
        self.comm = MPI.COMM_WORLD
        self.shape = shape(sl_rad).mask_
        self.bcast_var = None

    def _get_ownership(self, data):
        """Determine on which rank each subject currently resides

        Parameters
        ----------

        data: list of 4D arrays with subject data

        Returns
        -------

        list of ranks indicating the owner of each subject
        """
        rank = self.comm.rank

        B = [(rank, idx) for (idx, c) in enumerate(data) if c is not None]
        C = self.comm.allreduce(B)
        ownership = [None] * len(data)
        for c in C:
            ownership[c[1]] = c[0]
        return ownership

    def _get_blocks(self, mask):
        """Divide the volume into a set of blocks

        Ignore blocks that have no active voxels in the mask

        Parameters
        ----------

        mask: a boolean 3D array which is true at every active voxel

        Returns
        -------

        list of tuples containing block information:
           - a triple containing top left point of the block and
           - a triple containing the size in voxels of the block

        """
        blocks = []
        outerblk = self.max_blk_edge + 2*self.sl_rad
        for i in range(0, mask.shape[0], self.max_blk_edge):
            for j in range(0, mask.shape[1], self.max_blk_edge):
                for k in range(0, mask.shape[2], self.max_blk_edge):
                    block_shape = mask[i:i+outerblk,
                                       j:j+outerblk,
                                       k:k+outerblk
                                       ].shape
                    if np.any(
                        mask[i+self.sl_rad:i+block_shape[0]-self.sl_rad,
                             j+self.sl_rad:j+block_shape[1]-self.sl_rad,
                             k+self.sl_rad:k+block_shape[2]-self.sl_rad]):
                        blocks.append(((i, j, k), block_shape))
        return blocks

    def _get_block_data(self, mat, block):
        """Retrieve a block from a 3D or 4D volume

        Parameters
        ----------

        mat: a 3D or 4D volume

        block: a tuple containing block information:
          - a triple containing the lowest-coordinate voxel in the block
          - a triple containing the size in voxels of the block

        Returns
        -------

        In the case of a 3D array, a 3D subarray at the block location

        In the case of a 4D array, a 4D subarray at the block location,
        including the entire fourth dimension.
        """
        (pt, sz) = block
        if len(mat.shape) == 3:
            return mat[pt[0]:pt[0]+sz[0],
                       pt[1]:pt[1]+sz[1],
                       pt[2]:pt[2]+sz[2]].copy()
        elif len(mat.shape) == 4:
            return mat[pt[0]:pt[0]+sz[0],
                       pt[1]:pt[1]+sz[1],
                       pt[2]:pt[2]+sz[2],
                       :].copy()

    def _split_volume(self, mat, blocks):
        """Convert a volume into a list of block data

        Parameters
        ----------

        mat: A 3D or 4D array to be split

        blocks: a list of tuples containing block information:
          - a triple containing the top left point of the block and
          - a triple containing the size in voxels of the block


        Returns
        -------

        A list of the subarrays corresponding to each block

        """
        return [self._get_block_data(mat, block) for block in blocks]

    def _scatter_list(self, data, owner):
        """Distribute a list from one rank to other ranks in a cyclic manner

        Parameters
        ----------

        data: list of pickle-able data

        owner: rank that owns the data

        Returns
        -------

        A list containing the data in a cyclic layout across ranks

        """

        rank = self.comm.rank
        size = self.comm.size
        subject_submatrices = []
        nblocks = self.comm.bcast(len(data)
                                  if rank == owner else None, root=owner)

        # For each submatrix
        for idx in range(0, nblocks, size):
            padded = None
            extra = max(0, idx+size - nblocks)

            # Pad with "None" so scatter can go to all processes
            if data is not None:
                padded = data[idx:idx+size]
                if extra > 0:
                    padded = padded + [None]*extra

            # Scatter submatrices to all processes
            mytrans = self.comm.scatter(padded, root=owner)

            # Contribute submatrix to subject list
            if mytrans is not None:
                subject_submatrices += [mytrans]

        return subject_submatrices

    def distribute(self, subjects, mask):
        """Distribute data to MPI ranks

        Parameters
        ----------

        subjects : list of 4D arrays containing data for one or more subjects.
              Each entry of the list must be present on at most one rank,
              and the other ranks contain a "None" at this list location.

              For example, for 3 ranks you may lay out the data in the
              following manner:

              Rank 0: [Subj0, None, None]
              Rank 1: [None, Subj1, None]
              Rank 2: [None, None, Subj2]

              Or alternatively, you may lay out the data in this manner:

              Rank 0: [Subj0, Subj1, Subj2]
              Rank 1: [None, None, None]
              Rank 2: [None, None, None]

        mask: 3D array with "True" entries at active vertices

        """
        if mask.ndim != 3:
            raise ValueError('mask should be a 3D array')

        for (idx, subj) in enumerate(subjects):
            if subj is not None:
                if subj.ndim != 4:
                    raise ValueError('subjects[{}] must be 4D'.format(idx))

        self.mask = mask
        rank = self.comm.rank

        # Get/set ownership
        ownership = self._get_ownership(subjects)
        all_blocks = self._get_blocks(mask) if rank == 0 else None
        all_blocks = self.comm.bcast(all_blocks)

        # Divide data and mask
        splitsubj = [self._split_volume(s, all_blocks)
                     if s is not None else None
                     for s in subjects]
        submasks = self._split_volume(mask, all_blocks)

        # Scatter points, data, and mask
        self.blocks = self._scatter_list(all_blocks, 0)
        self.submasks = self._scatter_list(submasks,  0)
        self.subproblems = [self._scatter_list(s, ownership[s_idx])
                            for (s_idx, s) in enumerate(splitsubj)]

    def broadcast(self, bcast_var):
        """Distribute data to processes

        Parameters
        ----------

        bcast_var:    shared data which is broadcast to all processes

        """

        self.bcast_var = self.comm.bcast(bcast_var)

    def run_block_function(self, block_fn, extra_block_fn_params=None,
                           pool_size=None):
        """Perform a function for each block in a volume.

        Parameters
        ----------

        block_fn: function to apply to each block:

                Parameters

                data: list of 4D arrays containing subset of subject data,
                      which is padded with sl_rad voxels.

                mask: 3D array containing subset of mask data

                sl_rad: radius, in voxels, of the sphere inscribed in the
                cube

                bcast_var: shared data which is broadcast to all processes

                extra_params: extra parameters


                Returns

                3D array which is the same size as the mask
                input with padding removed

        extra_block_fn_params: tuple
            Extra parameters to pass to the block function

        pool_size: int
            Maximum number of processes running the block function in parallel.
            If None, number of available hardware threads, considering cpusets
            restrictions.
        """
        rank = self.comm.rank

        results = []
        usable_cpus = usable_cpu_count()
        if pool_size is None:
            processes = usable_cpus
        else:
            processes = min(pool_size, usable_cpus)
        with Pool(processes) as pool:
            for idx, block in enumerate(self.blocks):
                result = pool.apply_async(
                    block_fn,
                    ([subproblem[idx] for subproblem in self.subproblems],
                     self.submasks[idx],
                     self.sl_rad,
                     self.bcast_var,
                     extra_block_fn_params))
                results.append((block[0], result))
            local_outputs = [(result[0], result[1].get())
                             for result in results]

        # Collect results
        global_outputs = self.comm.gather(local_outputs)

        # Coalesce results
        outmat = np.empty(self.mask.shape, dtype=np.object)
        if rank == 0:
            for go_rank in global_outputs:
                for (pt, mat) in go_rank:
                    coords = np.s_[
                        pt[0]+self.sl_rad:pt[0]+self.sl_rad+mat.shape[0],
                        pt[1]+self.sl_rad:pt[1]+self.sl_rad+mat.shape[1],
                        pt[2]+self.sl_rad:pt[2]+self.sl_rad+mat.shape[2]
                    ]
                    outmat[coords] = mat
        return outmat

    def run_searchlight(self, voxel_fn, pool_size=None):
        """Perform a function at each voxel which is set to True in the
        user-provided mask. The mask passed to the searchlight function will be
        further masked by the user-provided searchlight shape.

        Parameters
        ----------

        voxel_fn: function to apply at each voxel

            Must be `serializeable using pickle
            <https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled>`_.

                Parameters

                subj: list of 4D arrays containing subset of subject data

                mask: 3D array containing subset of mask data

                sl_rad: radius, in voxels, of the sphere inscribed in the
                cube

                bcast_var: shared data which is broadcast to all processes


                Returns

                Value of any pickle-able type

        Returns
        -------

        A volume which is the same size as the mask, however a number of voxels
        equal to the searchlight radius has been removed from each border of
        the volume. This volume contains the values returned from the
        searchlight function at each voxel which was set to True in the mask,
        and None elsewhere.

        """

        extra_block_fn_params = (voxel_fn, self.shape,
                                 self.min_active_voxels_proportion)
        block_fn_result = self.run_block_function(_singlenode_searchlight,
                                                  extra_block_fn_params,
                                                  pool_size)
        return block_fn_result


def _singlenode_searchlight(l, msk, mysl_rad, bcast_var, extra_params):
    """Run searchlight function on block data in parallel.

    `extra_params` contains:

    - Searchlight function.
    - `Shape` mask.
    - Minimum active voxels proportion required to run the searchlight
      function.
    """

    voxel_fn = extra_params[0]
    shape_mask = extra_params[1]
    min_active_voxels_proportion = extra_params[2]
    outmat = np.empty(msk.shape, dtype=np.object)[mysl_rad:-mysl_rad,
                                                  mysl_rad:-mysl_rad,
                                                  mysl_rad:-mysl_rad]
    for i in range(0, outmat.shape[0]):
        for j in range(0, outmat.shape[1]):
            for k in range(0, outmat.shape[2]):
                if msk[i+mysl_rad, j+mysl_rad, k+mysl_rad]:
                    searchlight_slice = np.s_[
                        i:i+2*mysl_rad+1,
                        j:j+2*mysl_rad+1,
                        k:k+2*mysl_rad+1]
                    voxel_fn_mask = msk[searchlight_slice] * shape_mask
                    if (min_active_voxels_proportion == 0
                        or np.count_nonzero(voxel_fn_mask) / voxel_fn_mask.size
                            > min_active_voxels_proportion):
                        outmat[i, j, k] = voxel_fn(
                            [ll[searchlight_slice] for ll in l],
                            msk[searchlight_slice] * shape_mask,
                            mysl_rad,
                            bcast_var)
    return outmat
