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

import numpy as np
from mpi4py import MPI
import sys

"""Distributed Searchlight
"""

__all__ = [
        "Searchlight",
]


class Searchlight:
    """Distributed Searchlight

    Run a user-defined function over each voxel in a multi-subject
    dataset.

    Optionally, users can define a block function which runs over
    larger portions of the volume called blocks.
    """
    def __init__(self, sl_rad=1, max_blk_edge=10):
        """Constructor

        Parameters
        ----------

        sl_rad: radius, in voxels, of the sphere inscribed in the
                   searchlight cube, not counting the center voxel

        max_blk_edge: max edge length, in voxels, of the 3D block

        """
        self.sl_rad = sl_rad
        self.max_blk_edge = max_blk_edge
        self.comm = MPI.COMM_WORLD

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
        outerblk = self.max_blk_edge + 2*self.sl_rad
        return [((i, j, k),
                mask[i:i+outerblk,
                j:j+outerblk,
                k:k+outerblk].shape)
                for i in range(0, mask.shape[0], self.max_blk_edge)
                for j in range(0, mask.shape[1], self.max_blk_edge)
                for k in range(0, mask.shape[2], self.max_blk_edge)
                if np.any(
                mask[i+self.sl_rad:i+self.sl_rad+self.max_blk_edge,
                     j+self.sl_rad:j+self.sl_rad+self.max_blk_edge,
                     k+self.sl_rad:k+self.sl_rad+self.max_blk_edge])]

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
        else:
            sys.exit(0)

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

    def run_block_function(self, block_fn):
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


                Returns

                3D array which is the same size as the mask
                input with padding removed


        """
        rank = self.comm.rank

        # Apply searchlight
        local_outputs = [(mypt[0],
                          block_fn([d[idx] for d in self.subproblems],
                          self.submasks[idx], self.sl_rad, self.bcast_var))
                         for (idx, mypt) in enumerate(self.blocks)]

        # Collect results
        global_outputs = self.comm.gather(local_outputs)

        # Coalesce results
        outmat = np.empty(self.mask.shape, dtype=np.object)
        if rank == 0:
            for go_rank in global_outputs:
                for (pt, mat) in go_rank:
                    for i in range(0, mat.shape[0]):
                        for j in range(0, mat.shape[1]):
                            for k in range(0, mat.shape[2]):
                                outmat[pt[0]+self.sl_rad+i,
                                       pt[1]+self.sl_rad+j,
                                       pt[2]+self.sl_rad+k] = mat[i, j, k]

        return outmat

    def run_searchlight(self, voxel_fn, pool_size=None):
        """Perform a function at each active voxel

        Parameters
        ----------

        voxel_fn: function to apply at each voxel

                Parameters

                subj: list of 4D arrays containing subset of subject data

                mask: 3D array containing subset of mask data

                sl_rad: radius, in voxels, of the sphere inscribed in the
                cube

                bcast_var: shared data which is broadcast to all processes


                Returns

                Value of any pickle-able type

        pool_size:    Number of parallel processes in shared memory
                      process pool

        """

        def _singlenode_searchlight(l, msk, mysl_rad, bcast_var):

            outmat = np.empty(msk.shape, dtype=np.object)[mysl_rad:-mysl_rad,
                                                          mysl_rad:-mysl_rad,
                                                          mysl_rad:-mysl_rad]

            import pathos.multiprocessing
            inlist = [([ll[i:i+2*mysl_rad+1,
                           j:j+2*mysl_rad+1,
                           k:k+2*mysl_rad+1,
                           :]
                        for ll in l],
                       msk[i:i+2*mysl_rad+1,
                           j:j+2*mysl_rad+1,
                           k:k+2*mysl_rad+1],
                       mysl_rad,
                       bcast_var)
                      if msk[i+mysl_rad, j+mysl_rad, k+mysl_rad] else None
                      for i in range(0, outmat.shape[0])
                      for j in range(0, outmat.shape[1])
                      for k in range(0, outmat.shape[2])]
            outlist = list(pathos.multiprocessing.ProcessingPool(pool_size)
                           .map(lambda x:
                                voxel_fn(x[0], x[1], x[2], x[3])
                                if x is not None else None, inlist))

            cnt = 0
            for i in range(0, outmat.shape[0]):
                for j in range(0, outmat.shape[1]):
                    for k in range(0, outmat.shape[2]):
                        if outlist[cnt] is not None:
                            outmat[i, j, k] = outlist[cnt]
                        cnt = cnt + 1

            return outmat

        return self.run_block_function(_singlenode_searchlight)
