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
    def __init__(self, rad=1, region_size=10):
        """Constructor

        Parameters
        ----------

        rad: number of voxels in border region

        region_size: max dimension of a 3D region

        """
        self.rad = rad
        self.region_size = region_size

    def _get_ownership(self, l):
        comm = MPI.COMM_WORLD
        rank = comm.rank

        B = [(rank, idx) for (idx, c) in enumerate(l) if c is not None]
        C = comm.allreduce(B)
        ownership = [None] * len(l)
        for c in C:
            ownership[c[1]] = c[0]
        return ownership

    def _get_points(self, mask):
        outerblk = self.region_size + 2*self.rad
        return [((i, j, k),
                mask[i:i+outerblk,
                j:j+outerblk,
                k:k+outerblk].shape)
                for i in range(0, mask.shape[0], self.region_size)
                for j in range(0, mask.shape[1], self.region_size)
                for k in range(0, mask.shape[2], self.region_size)
                if np.sum(
                mask[i+self.rad:i+self.rad+self.region_size,
                     j+self.rad:j+self.rad+self.region_size,
                     k+self.rad:k+self.rad+self.region_size]) > 0]

    def _subarray(self, mat3d, pts):
        (pt, sz) = pts
        if len(mat3d.shape) == 3:
            return mat3d[pt[0]:pt[0]+sz[0],
                         pt[1]:pt[1]+sz[1],
                         pt[2]:pt[2]+sz[2]].copy()
        elif len(mat3d.shape) == 4:
            return mat3d[:,
                         pt[0]:pt[0]+sz[0],
                         pt[1]:pt[1]+sz[1],
                         pt[2]:pt[2]+sz[2]].copy()
        else:
            print('Error num dims')
            sys.exit(0)

    def _split_volume(self, mat3d, pts):
        return [self._subarray(mat3d, pt) for pt in pts]

    def _scatter_list(self, insubj, pts, owner):
        comm = MPI.COMM_WORLD
        size = comm.size
        subject_submatrices = []

        # For each submatrix
        for idx in range(0, len(pts), size):
            padded = None
            extra = max(0, idx+size - len(pts))

            # Pad with "None" so scatter can go to all processes
            if insubj is not None:
                padded = insubj[idx:idx+size]
                if extra > 0:
                    padded = padded + [None]*extra

            # Scatter submatrices to all processes
            mytrans = comm.scatter(padded, root=owner)

            # Contribute submatrix to subject list
            if mytrans is not None:
                subject_submatrices += [mytrans]

        return subject_submatrices

    def distribute(self, subj, mask):
        """Distribute data to processes

        Parameters
        ----------

        subj: list of 4D arrays containing subject data

        mask: 3D array with nonzero values at active vertices

        """
        self.mask = mask
        comm = MPI.COMM_WORLD
        rank = comm.rank

        # Get/set ownership
        ownership = self._get_ownership(subj)
        full_pts = self._get_points(mask) if rank == 0 else None
        full_pts = comm.bcast(full_pts)

        # Divide data and mask
        splitsubj = [self._split_volume(s, full_pts)
                     if s is not None else None
                     for s in subj]
        submasks = self._split_volume(mask, full_pts)

        # Scatter points, data, and mask
        self.pts = self._scatter_list(full_pts, full_pts, 0)
        self.submasks = self._scatter_list(submasks, full_pts, 0)
        self.subproblems = [self._scatter_list(s, full_pts, ownership[s_idx])
                            for (s_idx, s) in enumerate(splitsubj)]

    def searchlight_region(self, region_fn, bcast_var):
        """Perform a function for each region in a volume.

        Parameters
        ----------

        region_fn: function to apply to each region:

                Paramters
                ---------

                subj: list of 4D arrays containing subset of subject data

                mask: 3D array containing subset of mask data

                rad: number of voxels in border region

                bcast_var: shared data which is broadcast to all processes


                Returns
                -------

                3D array which is the same size as the mask
                input with padding removed

        bcast_var:    shared data which is broadcast to all processes

        """
        comm = MPI.COMM_WORLD
        rank = comm.rank
        bcast_var = comm.bcast(bcast_var)

        # Apply searchlight
        local_outputs = [(mypt[0],
                          region_fn([d[idx] for d in self.subproblems],
                          self.submasks[idx], self.rad, bcast_var))
                         for (idx, mypt) in enumerate(self.pts)]

        # Collect results
        global_outputs = comm.gather(local_outputs)

        # Coalesce results
        outmat = np.empty(self.mask.shape, dtype=np.object)
        if rank == 0:
            for go_rank in global_outputs:
                for (pt, mat) in go_rank:
                    for i in range(0, mat.shape[0]):
                        for j in range(0, mat.shape[1]):
                            for k in range(0, mat.shape[2]):
                                outmat[pt[0]+self.rad+i,
                                       pt[1]+self.rad+j,
                                       pt[2]+self.rad+k] = mat[i, j, k]

        return outmat

    def searchlight_voxel(self, voxel_fn, bcast_var):
        """Perform a function at each active voxel

        Parameters
        ----------

        voxel_fn: function to apply at each voxel

                Paramters
                ---------

                subj: list of 4D arrays containing subset of subject data

                mask: 3D array containing subset of mask data

                rad: number of voxels in border region

                bcast_var: shared data which is broadcast to all processes


                Returns
                -------

                Value of any pickle-able type

        bcast_var:    shared data which is broadcast to all processes

        """

        def _singlenode_searchlight(l, msk, myrad, bcast_var):

            outmat = np.empty(msk.shape, dtype=np.object)[myrad:-myrad,
                                                          myrad:-myrad,
                                                          myrad:-myrad]

            import pathos.multiprocessing
            inlist = [([ll[:,
                           i:i+2*myrad+1,
                           j:j+2*myrad+1,
                           k:k+2*myrad+1]
                        for ll in l],
                       msk[i:i+2*myrad+1,
                           j:j+2*myrad+1,
                           k:k+2*myrad+1],
                       myrad,
                       bcast_var)
                      if msk[i+myrad, j+myrad, k+myrad] else None
                      for i in range(0, outmat.shape[0])
                      for j in range(0, outmat.shape[1])
                      for k in range(0, outmat.shape[2])]
            outlist = list(pathos.multiprocessing.ProcessingPool()
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

        return self.searchlight_region(_singlenode_searchlight, bcast_var)
