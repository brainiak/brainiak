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

"""Distributed Searchlight Test
"""

def test_instantiate():
  sl = Searchlight(sl_rad=5, max_blk_edge=10)
  assert sl

def test_correctness():
  def voxel_test(data, mask, max_blk_edge, rad):
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    nsubj = len(data)
    (dim0, dim1, dim2) = mask.shape
    
    # Initialize dataset with known pattern
    for subj in data:
      if subj is not None:
        for tr in range(subj.shape[0]):
          for d1 in range(dim0):
            for d2 in range(dim1):
              for d3 in range(dim2): 
                subj[tr,d1,d2,d3] = np.array([tr, d1, d2, d3])
    
    def sfn(l,msk,myrad,bcast_var):
      # Check each point
      for subj in l:
        for tr in subj:
          midpt = tr[rad,rad,rad]
          for d0 in range(tr.shape[0]):
            for d1 in range(tr.shape[1]):
              for d2 in range(tr.shape[2]):
                assert np.array_equal(tr[d0,d1,d2] - midpt, np.array([0,d0-rad,d1-rad,d2-rad]))
    
      # Determine midpoint
      midpt = l[0][0,rad,rad,rad]
      midpt = (midpt[1], midpt[2], midpt[3])
    
      for d0 in range(msk.shape[0]):
        for d1 in range(msk.shape[1]):
          for d2 in range(msk.shape[2]):
            pt = (midpt[0] - rad + d0, midpt[1] - rad + d1, midpt[2] - rad + d2)
            assert bcast_var[pt] == msk[d0,d1,d2]
    
      # Return midpoint
      return midpt
    
  
    sl = Searchlight(sl_rad=rad, max_blk_edge=max_blk_edge)
    sl.distribute(data, mask)
    sl.broadcast(mask)
    global_outputs = sl.searchlight_voxel(sfn)
  
    if rank == 0:
      for d0 in range(rad, global_outputs.shape[0]-rad):
        for d1 in range(rad, global_outputs.shape[1]-rad):
          for d2 in range(rad, global_outputs.shape[2]-rad):
            if mask[d0, d1, d2]:
              assert np.array_equal(np.array(global_outputs[d0,d1,d2]), np.array([d0,d1,d2]))
  
  
  def block_test(data, mask, max_blk_edge, rad):
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    nsubj = len(data)
    (dim0, dim1, dim2) = mask.shape
    
    # Initialize dataset with known pattern
    for subj in data:
      if subj is not None:
        for tr in range(subj.shape[0]):
          for d1 in range(dim0):
            for d2 in range(dim1):
              for d3 in range(dim2): 
                subj[tr,d1,d2,d3] = np.array([tr, d1, d2, d3])
    
    def sfn(l,msk,myrad,bcast_var):
      outmat = l[0][0,:,:,:] 
      outmat[~msk] = None
      return outmat[rad:-rad,rad:-rad,rad:-rad] 
    
    sl = Searchlight(sl_rad=rad, max_blk_edge=max_blk_edge)
    sl.distribute(data, mask)
    sl.broadcast(mask)
    global_outputs = sl.searchlight_block(sfn)
  
    if rank == 0:
      for d0 in range(rad, global_outputs.shape[0]-rad):
        for d1 in range(rad, global_outputs.shape[1]-rad):
          for d2 in range(rad, global_outputs.shape[2]-rad):
            if mask[d0, d1, d2]:
              assert np.array_equal(np.array(global_outputs[d0,d1,d2]), np.array([0,d0,d1,d2]))
  
  # Create dataset
  def do_test(dim0, dim1, dim2, ntr, nsubj, max_blk_edge, rad):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    mask = np.random.choice([True, False], (dim0,dim1,dim2))
    data = [np.empty((ntr,dim0,dim1,dim2), dtype=np.object) if i % size == rank else None for i in range(0, nsubj)]
    voxel_test(data, mask, max_blk_edge, rad)
    block_test(data, mask, max_blk_edge, rad)
  
  
  do_test(dim0=7, dim1=5, dim2=9, ntr=5, nsubj=1, max_blk_edge=4, rad=1)
  do_test(dim0=7, dim1=5, dim2=9, ntr=5, nsubj=5, max_blk_edge=4, rad=1)
  do_test(dim0=1, dim1=5, dim2=9, ntr=5, nsubj=5, max_blk_edge=4, rad=1)
  do_test(dim0=0, dim1=10, dim2=8, ntr=5, nsubj=5, max_blk_edge=4, rad=1)
  do_test(dim0=7, dim1=5, dim2=9, ntr=5, nsubj=1, max_blk_edge=4, rad=2)
  do_test(dim0=7, dim1=5, dim2=9, ntr=5, nsubj=1, max_blk_edge=4, rad=3)

