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

"""Distributed Searchlight Example
example usage: mpirun -n 4 python3 example_searchlight.py
"""

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Dataset size parameters
dim = 40
ntr = 400
maskrad = 15

# Predictive point parameters
pt = (23,23,23)
kernel_dim = 5
weight = 1

# Generate data
data = np.random.random((dim,dim,dim,ntr)) if rank == 0 else None
mask = np.zeros((dim,dim,dim), dtype=np.bool)
for i in range(dim):
  for j in range(dim):
    for k in range(dim):
      dist = np.sqrt(((dim/2)-i)**2 + ((dim/2)-j)**2 + ((dim/2)-k)**2)
      if(dist < maskrad):
        mask[i,j,k] = 1

# Generate labels
labels = np.random.choice([True, False], (ntr,)) if rank == 0 else None

# Inject predictive region in random data
if rank == 0:
  kernel = np.zeros((kernel_dim,kernel_dim,kernel_dim))
  for i in range(kernel_dim):
    for j in range(kernel_dim):
      for k in range(kernel_dim):
        arr = np.array([i-(kernel_dim/2),j-(kernel_dim/2),k-(kernel_dim/2)])
        kernel [i,j,k] = np.exp(-np.dot(arr.T,arr))
  kernel = kernel / np.sum(kernel)

  for (idx, l) in enumerate(labels):
    if l:
      data[pt[0]:pt[0]+kernel_dim,pt[1]:pt[1]+kernel_dim,pt[2]:pt[2]+kernel_dim,idx] += kernel * weight
    else:
      data[pt[0]:pt[0]+kernel_dim,pt[1]:pt[1]+kernel_dim,pt[2]:pt[2]+kernel_dim,idx] -= kernel * weight

# Create searchlight object
sl = Searchlight(sl_rad=1, max_blk_edge=5, shape=Diamond,
                 min_active_voxels_proportion=0)

# Distribute data to processes
sl.distribute([data], mask)
sl.broadcast(labels)

# Define voxel function
def sfn(l, msk, myrad, bcast_var):
  import sklearn.svm
  import sklearn.model_selection
  classifier = sklearn.svm.SVC()
  data = l[0][msk,:].T
  return np.mean(sklearn.model_selection.cross_val_score(classifier, data, bcast_var,n_jobs=1))

# Run searchlight
global_outputs = sl.run_searchlight(sfn)

# Visualize result
if rank == 0:
  print(global_outputs)
  global_outputs = np.array(global_outputs, dtype=np.float)
  import matplotlib.pyplot as plt
  for (cnt, img) in enumerate(global_outputs):
    plt.imshow(img,cmap='hot',vmin=0,vmax=1)
    plt.savefig('img' + str(cnt) + '.png')
    plt.clf()





