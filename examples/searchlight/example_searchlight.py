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

from brainiak.searchlight import example_fn
from brainiak.searchlight.searchlight import Searchlight

"""Distributed Searchlight Example
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
data = np.random.random((ntr,dim,dim,dim)) if rank == 0 else None
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
      data[idx][pt[0]:pt[0]+kernel_dim,pt[1]:pt[1]+kernel_dim,pt[2]:pt[2]+kernel_dim] += kernel * weight
    else:
      data[idx][pt[0]:pt[0]+kernel_dim,pt[1]:pt[1]+kernel_dim,pt[2]:pt[2]+kernel_dim] -= kernel * weight

# Create searchlight object
sl = Searchlight(sl_rad=1, max_blk_edge=5)

# Distribute data to processes
sl.distribute([data], mask)
sl.broadcast(labels)

# Define voxel function
def sfn(l, msk, myrad, bcast_var):
  import sklearn.svm
  import sklearn.model_selection
  classifier = sklearn.svm.SVC()
  data = l[0][:,msk]
  return np.mean(sklearn.model_selection.cross_val_score(classifier, data, bcast_var,n_jobs=1))

# Run searchlight
global_outputs = sl.searchlight_voxel(sfn)

# Visualize result
if rank == 0:
  print(global_outputs)
  global_outputs = np.array(global_outputs, dtype=np.float)
  import matplotlib.pyplot as plt
  for (cnt, img) in enumerate(global_outputs):
    plt.imshow(img,cmap='hot',vmin=0,vmax=1)
    plt.savefig('img' + str(cnt) + '.png')
    plt.clf()





