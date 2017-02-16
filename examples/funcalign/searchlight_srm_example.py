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

# Author: Hejia Zhang (Princeton University ELE Department)

"""Distributed Searchlight Example (if run 16 ranks)
example usage: mpirun -n 16 python3 searchlight_srm_example.py
"""

import numpy as np
from mpi4py import MPI
import sys
import scipy.io as sio
from scipy.stats import stats
from brainiak.searchlight.searchlight import Searchlight
from brainiak.funcalign.srm import SRM
import warnings

# parameters
sl_rad = 1 #searchlight length (of each edge) will be 1+2*sl_rad
max_blk_edge = 1 #won't change computational results, has effect on MPI processing
nfeature = 10 #number of features in SRM for each searchlight
n_iter = 10 #number of interations in SRM

# sanity check
if sl_rad <= 0 or max_blk_edge <= 0:
  raise ValueError('sl_rad and max_blk_edge must be positive')
if nfeature > (1+2*sl_rad)**3:
  print ('nfeature truncated')
  nfeature = int((1+2*sl_rad)**3)

# MPI parameters, do not need to change
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# load data
movie_file = sio.loadmat('data/sl_movie_data.mat')
movie_data = movie_file['data']

# Dataset size parameters
dim1,dim2,dim3,ntr,nsubj = movie_data.shape

# preprocess data, zscore and set NaN to 0
all_data = [] # first half train, second half test
for s in range(nsubj):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    train_tmp = np.nan_to_num(stats.zscore(movie_data[:,:,:,:int(ntr/2),s],axis=3,ddof=1))
    test_tmp = np.nan_to_num(stats.zscore(movie_data[:,:,:,int(ntr/2):,s],axis=3,ddof=1))
  all_data.append(np.concatenate((train_tmp,test_tmp),axis=3))

# Generate mask: mask is a 3D binary array, with active voxels being 1. I simply set 
# all voxels to be active in this example, but you should set the mask to fit your ROI
# in practice.
mask = np.ones((dim1,dim2,dim3), dtype=np.bool)

# Create searchlight object
sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)

# Distribute data to processes
# the first argument of "distribute" is a list of 4D arrays, and each 4D array is data 
# from a single subject
sl.distribute(all_data, mask)
# broadcast something that should be shared by all ranks 
sl.broadcast([n_iter,nfeature])

# time segment matching experiment. Define your own experiment function here
def timesegmentmatching_accuracy(data, win_size=6): 
  nsubjs = len(data)
  (ndim, nsample) = data[0].shape
  accu = np.zeros(shape=nsubjs)
  nseg = nsample - win_size 
  # mysseg prediction prediction
  trn_data = np.zeros((ndim*win_size, nseg))
  # the trn data also include the tst data, but will be subtracted when 
  # calculating A
  for m in range(nsubjs):
      for w in range(win_size):
          trn_data[w*ndim:(w+1)*ndim,:] += data[m][:,w:(w+nseg)]
  for tst_subj in range(nsubjs):
    tst_data = np.zeros((ndim*win_size, nseg))
    for w in range(win_size):
      tst_data[w*ndim:(w+1)*ndim,:] = data[tst_subj][:,w:(w+nseg)]

    A =  np.nan_to_num(stats.zscore((trn_data - tst_data),axis=0, ddof=1))
    B =  np.nan_to_num(stats.zscore(tst_data,axis=0, ddof=1))
    # normalize A and B
    A = A/np.linalg.norm(A,axis=0)
    B = B/np.linalg.norm(B,axis=0)
    corr_mtx = B.T.dot(A)
    for i in range(nseg):
      for j in range(nseg):
        if abs(i-j)<win_size and i != j :
          corr_mtx[i,j] = -np.inf
    max_idx =  np.argmax(corr_mtx, axis=1)
    accu[tst_subj] = sum(max_idx == range(nseg)) / float(nseg)

  return accu

# Define voxel function: The function to be applied on each searchlight
def sfn(l, msk, myrad, bcast_var):
  # Arguments:
  # l -- a list of 4D arrays, containing data from a single searchlight
  # msk -- a 3D binary array, mask of this searchlight
  # myrad -- an integer, sl_rad
  # bcast_var -- whatever is broadcasted 

  # extract training and testing data
  train_data = []
  test_data = []
  d1,d2,d3,ntr = l[0].shape
  nvx = d1*d2*d3
  for s in l:
    train_data.append(np.reshape(s[:,:,:,:int(ntr/2)],(nvx,int(ntr/2))))
    test_data.append(np.reshape(s[:,:,:,int(ntr/2):],(nvx,ntr-int(ntr/2))))
  # train an srm model 
  srm = SRM(bcast_var[0],bcast_var[1])
  srm.fit(train_data)
  # transform test data
  shared_data = srm.transform(test_data)
  for s in range(len(l)):
    shared_data[s] = np.nan_to_num(stats.zscore(shared_data[s],axis=1,ddof=1))
  # run experiment
  accu = timesegmentmatching_accuracy(shared_data,6)

  # return: can also return several values. In that case, the final output will be 
  # a 3D array of tuples
  return np.mean(accu) 

# Run searchlight
acc = sl.run_searchlight(sfn) # output is a 3D array in shape (dim1,dim2,dim3)

# save result
if rank == 0:
  print (acc)
  np.savez_compressed('data/searchlight_srm_tsm_acc.npz',acc=acc)  




