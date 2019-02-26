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

"""Distributed Multi-dataset multi-subject (MDMS) SRM analysis Example.

This example runs MDMS on time segment matching experiment. 
To get a better understanding of the code, please look at 
mdms_time_segment_matching_example.ipynb first.

Example Usage
-------
If run 4 ranks:
    $ mpirun -n 4 python3 mdms_time_segment_matching_distributed.py

Author
-------
Hejia Zhang (Princeton University ELE Department)

Notes
-------
It's an implementation of:
.. [Zhang2018] "Transfer learning on fMRI datasets",
   H. Zhang, P.-H. Chen, P. Ramadge
   The 21st International Conference on Artificial Intelligence and Statistics (AISTATS), 2018.
   http://proceedings.mlr.press/v84/zhang18b/zhang18b.pdf
"""

import numpy as np
from mpi4py import MPI
from scipy.stats import stats
import pickle as pkl
from brainiak.fcma.util import compute_correlation
from brainiak.funcalign.mdms import MDMS, Dataset


# parameters
features = 75 # number of features, k
n_iter = 30 # number of iterations of EM
test_ds = 'milky'

# MPI parameters, do not need to change
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
if rank == 0:
    print ('comm size : {}'.format(size))

# load and preprocess data in rank 0
if rank == 0:
    # load data
    with open('data/multi_dataset.pickle','rb') as f:
        all_data = pkl.load(f)

    # load dataset structure
    ds_struct = Dataset('data/multi_dataset.json')

    # separate train and test data
    # save info of test data to rank 0, and the testing will run at rank 0
    test_subj_list = ds_struct.subj_in_dataset[test_ds]
    test_data = all_data[test_ds]

    # remove test dataset from the dataset structure without changing the data and MDMS will handle it automatically
    _ = ds_struct.remove_dataset([test_ds])

    # remove subjects in test_ds that are not in any training dataset
    train_subj = set(ds_struct.get_subjects_list()) # all subjects in training set
    test_subj_idx_to_keep = [] # index of subjects to keep
    for idx, subj in enumerate(test_subj_list):
        if subj in train_subj:
            test_subj_idx_to_keep.append(idx)
    test_subj_list = [test_subj_list[idx] for idx in test_subj_idx_to_keep]
    test_data = [test_data[idx] for idx in test_subj_idx_to_keep]

    # compute voxels mean and std of each subject from training data and use them to standardize training and testing data
    mean, std = {}, {} # mean and std of each subject
    matrix_csr = ds_struct.matrix.tocsr(copy=True)
    for subj in range(ds_struct.num_subj): # iterate through all subjects
        subj_name = ds_struct.idx_to_subject[subj]
        indices = matrix_csr[subj,:].indices # indices of datasets with this subject
        # aggregate all data from this subject
        for idx, ds_idx in enumerate(indices):
            if idx == 0:
                mtx_tmp = all_data[ds_struct.idx_to_dataset[ds_idx]][ds_struct.dok_matrix[subj,ds_idx]-1]
            else:
                mtx_tmp = np.concatenate((mtx_tmp, all_data[ds_struct.idx_to_dataset[ds_idx]][ds_struct.dok_matrix[subj,ds_idx]-1]),axis=1)
        # compute mean and std
        mean[subj_name] = np.mean(mtx_tmp, axis=1)
        std[subj_name] = np.std(mtx_tmp, axis=1)
        # standardize training data
        for ds_idx in indices:
            ds_name, idx_in_ds = ds_struct.idx_to_dataset[ds_idx], ds_struct.dok_matrix[subj,ds_idx]-1
            all_data[ds_name][idx_in_ds] = np.nan_to_num((all_data[ds_name][idx_in_ds]-mean[subj_name][:,None])/std[subj_name][:,None])
            
    # use the mean and std computed from training data to standardize testing data
    for idx, subj in enumerate(test_subj_list):
        test_data[idx] = np.nan_to_num((test_data[idx]-mean[subj][:,None])/std[subj][:,None])

    # delete testing data from 'all_data' to save space
    del all_data[test_ds]

    # get the membership and compute the tag for MPI communication for every data point in 'all_data'
    data_mem = {}
    tag_s = 0 # tag start from 0
    for ds in all_data:
        length = len(all_data[ds])
        mem = np.random.randint(low=0,high=size,size=length) # which rank it belongs to
        tag = list(range(tag_s, tag_s+length))
        tag_s += length
        data_mem[ds] = [mem, tag]

else:
    ds_struct = None
    data_mem = None

# broadcast data_mem and ds_struct to all ranks and initialize data in each rank
data_mem = comm.bcast(data_mem, root=0)
ds_struct = comm.bcast(ds_struct, root=0)

data = {}
for ds in data_mem:
    data[ds] = [None]*len(data_mem[ds][0])

# distribute data
if rank == 0:
    for ds in data:
        for idx, (mem, tag) in enumerate(zip(data_mem[ds][0], data_mem[ds][1])):
            if mem != 0:
                comm.send(all_data[ds][idx], dest=mem, tag=tag)
            else:
                data[ds][idx] = all_data[ds][idx]
    del all_data
else:
    for ds in data:
        for idx, (mem, tag) in enumerate(zip(data_mem[ds][0], data_mem[ds][1])):
            if mem == rank:
                data[ds][idx] = comm.recv(source=0, tag=tag)

# Fit MDMS model
model = MDMS(features=features, n_iter=n_iter, comm=comm)
model.fit(data, ds_struct)

# run the testing in rank 0
if rank == 0:
    # define time segment matching experiment
    def time_segment_matching(data, win_size=6): 
        nsubjs = len(data)
        (ndim, nsample) = data[0].shape
        accu = np.zeros(shape=nsubjs)
        nseg = nsample - win_size 
        # mysseg prediction prediction
        trn_data = np.zeros((ndim*win_size, nseg),order='f')
        # the trn data also include the tst data, but will be subtracted when 
        # calculating A
        for m in range(nsubjs):
            for w in range(win_size):
                trn_data[w*ndim:(w+1)*ndim,:] += data[m][:,w:(w+nseg)]
        for tst_subj in range(nsubjs):
            tst_data = np.zeros((ndim*win_size, nseg),order='f')
            for w in range(win_size):
                tst_data[w*ndim:(w+1)*ndim,:] = data[tst_subj][:,w:(w+nseg)]

            A =  np.nan_to_num(stats.zscore((trn_data - tst_data),axis=0, ddof=1))
            B =  np.nan_to_num(stats.zscore(tst_data,axis=0, ddof=1))

            # compute correlation matrix
            corr_mtx = compute_correlation(B.T,A.T)
        
            for i in range(nseg):
                for j in range(nseg):
                    if abs(i-j)<win_size and i != j :
                        corr_mtx[i,j] = -np.inf
            max_idx =  np.argmax(corr_mtx, axis=1)
            accu[tst_subj] = sum(max_idx == range(nseg)) / float(nseg)

        return accu

    # transform testing data
    transformed = model.transform(test_data, test_subj_list) # test_subj_list: element i is the name of subject of X[i]

    # zscore the transformed data
    for subj in range(len(transformed)):
        transformed[subj] = stats.zscore(transformed[subj], axis=1, ddof=1)

    # run the experiment
    accu = time_segment_matching(transformed)
    accu_mean = np.mean(accu)
    accu_se = stats.sem(accu)
    print ('Accuracy is {} +- {}'.format(accu_mean, accu_se))

