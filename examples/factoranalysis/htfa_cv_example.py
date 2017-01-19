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
import os
import sys
import math
import requests 
import scipy.io
import numpy as np
from mpi4py import MPI
from subprocess import call
from scipy.stats import stats
from sklearn import model_selection 
from sklearn.metrics import mean_squared_error

def recon_err(data, F, W):
    """Calcuate reconstruction error

    Parameters
    ----------

    data : 2D array 
        True data to recover.

    F : 2D array 
        HTFA factor matrix.

    W : 2D array 
        HTFA weight matrix.
   

    Returns
    -------
    float 
        Returns root mean squared reconstruction error.

    """

    recon = F.dot(W).ravel()
    err = mean_squared_error(
        data.ravel(),
        recon,
        multioutput='uniform_average')
    return math.sqrt(err)

def get_train_err(htfa, data, F):
    """Calcuate training error

    Parameters
    ----------

    htfa : HTFA
        An instance of HTFA, factor anaysis class in BrainIAK.
   
    data : 2D array 
        Input data to HTFA.

    F : 2D array 
        HTFA factor matrix.
        
    Returns
    -------
    float 
        Returns root mean squared error on training.

    """

    W = htfa.get_weights(data, F)
    return recon_err(data, F, W)

def get_test_err(htfa, test_weight_data, test_recon_data,
        test_weight_R, test_recon_R, centers, widths):

    """Calcuate test error
    
    Parameters
    ----------

    htfa : HTFA
        An instance of HTFA, factor anaysis class in BrainIAK.
   
    test_weigth_data : 2D array 
        Data used for testing weights.
    
    test_recon_data : 2D array 
        Data used for testing reconstruction error.

    test_weigth_R : 2D array 
        Coordinate matrix used for testing weights.
    
    test_recon_R : 2D array 
        Coordinate matrix used for testing reconstruction error.

    centers : 2D array 
        Center matrix of HTFA factors.
    
    widths : 1D array 
        Width matrix of HTFA factors.
        
    Returns
    -------
    float 
        Returns root mean squared error on test.

    """

    # calculate F on test_weight_R, based on trained centers/widths
    unique_R, inds = htfa.get_unique_R(test_weight_R)
    F = htfa.get_factors(unique_R,
                         inds,
                         centers,
                         widths)
    # calculate weights on test_weight_data
    W = htfa.get_weights(test_weight_data, F)

    # calculate F on final test_recon_data
    unique_R, inds = htfa.get_unique_R(test_recon_R)
    F = htfa.get_factors(unique_R,
                         inds,
                         centers,
                         widths)
    return recon_err(test_recon_data, F, W)

n_subj = 2
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
group_id = int(rank/n_subj)
n_group = math.ceil(size/n_subj)
htfa_comm = comm.Split(group_id, rank)
htfa_rank = htfa_comm.Get_rank()
htfa_size = htfa_comm.Get_size()

if rank == 0:
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

data_dir = os.path.join(os.getcwd(), 'data')
if rank == 0 and not os.path.exists(data_dir):
    os.makedirs(data_dir)

url = []
url.append(' https://www.dropbox.com/s/r5s9tg4ekxzbrco/s0.mat?dl=0')
url.append(' https://www.dropbox.com/s/39tr01m76vxwaqa/s1.mat?dl=0')

for idx in range(n_subj):
    if idx % size == rank:
        file_name = os.path.join(data_dir, 's' + str(idx) + '.mat')
        #check if file has already been downloaded
        if not os.path.exists(file_name):
            #check if URL exists
            ret = requests.head(url[idx])
            if ret.status_code == 200:
                #download data
                cmd = 'curl --location -o ' + file_name + url[idx]
                try:
                    retcode = call(cmd, shell=True)
                    if retcode < 0:
                        print("File download was terminated by signal", -retcode, file=sys.stderr)
                    else:
                        print("File download returned", retcode, file=sys.stderr)
                except OSError as e:
                    print("File download failed:", e, file=sys.stderr)
            else:
                 print("File s%d.mat does not exist!\n"%idx)

comm.Barrier()

#get fMRI data and scanner RAS coordinates
data = []
R = []
mapping = {}
n_local_subj = 0
for idx in range(n_subj):
    if idx % htfa_size == htfa_rank:
        file_name = os.path.join(data_dir, 's' + str(idx) + '.mat')
        all_data = scipy.io.loadmat(file_name)
        bold = all_data['data']
        # z-score the data
        bold = stats.zscore(bold, axis=1, ddof=1)
        data.append(bold)
        R.append(all_data['R'])
        mapping[str(n_local_subj)] = idx
        n_local_subj += 1


min_K = 3
max_K = 6
n_K = 2
Ks  = np.linspace(min_K, max_K, n_K, endpoint=True).astype(int)
n_splits = 3
# recon_err in shape n_splits*n_K
test_recon_errs = np.zeros((n_subj, n_splits, n_K))
tmp_test_recon_errs = np.zeros((n_subj, n_splits, n_K))
train_recon_errs = np.zeros((n_subj, n_splits, n_K))
tmp_train_recon_errs = np.zeros((n_subj, n_splits, n_K))

local_size = math.ceil(n_subj/size)
if n_local_subj > 0:
    from brainiak.factoranalysis.htfa import HTFA
    n_voxel, n_tr = data[0].shape
    n_dim = R[0].shape[1]
    test_size = 0.3
    rnd_seed_voxel = 30000
    rnd_seed_tr = 3000
    tr_solver = 'exact'
    nlss_method = 'dogbox'
    nlss_loss = 'linear'
    upper_ratio = 1.8
    lower_ratio = 0.1
    voxel_ratio = 0.25
    tr_ratio = 0.1
    max_voxel = 2000
    max_tr = 200
    max_sample_voxel = min(max_voxel,
                          int(voxel_ratio * n_voxel))
    max_sample_tr = min(max_tr, int(tr_ratio * n_tr))

    #split voxel and TR for two-level cross validation
    ss_voxel = model_selection.ShuffleSplit(
                                     n_splits=n_splits,
                                     test_size=test_size,
                                     random_state=rnd_seed_voxel)
    voxel_indices = np.arange(n_voxel)
    ss_voxel.get_n_splits(voxel_indices)

    ss_tr = model_selection.ShuffleSplit(
                                     n_splits=n_splits,
                                     test_size=test_size,
                                     random_state=rnd_seed_tr)
    tr_indices = np.arange(n_tr)
    ss_tr.get_n_splits(tr_indices)

    train_voxels = []
    test_voxels = []
    train_trs = []
    test_trs = []
    for train_index, test_index in ss_voxel.split(voxel_indices):
        train_voxels.append(train_index)
        test_voxels.append(test_index)

    for train_index, test_index in ss_tr.split(tr_indices):
        train_trs.append(train_index)
        test_trs.append(test_index)


    for p in range(n_splits):
        for idx in range(n_K):
            index = p*n_K + idx
            if index % n_group == group_id:
                #split data and R
                train_voxel_indices = train_voxels[p]
                test_voxel_indices = test_voxels[p]
                train_tr_indices = train_trs[p]
                test_tr_indices = test_trs[p]

                train_data = []
                total_test_data = []
                test_weight_data = []
                test_recon_data = []
                test_weight_R = []
                test_recon_R = []
                for s in range(n_local_subj):
                    train_data.append(data[s][:, train_tr_indices])
                    total_test_data.append(data[s][:, test_tr_indices])
                    test_weight_data.append(
                            total_test_data[s][train_voxel_indices, :])
                    test_recon_data.append(
                            total_test_data[s][test_voxel_indices, :])
                    test_weight_R.append(R[s][train_voxel_indices])
                    test_recon_R.append(R[s][test_voxel_indices])

                htfa = HTFA(K=Ks[idx],
                        max_global_iter=5,
                        max_local_iter=2,
                        n_subj=n_subj,
                        nlss_method=nlss_method,
                        nlss_loss=nlss_loss,
                        tr_solver=tr_solver,
                        upper_ratio=upper_ratio,
                        lower_ratio=lower_ratio,
                        max_tr=max_sample_tr,
                        max_voxel=max_sample_voxel,
                        comm=htfa_comm,
                        verbose=True)
                htfa.fit(train_data, R)

                for s in range(n_local_subj):
                    #get posterior for each subject
                    subj_idx = mapping[str(s)]
                    start_idx = s * htfa.prior_size
                    end_idx = (s + 1) * htfa.prior_size
                    local_posteiror = htfa.local_posterior_[start_idx:end_idx]
                    local_centers = htfa.get_centers(local_posteiror)
                    local_widths = htfa.get_widths(local_posteiror)

                    htfa.n_dim = n_dim
                    htfa.cov_vec_size = np.sum(np.arange(htfa.n_dim) + 1)
                    htfa.map_offset = htfa.get_map_offset()
                    #training happens on all voxels, but part of TRs
                    unique_R_all, inds_all = htfa.get_unique_R(R[s])
                    train_F = htfa.get_factors(unique_R_all,
                                             inds_all,
                                             local_centers,
                                             local_widths)

                    #calculate train_recon_err
                    tmp_train_recon_errs[subj_idx, p,idx] = get_train_err(htfa,
                                                         train_data[s],
                                                         train_F)

                    #calculate weights on test_weight_data, test_recon_err on test_recon_data
                    tmp_test_recon_errs[subj_idx, p,idx] = get_test_err(htfa,
                                                        test_weight_data[s],
                                                        test_recon_data[s],
                                                        test_weight_R[s],
                                                        test_recon_R[s],
                                                        local_centers,
                                                        local_widths)

comm.Reduce(tmp_test_recon_errs, test_recon_errs, op=MPI.SUM)
comm.Reduce(tmp_train_recon_errs, train_recon_errs, op=MPI.SUM)

if rank == 0:
    errs = train_recon_errs.reshape(n_subj * n_splits, n_K)
    mean_errs = np.average(errs, axis=0)
    print("train error on each K is\n")
    print(mean_errs)
    errs = test_recon_errs.reshape(n_subj * n_splits, n_K)
    mean_errs = np.average(errs, axis=0)
    print("test error on each K is\n")
    print(mean_errs)
    best_idx = np.argmin(mean_errs)
    print("best K for test recon is %d " % (Ks[best_idx]))
