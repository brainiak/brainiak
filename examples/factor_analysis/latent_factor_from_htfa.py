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
import scipy.io
from scipy.stats import stats
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

n_subj = 2
data_dir = os.getcwd() + '/data'
url = []
url.append(' https://www.dropbox.com/s/r5s9tg4ekxzbrco/s0.mat?dl=0')
url.append(' https://www.dropbox.com/s/39tr01m76vxwaqa/s1.mat?dl=0')

#get fMRI data and scanner RAS coordinates
data = []
R = []
for idx in range(n_subj):
    if idx % size == rank:
        #download data
        file_name = os.path.join(data_dir, 's' + str(idx) + '.mat')
        cmd = 'curl --location --create-dirs -o ' + file_name + url[idx]
        os.system(cmd)
        all_data = scipy.io.loadmat(file_name)
        bold = all_data['data']
        # z-score the data
        bold = stats.zscore(bold, axis=1, ddof=1)
        data.append(bold)
        R.append(all_data['R'])

n_voxel, n_tr = data[0].shape

# Run HTFA with downloaded data
from brainiak.factor_analysis.htfa import HTFA
help(HTFA)

K = 5
htfa = HTFA(K=K,
        max_global_iter=5,
        max_local_iter=2,
        voxel_ratio=0.5,
        tr_ratio=0.5,
        max_voxel=n_voxel,
        max_tr=n_tr,
        verbose=True)
htfa.fit(data, R)

if rank == 0:
    print("\n centers of global latent factors are:")
    print(htfa.get_centers(htfa.global_posterior_))
    print("\n widths of global latent factors are:")
    widths = htfa.get_widths(htfa.global_posterior_)
    print(widths)
    print("\n stds of global latent RBF factors are:")
    rbf_std = np.sqrt(widths/(2.0))
    print(rbf_std)
