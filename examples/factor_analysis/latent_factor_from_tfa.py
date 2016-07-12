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
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#download data
data_dir = os.getcwd() + '/data'
file_name = os.path.join(data_dir, 's1.mat')
url = ' https://www.dropbox.com/s/r5s9tg4ekxzbrco/s0.mat?dl=0'
url = ' https://www.dropbox.com/s/39tr01m76vxwaqa/s1.mat?dl=0'
cmd = 'curl --location --create-dirs -o ' + file_name + url
os.system(cmd)

#get fMRI data and scanner RAS coordinates
all_data = scipy.io.loadmat(file_name)
data = all_data['data']
R = all_data['R']

# Z-score the data
data = stats.zscore(data, axis=1, ddof=1)
n_voxel, n_tr = data.shape

# Run TFA with downloaded data
from brainiak.factor_analysis.tfa import TFA
help(TFA)

tfa = TFA(K=5,
        max_num_voxel=int(n_voxel*0.5),
        max_num_tr=int(n_tr*0.5),
        verbose=True)
tfa.fit(data, R)

print("\n centers of latent factors are:")
print(tfa.get_centers(tfa.local_posterior_))
print("\n widths of latent factors are:")
widths = tfa.get_widths(tfa.local_posterior_)
print(widths)
print("\n stds of latent RBF factors are:")
rbf_std = np.sqrt(widths/(2.0))
print(rbf_std)

