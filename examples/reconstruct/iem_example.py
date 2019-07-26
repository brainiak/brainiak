#  Copyright 2018 David Huberdeau & Peter Kok
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

"""

    Inverted Encoding Model (IEM) Test with simulated data.

    This example uses simulated data and the brainiak IEM implementation
    to train an inverted encoding model, and then test reconstruction
    from new simulated data.

    To simulate the data, it is assumed that some feature is parameterized
    along one dimension (e.g. this could be the appearance of a target at
    four different locations along a horizontal line across someone's
    visual field). It is assumed that there are N voxels that are each
    selective to some weighting of these features. Four basis functions
    span the feature space, and each voxel contains a unique weighting of
    basis functions. Functional activity is simulated from each voxel by
    sampling from the basis functions weighted by that voxel's unique
    weight vector. (See README for more detailed description of how this
    data is simulated.)

    This example then trains and tests the IEM implementation given the
    simulated data.

    author: David Huberdeau

"""

import numpy as np
import brainiak.reconstruct.iem
import logging

logger = logging.getLogger(__name__)

# Define feature space, target locations, and basis functions
k_exponent = 4
feature_space = list(range(1, 150, 1))
targets = list(range(30, 121, 30))
basis_fcns = [[0]*len(feature_space)]*len(targets)
for i_func in range(0, len(targets)):
    x_ax = list(map(lambda x:x - targets[i_func], feature_space))
    this_fcn = list(map(lambda x:pow(x,k_exponent),
                         np.cos(np.deg2rad(x_ax))))
    basis_fcns[i_func] = this_fcn

# Define voxel weights for basis functions
n,w = 5,4 # n - defines dimensions of ROI (assumed an nxn square
# of voxels), w - number of weights
weight_fcn = np.linspace(0, 1, n)
voxel_def = []
roll_amt_1 = list(range(0, w))
roll_amt_2 = list(range(0, n))
for i_wt in range(0, w):
    this_wt_fcn = np.roll(weight_fcn, roll_amt_1[i_wt])
    this_voxel_list = []
    for i_vox1 in range(0, n):
        this_vox_fcn = list(np.roll(this_wt_fcn, roll_amt_2[i_vox1]))
        this_voxel_list.append(this_vox_fcn)
    voxel_def.append(this_voxel_list)

# define function that samples from voxels given the weights
def sample_voxel_activations(stimulus_list, s_noise):
    # function input:
        # stimulus_list - 1-dimensional list of stimulus categories
        # n - number of samples per stimulus desired
        # s_noise - the standard deviation of the gaussian noise
    voxel_sample = []
    for i_stim in range(0, len(stimulus_list)):
        for i_vox1 in range(0, n):
            for i_vox2 in range(0, n):
                voxel_sample.append(
                    voxel_def[stimulus_list[i_stim]][i_vox1][i_vox2]\
                    *basis_fcns[stimulus_list[i_stim]]\
                        [targets[stimulus_list[i_stim]]]\
                    + s_noise*np.random.normal())

    voxel_sample_out = \
        np.reshape(voxel_sample, (len(stimulus_list), n*n))
    return voxel_sample_out

# Simulate a sample of voxel activations for training:
s_noise = 0.1
n_samples_per_stim = 10
stimulus_list_ = np.matlib.repmat(list(range(0, w)), 1,
                                 n_samples_per_stim)
stim_list_train = stimulus_list_[0]
voxel_sample_train = sample_voxel_activations(
    stim_list_train, s_noise)

stim_direction_train = []
for i_stim in range(0,len(stim_list_train)):
    stim_direction_train.append(targets[stim_list_train[i_stim]])

# Create IEM object
Invt_model = brainiak.reconstruct.iem.InvertedEncoding(4, # channels
                                       4, # exponent
                                       0, # initital feature value
                                       150) # final feature value

# Train IEM object
Invt_model.fit(voxel_sample_train, stim_direction_train)

# Simulate a sample of voxels activations for testing:
s_noise = 0.1
n_samples_per_stim = 10
stimulus_list_ = np.matlib.repmat(list(range(0, w)), 1,
                                 n_samples_per_stim)
stim_list_test = stimulus_list_[0]
voxel_sample_test = sample_voxel_activations(
    stim_list_train, s_noise)

stim_direction_test = []
for i_stim in range(0, len(stim_list_test)):
    stim_direction_test.append(targets[stim_list_test[i_stim]])

# Test reconstruction with simulated test voxels:
c_predict = Invt_model._predict_channel_responses(voxel_sample_test)
d_predict = Invt_model._predict_direction_responses(voxel_sample_test)
s_predict = Invt_model.predict(voxel_sample_test)
score = Invt_model.score(voxel_sample_test, stim_direction_test)

logger.info('Scores: ' + str(score))