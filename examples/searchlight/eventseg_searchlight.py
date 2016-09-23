#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#               http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import brainiak.eventseg.event
import brainiak.searchlight.searchlight
import matplotlib.pyplot as plt
from mpi4py import MPI

""" Distributed searchlight for event segmentation.
Creates random data and fits the event segmentation
model on each 3x3 voxel region. 
Then tests on more random data and plots
the results. 

Citation: Discovering event structure in continuous narrative perception 
and memory Christopher Baldassano, Janice Chen, Asieh Zadbood, 
Jonathan W Pillow, Uri Hasson, Kenneth A Norman
"""


# Configuration
num_events = 10
trs_per_event = 10
num_trs = num_events*trs_per_event
num_voxels = (5,5,5)
train_data_noise = 0.1
test_data_noise = 0.1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Rank 0 generates random data
slData = None
if rank == 0:
    # sample labels
    labels = np.repeat(range(0,num_events),trs_per_event)

    # sample data
    def randomTR(seed, nvoxels):
        np.random.seed(seed)
        return np.random.random(nvoxels)

    simData = [randomTR(seed,num_voxels) for seed in range(0, num_events)]

    def noisyTR(tr, noise):
        return (1.0-noise) * tr + noise * np.random.random(tr.shape)

    probData = np.array([noisyTR(simData[label], train_data_noise) for label in labels])
    testData = np.array([noisyTR(simData[label], test_data_noise) for label in labels])

    slData = [probData, testData]

# Searchlight function fits training data and tests on test data
def fitFn(a, mask, bcast_var):

    # Create EventSegmentation class
    es = brainiak.eventseg.event.EventSegment(num_events)

    # Flatten train and test data
    trainData = a[0][:,mask]
    testData = a[1][:,mask]

    # Train and test
    es.fit(trainData)

    # Return test predictions
    gamma, LL = es.find_events(testData)
    return np.argmax(gamma, axis=1)

# Create searchlight
sl = brainiak.searchlight.searchlight.Searchlight(1, fitFn)

# Run searchlight
output = sl.fit_transform((slData, np.ones(num_voxels, dtype=np.bool)))

# Plot the results
if rank == 0:
    plt.figure()
    plt.subplot(1, 1, 1)

    for i in range(1, num_voxels[0]-1):
        for j in range(1, num_voxels[1]-1):
            for k in range(1, num_voxels[2]-1):
                plt.plot(output[i,j,k])

    plt.xlabel('Timepoints')
    plt.ylabel('Event label')
    plt.savefig('fig.png')
    plt.show()
