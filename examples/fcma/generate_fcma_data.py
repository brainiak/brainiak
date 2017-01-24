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

"""Generate simulated data for FCMA example

Generate example data for FCMA analyses that uses fmrisim

 Authors: Cameron Ellis (Princeton) 2017
"""
import logging
import numpy as np
from brainiak.utils import fmrisim as sim
import nibabel
import os

logger = logging.getLogger(__name__)


# Set the experimental parameters
participants = 5
epochs = 5
dimensions = np.array([10, 10, 10])  # What is the size of the brain

# Where will the data be stored?
directory = 'examples/fcma/simulated/'

# Make the directory if it hasn't been made yet
if os.path.isdir(directory) is False:
    os.mkdir(directory)

# Prepare the feature attributes
feature_size = [2]
feature_type = ['cube']
coordinates_a = np.array(
    [[3, 3, 2], [3, 5, 4]])
coordinates_b = np.array(
    [[6, 4, 2], [6, 5, 4]])
signal_magnitude = [1]

# Inputs for generate_stimfunction
onsets_a = []
onsets_b = []
weights = []
tr_duration = 1
event_durations = [1]
duration = 100

# Create the epoch cube
epoch = np.zeros([2, epochs * 2, int(duration / tr_duration)], np.int8,
                 order='C')

for idx in list(range(0, epochs)):
    onsets_a += list(range((idx * 20) + 10, (idx * 20) + 17, tr_duration))
    onsets_b += list(range(idx * 20, (idx * 20) + 7, tr_duration))

    epoch[0, idx * 2, ((idx * 20) + 10):((idx * 20) + 17)] = 1
    epoch[1, ((idx * 2) + 1), (idx * 20):((idx * 20) + 7)] = 1

    # The pattern of activity for each trial
    weights += [0.5, 1.0, 1.0, 0.0, -1.0, -1.0, -0.5]

# Generate a volume representing the location and quality of the signal
volume_static_a = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=coordinates_a,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )

volume_static_b = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=coordinates_b,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )

# Create the time course for the signal to be generated
stimfunction_a = sim.generate_stimfunction(onsets=onsets_a,
                                           event_durations=event_durations,
                                           total_time=duration,
                                           weights=weights,
                                           )

stimfunction_b = sim.generate_stimfunction(onsets=onsets_b,
                                           event_durations=event_durations,
                                           total_time=duration,
                                           weights=weights,
                                           )

# Convolve the HRF with the stimulus sequence
signal_function_a = sim.double_gamma_hrf(stimfunction=stimfunction_a,
                                         tr_duration=tr_duration,
                                         )

signal_function_b = sim.double_gamma_hrf(stimfunction=stimfunction_b,
                                         tr_duration=tr_duration,
                                         )

# Multiply the HRF timecourse with the signal
signal_a = sim.apply_signal(signal_function=signal_function_a,
                            volume_static=volume_static_a,
                            )

signal_b = sim.apply_signal(signal_function=signal_function_b,
                            volume_static=volume_static_b,
                            )

# Combine the signals from the two conditions
signal = signal_a + signal_b

# Combine the stim functions
stimfunction = list(np.add(stimfunction_a, stimfunction_b))

# Generate the mask of the signal
mask = sim.mask_brain(signal)

# Mask the signal to the shape of a brain (attenuates signal according
# to grey matter likelihood)
signal *= mask

# Iterate through the participants and store participants
epochs = []
for participantcounter in list(range(1, participants + 1)):

    # Add the epoch cube
    epochs += [epoch]

    # Save a file name
    savename = directory + 'p' + str(participantcounter) + '.nii'

    # Create the noise volumes (using the default parameters
    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction=stimfunction,
                               tr_duration=tr_duration,
                               mask=mask,
                               )

    # Combine the signal and the noise
    brain = signal + noise

    # Save the volume
    affine_matrix = np.diag([-1, 1, 1, 1])  # LR gets flipped
    brain_nifti = nibabel.Nifti1Image(brain, affine_matrix)
    nibabel.save(brain_nifti, savename)

# Save the epochs
np.save(directory + 'epoch_labels.npy', epochs)

# Store the mask
mask[mask > 0] = 1
mask = mask[:, :, :, 0]
brain_nifti = nibabel.Nifti1Image(mask, affine_matrix)
nibabel.save(brain_nifti, directory + 'mask.nii')
