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

logger = logging.getLogger(__name__)

# Set the experimental parameters
participants = 5
epochs = 5
dimensions = np.array([10, 10, 10])  # What is the size of the brain

# Prepare the feature attributes
feature_size = [2]
feature_type = ['cube']
coordinates_A = np.array(
    [[3, 3, 3], [5, 5, 3]])
coordinates_B = np.array(
    [[6, 6, 6], [4, 4, 6]])
signal_magnitude = [1]

# Inputs for generate_stimfunction
onsets_A = []
onsets_B = []
weights = []
tr_duration = 1

for idx in list(range(0, epochs)):
    onsets_A += list(range((idx * 20) + 10, (idx * 20) + 17, tr_duration))
    onsets_B += list(range(idx * 20, (idx * 20) + 7, tr_duration))
    # The pattern of activity for each trial
    weights += [0.5, 1.0, 1.0, 0.0, -1.0, -1.0, -0.5]

event_durations = [1]
duration = 100

# Generate a volume representing the location and quality of the signal
volume_static_A = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=coordinates_A,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )

volume_static_B = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=coordinates_B,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )

# Create the time course for the signal to be generated
stimfunction_A = sim.generate_stimfunction(onsets=onsets_A,
                                           event_durations=event_durations,
                                           total_time=duration,
                                           weights=weights,
                                           )

stimfunction_B = sim.generate_stimfunction(onsets=onsets_B,
                                           event_durations=event_durations,
                                           total_time=duration,
                                           weights=weights,
                                           )

# Convolve the HRF with the stimulus sequence
signal_function_A = sim.double_gamma_hrf(stimfunction=stimfunction_A,
                                         tr_duration=tr_duration,
                                         )

signal_function_B = sim.double_gamma_hrf(stimfunction=stimfunction_B,
                                         tr_duration=tr_duration,
                                         )

# Multiply the HRF timecourse with the signal
signal_A = sim.apply_signal(signal_function=signal_function_A,
                            volume_static=volume_static_A,
                            )

signal_B = sim.apply_signal(signal_function=signal_function_B,
                            volume_static=volume_static_B,
                            )

# Combine the signals from the two conditions
signal = signal_A + signal_B

# Combine the stim functions
stimfunction = list(np.add(stimfunction_A, stimfunction_B))

# Generate the mask of the signal
mask = sim.mask_brain(signal)

# Mask the signal to the shape of a brain (attenuates signal according to grey
# matter likelihood)
signal *= mask

# Iterate through the participants and store participants
for participantcounter in list(range(1, participants + 1)):

    # Save a file name
    savename = 'examples/fcma/p' + str(participantcounter) + '.nii'

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
