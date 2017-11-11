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

Generate example data for FCMA analyses that uses fmrisim.

This creates two conditions, a and b, with 5 trials for each condtion and
5 participants total. Each trial is 10 TRs long with 7 TRs of stimulation.
Brains are extremely downsampled (10 voxels cubed) in order to help
processing speed.

The signal that discriminates these conditions is such that one region
responds to both conditions whereas another region responds differently to
the two conditions. For instance, imagine voxel X responds to common
activation but voxel A only responds in condition A and voxel B only
responds in condition B.

 Authors: Cameron Ellis (Princeton) 2017
"""
import logging
import numpy as np
from brainiak.utils import fmrisim as sim
import nibabel
import os

logger = logging.getLogger(__name__)


# Default experimental parameters (These can all be changed (will affect
# processing time)
participants = 5  # How many participants are being created
epochs = 5  # How many trials
dimensions = np.array([10, 10, 10])  # What is the size of the brain
stim_dur = 7  # How long is each stimulation period
rest_dur = 3  # How long is the rest between stimulation
conds = 2  # How many conditions are there
fcma_better = 1  # this data is made so fcma will succeed and mvpa will fail

# Where will the data be stored?
directory = 'simulated/'

# Make the directory if it hasn't been made yet
if os.path.isdir(directory) is False:
    os.mkdir(directory)

# Prepare the feature attributes
feature_size = [1]
feature_type = ['cube']
coordinates=[]
coordinates += [np.array(
    [[3, 3, 3], [3, 5, 4]])]
coordinates += [np.array(
    [[3, 3, 3], [6, 5, 4]])]
coordinates += [np.array(
    [[3, 3, 3], [5, 6, 4]])]
signal_magnitude = [1]  # How big is the signal (in SD)


# Inputs for generate_stimfunction
onsets = list(range(conds))
weights = list(range(conds))
tr_duration = 2
event_durations = [tr_duration]
trial_dur = stim_dur + rest_dur
duration = epochs * trial_dur * conds

# Create the epoch cube
epoch = np.zeros([conds, epochs * conds, int(duration / tr_duration)], np.int8,
                 order='C')

# Iterate through the epochs and conditions
for cond_counter in list(range(conds)):

    onsets[cond_counter] = []  # Add a list for this condition
    weights[cond_counter] = []
    for idx in list(range(0, epochs)):

        # When does each epoch start and end
        start_idx = (idx * trial_dur * conds) + (trial_dur * cond_counter)
        end_idx = start_idx + stim_dur

        # Store these start and end times
        onsets[cond_counter] += list(range(start_idx, end_idx, tr_duration))
        epoch[cond_counter, idx * conds + cond_counter, start_idx:end_idx] = 1

        # The pattern of activity for each trial
        weight = ([1] * int(np.floor(stim_dur / 2))) + ([-1]*int(np.ceil(
            stim_dur / 2)))
        weights[cond_counter] += weight

# Iterate through the conditions to make the necessary functions
for cond in list(range(conds)):

    # Generate a volume representing the location and quality of the signal
    volume_signal = sim.generate_signal(dimensions=dimensions,
                                        feature_coordinates=coordinates[cond],
                                        feature_type=feature_type,
                                        feature_size=feature_size,
                                        signal_magnitude=signal_magnitude,
                                        )

    # Create the time course for the signal to be generated
    stimfunction_cond = sim.generate_stimfunction(onsets=onsets[cond],
                                                  event_durations=
                                                  event_durations,
                                                  total_time=duration,
                                                  weights=weights[cond],
                                                  )

    # Convolve the HRF with the stimulus sequence
    signal_function = sim.double_gamma_hrf(stimfunction=stimfunction_cond,
                                           tr_duration=tr_duration,
                                           )

    # Multiply the HRF timecourse with the signal
    signal_cond = sim.apply_signal(signal_function=signal_function,
                                   volume_signal=volume_signal,
                                   )

    # Concatenate all the signal and function files
    if cond == 0:
        stimfunction = stimfunction_cond
        signal = signal_cond
    else:
        stimfunction = list(np.add(stimfunction, stimfunction_cond))
        signal += signal_cond

# Generate the mask of the signal
mask, template = sim.mask_brain(signal)

# Mask the signal to the shape of a brain (does not attenuate signal according
# to grey matter likelihood)
signal *= mask.reshape(dimensions[0], dimensions[1], dimensions[2], 1)

# Downsample the stimulus function to generate it in TR time
stimfunction_tr = stimfunction[::int(tr_duration * 1000)]

# Iterate through the participants and store participants
epochs = []
for participantcounter in range(1, participants + 1):

    # Add the epoch cube
    epochs += [epoch]

    # Save a file name
    savename = directory + 'p' + str(participantcounter) + '.nii'

    # Create the noise volumes (using the default parameters
    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction_tr=stimfunction_tr,
                               tr_duration=tr_duration,
                               template=template,
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
brain_nifti = nibabel.Nifti1Image(mask, affine_matrix)
nibabel.save(brain_nifti, directory + 'mask.nii')
