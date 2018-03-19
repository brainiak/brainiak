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

"""fMRI Simulator example script

Example script to generate a run of a participant's data. This generates
data representing a pair of conditions that are then combined

 Authors: Cameron Ellis (Princeton) 2016
"""
import logging
import numpy as np
from brainiak.utils import fmrisim as sim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import nibabel

logger = logging.getLogger(__name__)

# Inputs for generate_signal
dimensions = np.array([64, 64, 36])  # What is the size of the brain
feature_size = [9, 4, 9, 9]
feature_type = ['loop', 'cube', 'cavity', 'sphere']
coordinates_A = np.array(
    [[32, 32, 18], [26, 32, 18], [32, 26, 18], [32, 32, 12]])
coordinates_B = np.array(
    [[32, 32, 18], [38, 32, 18], [32, 38, 18], [32, 32, 24]])
signal_magnitude = [1, 0.5, 0.25, -1] # In percent signal change

# Inputs for generate_stimfunction
onsets_A = [10, 30, 50, 70, 90]
onsets_B = [0, 20, 40, 60, 80]
event_durations = [6]
tr_duration = 2
temporal_res = 1000.0  # How many elements per second are there
duration = 100

# Specify a name to save this generated volume.
savename = 'examples/utils/example.nii'

# Generate a volume representing the location and quality of the signal
volume_signal_A = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=coordinates_A,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )

volume_signal_B = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=coordinates_B,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )

# Visualize the signal that was generated for condition A
fig = plt.figure()
sim.plot_brain(fig,
               volume_signal_A)
plt.show()

# Create the time course for the signal to be generated
stimfunction_A = sim.generate_stimfunction(onsets=onsets_A,
                                           event_durations=event_durations,
                                           total_time=duration,
                                           temporal_resolution=temporal_res,
                                           )

stimfunction_B = sim.generate_stimfunction(onsets=onsets_B,
                                           event_durations=event_durations,
                                           total_time=duration,
                                           temporal_resolution=temporal_res,
                                           )

# Convolve the HRF with the stimulus sequence
signal_function_A = sim.convolve_hrf(stimfunction=stimfunction_A,
                                     tr_duration=tr_duration,
                                     temporal_resolution=temporal_res,
                                     )

signal_function_B = sim.convolve_hrf(stimfunction=stimfunction_B,
                                     tr_duration=tr_duration,
                                     temporal_resolution=temporal_res,
                                     )

# Multiply the HRF timecourse with the signal
signal_A = sim.apply_signal(signal_function=signal_function_A,
                            volume_signal=volume_signal_A,
                            )

signal_B = sim.apply_signal(signal_function=signal_function_B,
                            volume_signal=volume_signal_B,
                            )

# Combine the signals from the two conditions
signal = signal_A + signal_B

# Combine the stim functions
stimfunction = list(np.add(stimfunction_A, stimfunction_B))
stimfunction_tr = stimfunction[::int(tr_duration * temporal_res)]

# Generate the mask of the signal
mask, template = sim.mask_brain(signal, mask_threshold=0.2)

# Mask the signal to the shape of a brain (attenuates signal according to grey
# matter likelihood)
signal *= mask.reshape(dimensions[0], dimensions[1], dimensions[2], 1)

# Generate original noise dict for comparison later
orig_noise_dict = sim._noise_dict_update({})

# Create the noise volumes (using the default parameters
noise = sim.generate_noise(dimensions=dimensions,
                           stimfunction_tr=stimfunction_tr,
                           tr_duration=tr_duration,
                           mask=mask,
                           template=template,
                           noise_dict=orig_noise_dict,
                           )

# Standardize the signal activity to make it percent signal change
mean_act = (mask * orig_noise_dict['max_activity']).sum() / (mask > 0).sum()
signal = signal * mean_act / 100

# Combine the signal and the noise
brain = signal + noise

# Display the brain
fig = plt.figure()
for tr_counter in list(range(0, brain.shape[3])):

    # Get the axis to be plotted
    ax = sim.plot_brain(fig,
                        brain[:, :, :, tr_counter],
                        mask=mask,
                        percentile=99.9)

    # Wait for an input
    logging.info(tr_counter)
    plt.pause(0.5)

# Save the volume
affine_matrix = np.diag([-1, 1, 1, 1])  # LR gets flipped
brain_nifti = nibabel.Nifti1Image(brain, affine_matrix)  # Create a nifti brain
nibabel.save(brain_nifti, savename)

# Load in the test dataset and generate a random volume based on it

# Pull out the data and associated data
volume = nibabel.load(savename).get_data()
dimensions = volume.shape[0:3]
total_time = volume.shape[3] * tr_duration
stimfunction = sim.generate_stimfunction(onsets=[],
                                         event_durations=[0],
                                         total_time=total_time,
                                         )
stimfunction_tr = stimfunction[::int(tr_duration * temporal_res)]

# Calculate the mask
mask, template = sim.mask_brain(volume=volume,
                                mask_self=True,
                                )

# Calculate the noise parameters
noise_dict = sim.calc_noise(volume=volume,
                            mask=mask,
                            )

# Create the noise volumes (using the default parameters
noise = sim.generate_noise(dimensions=dimensions,
                           tr_duration=tr_duration,
                           stimfunction_tr=stimfunction_tr,
                           template=template,
                           mask=mask,
                           noise_dict=noise_dict,
                           )

# Create a nifti brain
brain_noise = nibabel.Nifti1Image(noise, affine_matrix)
nibabel.save(brain_noise, 'examples/utils/example2.nii')  # Save
