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
from mpl_toolkits.mplot3d import Axes3D # noqa: F401

logger = logging.getLogger(__name__)

# Inputs for generate_signal
dimensions = np.array([64, 64, 36]) # What is the size of the brain
feature_size = [9, 4, 9, 9]
feature_type = ['loop', 'cube', 'cavity', 'sphere']
feature_coordinates_A = np.array(
    [[32, 32, 18], [26, 32, 18], [32, 26, 18], [32, 32, 12]])
feature_coordinates_B = np.array(
    [[32, 32, 18], [38, 32, 18], [32, 38, 18], [32, 32, 24]])
signal_magnitude = [30, 30, 30, 30]


# Inputs for generate_stimfunction
onsets_A = [10, 30, 50, 70, 90]
onsets_B = [0, 20, 40, 60, 80]
event_durations = [6]
tr_duration = 2
duration = 100

# Generate a volume representing the location and quality of the signal
volume_static_A = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=feature_coordinates_A,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )

volume_static_B = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=feature_coordinates_B,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )


# Visualize the signal that was generated for condition A
fig = plt.figure()
sim.plot_brain(fig,
               volume_static_A)
plt.show()

# Create the time course for the signal to be generated
stimfunction_A = sim.generate_stimfunction(onsets=onsets_A,
                                           event_durations=event_durations,
                                           total_time=duration,
                                           tr_duration=tr_duration,
                                           )

stimfunction_B = sim.generate_stimfunction(onsets=onsets_B,
                                           event_durations=event_durations,
                                           total_time=duration,
                                           tr_duration=tr_duration,
                                           )

# Convolve the HRF with the stimulus sequence
signal_function_A = sim.double_gamma_hrf(stimfunction=stimfunction_A,
                                         )

signal_function_B = sim.double_gamma_hrf(stimfunction=stimfunction_B,
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

# Create the noise volumes (using the default parameters
noise = sim.generate_noise(dimensions=dimensions,
                           stimfunction=stimfunction,
                           tr_duration=tr_duration,
                           )

# Combine the signal and the noise
volume = signal + noise

# Mask the volume to be the same shape as a brain
brain = sim.mask_brain(volume)

# Display the brain
fig = plt.figure()
for tr_counter in list(range(0, brain.shape[3])):

    # Get the axis to be plotted
    ax = sim.plot_brain(fig,
                        brain[:, :, :, tr_counter],
                        percentile=99.9)

    # Wait for an input
    logging.info(tr_counter)
    plt.pause(0.5)