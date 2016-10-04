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

"""Brain Simulator example script

Example script to generate a run of a participant's data.

 Authors: Cameron Ellis (Princeton) 2016
"""
import logging

import numpy as np
from brainiak.utils import simulator as sim

logger = logging.getLogger(__name__)


# Inputs for generate_signal
dimensions = np.array([64, 64, 36]) # What is the size of the brain
feature_size = [9, 4, 9, 9]
feature_type = ['loop', 'cube', 'cavity', 'sphere']
feature_coordinates = np.array(
    [[32, 32, 18], [26, 32, 18], [32, 26, 18], [32, 32, 12]])
signal_magnitude = [30, 30, 30, 30]


# Inputs for generate_stimfunction
onsets = [10, 30, 50, 70, 90]
event_durations = [6]
tr_duration = 2
duration = 100

# Generate a volume representing the location and quality of the signal
volume_static = sim.generate_signal(dimensions=dimensions,
                                    feature_coordinates=feature_coordinates,
                                    feature_type=feature_type,
                                    feature_size=feature_size,
                                    signal_magnitude=signal_magnitude,
                                    )

# Visualize the signal that was generated
sim.plot_brain(volume_static)

# Create the time course for the signal to be generated
stimfunction = sim.generate_stimfunction(onsets=onsets,
                                         event_durations=event_durations,
                                         total_time=duration,
                                         tr_duration=tr_duration,
                                         )


# Create the signal function
signal_function = sim.double_gamma_hrf(stimfunction=stimfunction,
                                       )

# Convolve the HRF with the stimulus sequence
signal = sim.apply_signal(signal_function=signal_function,
                          volume_static=volume_static,
                          )

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
sim.plot_brain(brain,
               percentile=99.9)
