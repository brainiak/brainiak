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

"""fMRI Simulator test script

Test script for generating a run of a participant's data.

 Authors: Cameron Ellis (Princeton) 2016
"""
import numpy as np
import math
from brainiak.utils import fmrisim as sim


def test_generate_signal():

    # Inputs for generate_signal
    dimensions = np.array([64, 64, 36]) # What is the size of the brain
    feature_size = [2]
    feature_type = ['cube']
    feature_coordinates = np.array(
        [[32, 32, 18]])
    signal_magnitude = [30]

    # Generate a volume representing the location and quality of the signal
    volume_static = sim.generate_signal(dimensions=dimensions,
                                        feature_coordinates=feature_coordinates,
                                        feature_type=feature_type,
                                        feature_size=feature_size,
                                        signal_magnitude=signal_magnitude,
                                        )

    assert np.all(volume_static.shape == dimensions), "Check signal shape"
    assert np.max(volume_static) == signal_magnitude, "Check signal magnitude"
    assert np.sum(volume_static>0) == math.pow(feature_size[0], 3), "Check " \
                                                                    "feature size"
    assert volume_static[32,32,18] == signal_magnitude, "Check signal location"
    assert volume_static[32,32,10] == 0, "Check noise location"

    feature_coordinates = np.array(
        [[32, 32, 18], [32, 28, 18], [28, 32, 18]])

    volume_static = sim.generate_signal(dimensions=dimensions,
                                        feature_coordinates=feature_coordinates,
                                        feature_type=['loop', 'cavity', 'sphere'],
                                        feature_size=[9],
                                        signal_magnitude=signal_magnitude,
                                        )
    assert volume_static[32, 32, 18] == 0, "Loop is empty"
    assert volume_static[32, 28, 18] == 0, "Cavity is empty"
    assert volume_static[28, 32, 18] != 0, "Sphere is not empty"


def test_generate_stimfunction():

    # Inputs for generate_stimfunction
    onsets = [10, 30, 50, 70, 90]
    event_durations = [6]
    tr_duration = 2
    duration = 100

    # Create the time course for the signal to be generated
    stimfunction = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             tr_duration=tr_duration,
                                             )

    assert len(stimfunction) == duration / tr_duration, "stimfunction incorrect " \
                                                        "length"
    assert np.sum(stimfunction) == np.sum(event_durations * len(onsets)) / \
                                   tr_duration, "Event number"


    # Create the signal function
    signal_function = sim.double_gamma_hrf(stimfunction=stimfunction,
                                           )
    assert len(signal_function) == len(stimfunction), "The length did not change"

    onsets = [10]
    stimfunction = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             tr_duration=tr_duration,
                                             )

    signal_function = sim.double_gamma_hrf(stimfunction=stimfunction,
                                           )
    assert np.sum(signal_function < 0) > 0, "No values below zero"


def test_apply_signal():

    dimensions = np.array([64, 64, 36]) # What is the size of the brain
    feature_size = [2]
    feature_type = ['cube']
    feature_coordinates = np.array(
        [[32, 32, 18]])
    signal_magnitude = [30]

    # Generate a volume representing the location and quality of the signal
    volume_static = sim.generate_signal(dimensions=dimensions,
                                        feature_coordinates=feature_coordinates,
                                        feature_type=feature_type,
                                        feature_size=feature_size,
                                        signal_magnitude=signal_magnitude,
                                        )

    # Inputs for generate_stimfunction
    onsets = [10, 30, 50, 70, 90]
    event_durations = [6]
    tr_duration = 2
    duration = 100

    # Create the time course for the signal to be generated
    stimfunction = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             tr_duration=tr_duration,
                                             )

    signal_function = sim.double_gamma_hrf(stimfunction=stimfunction,
                                           )

    # Convolve the HRF with the stimulus sequence
    signal = sim.apply_signal(signal_function=signal_function,
                              volume_static=volume_static,
                              )

    assert signal.shape == (dimensions[0], dimensions[1], dimensions[2],
                            duration / tr_duration), "The output is the wrong size"

    signal = sim.apply_signal(signal_function=stimfunction,
                              volume_static=volume_static,
                              )

    assert np.any(signal == signal_magnitude), "The stimfunction is not binary"


def test_generate_noise():


    dimensions = np.array([64, 64, 36]) # What is the size of the brain
    feature_size = [2]
    feature_type = ['cube']
    feature_coordinates = np.array(
        [[32, 32, 18]])
    signal_magnitude = [30]

    # Generate a volume representing the location and quality of the signal
    volume_static = sim.generate_signal(dimensions=dimensions,
                                        feature_coordinates=feature_coordinates,
                                        feature_type=feature_type,
                                        feature_size=feature_size,
                                        signal_magnitude=signal_magnitude,
                                        )

    # Inputs for generate_stimfunction
    onsets = [10, 30, 50, 70, 90]
    event_durations = [6]
    tr_duration = 2
    duration = 100

    # Create the time course for the signal to be generated
    stimfunction = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             tr_duration=tr_duration,
                                             )

    signal_function = sim.double_gamma_hrf(stimfunction=stimfunction,
                                           )

    # Convolve the HRF with the stimulus sequence
    signal = sim.apply_signal(signal_function=signal_function,
                              volume_static=volume_static,
                              )

    # Create the noise volumes (using the default parameters)

    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction=stimfunction,
                               tr_duration=tr_duration,
                               )

    assert signal.shape == noise.shape, "The dimensions of signal " \
                                                  "and noise the same"

    Z_noise = sim._generate_noise_temporal(stimfunction, tr_duration, 1)
    noise = sim._generate_noise_temporal(stimfunction, tr_duration, 0)

    assert np.std(Z_noise) < np.std(noise), "Z scoring is not working"

    # Combine the signal and the noise
    volume = signal + noise

    assert np.std(signal) < np.std(noise), "Noise was not created"

    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction=stimfunction,
                               tr_duration=tr_duration,
                               noise_strength=[0, 0, 0]
                               )

    assert np.sum(noise) == 0, "Noise strength could not be manipulated"
    assert np.std(noise) == 0, "Noise strength could not be manipulated"


def test_mask_brain():

    # Inputs for generate_signal
    dimensions = np.array([64, 64, 36]) # What is the size of the brain
    feature_size = [2]
    feature_type = ['cube']
    feature_coordinates = np.array(
        [[32, 32, 18]])
    signal_magnitude = [30]

    # Generate a volume representing the location and quality of the signal
    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=feature_type,
                                 feature_size=feature_size,
                                 signal_magnitude=signal_magnitude,
                                 )

    # Mask the volume to be the same shape as a brain
    brain = sim.mask_brain(volume)

    assert np.sum(brain != 0) == np.sum(volume != 0), "Masking did not work"
    assert brain[0,0,0,0] == 0, "Masking did not work"
    assert brain[32,32,18,0] != 0, "Masking did not work"

    feature_coordinates = np.array(
        [[3, 3, 3]])

    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=feature_type,
                                 feature_size=feature_size,
                                 signal_magnitude=signal_magnitude,
                                 )

    # Mask the volume to be the same shape as a brain
    brain = sim.mask_brain(volume)

    assert np.sum(brain != 0) < np.sum(volume != 0), "Masking did not work"