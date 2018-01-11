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
    dimensions = np.array([10, 10, 10])  # What is the size of the brain
    feature_size = [3]
    feature_type = ['cube']
    feature_coordinates = np.array(
        [[5, 5, 5]])
    signal_magnitude = [30]

    # Generate a volume representing the location and quality of the signal
    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=feature_type,
                                 feature_size=feature_size,
                                 signal_magnitude=signal_magnitude,
                                 )

    assert np.all(volume.shape == dimensions), "Check signal shape"
    assert np.max(volume) == signal_magnitude, "Check signal magnitude"
    assert np.sum(volume > 0) == math.pow(feature_size[0], 3), (
        "Check feature size")
    assert volume[5, 5, 5] == signal_magnitude, "Check signal location"
    assert volume[5, 5, 1] == 0, "Check noise location"

    feature_coordinates = np.array(
        [[5, 5, 5], [3, 3, 3], [7, 7, 7]])

    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=['loop', 'cavity', 'sphere'],
                                 feature_size=[3],
                                 signal_magnitude=signal_magnitude)
    assert volume[5, 5, 5] == 0, "Loop is empty"
    assert volume[3, 3, 3] == 0, "Cavity is empty"
    assert volume[7, 7, 7] != 0, "Sphere is not empty"


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
                                             )

    assert stimfunction.shape[0] == duration * 1000, "stimfunc incorrect " \
                                                     "length"
    eventNumber = np.sum(event_durations * len(onsets)) * 1000
    assert np.sum(stimfunction) == eventNumber, "Event number"

    # Create the signal function
    signal_function = sim.convolve_hrf(stimfunction=stimfunction,
                                       tr_duration=tr_duration,
                                       )

    stim_dur = stimfunction.shape[0] / (tr_duration * 1000)
    assert signal_function.shape[0] == stim_dur, "The length did not change"

    onsets = [10]
    stimfunction = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             )

    signal_function = sim.convolve_hrf(stimfunction=stimfunction,
                                       tr_duration=tr_duration,
                                       )
    assert np.sum(signal_function < 0) > 0, "No values below zero"


def test_apply_signal():

    dimensions = np.array([10, 10, 10])  # What is the size of the brain
    feature_size = [2]
    feature_type = ['cube']
    feature_coordinates = np.array(
        [[5, 5, 5]])
    signal_magnitude = [30]

    # Generate a volume representing the location and quality of the signal
    volume = sim.generate_signal(dimensions=dimensions,
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
                                             )

    signal_function = sim.convolve_hrf(stimfunction=stimfunction,
                                       tr_duration=tr_duration,
                                       )

    # Convolve the HRF with the stimulus sequence
    signal = sim.apply_signal(signal_function=signal_function,
                              volume_signal=volume,
                              )

    assert signal.shape == (dimensions[0], dimensions[1], dimensions[2],
                            duration / tr_duration), "The output is the " \
                                                     "wrong size"

    signal = sim.apply_signal(signal_function=stimfunction,
                              volume_signal=volume,
                              )

    assert np.any(signal == signal_magnitude), "The stimfunction is not binary"


def test_generate_noise():

    dimensions = np.array([10, 10, 10])  # What is the size of the brain
    feature_size = [2]
    feature_type = ['cube']
    feature_coordinates = np.array(
        [[5, 5, 5]])
    signal_magnitude = [1]

    # Generate a volume representing the location and quality of the signal
    volume = sim.generate_signal(dimensions=dimensions,
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
                                             )

    signal_function = sim.convolve_hrf(stimfunction=stimfunction,
                                       tr_duration=tr_duration,
                                       )

    # Convolve the HRF with the stimulus sequence
    signal = sim.apply_signal(signal_function=signal_function,
                              volume_signal=volume,
                              )

    # Generate the mask of the signal
    mask, template = sim.mask_brain(signal, mask_threshold=0.1)

    assert min(mask[mask > 0]) > 0.1, "Mask thresholding did not work"
    assert len(np.unique(template) > 2), "Template creation did not work"

    stimfunction_tr = stimfunction[::int(tr_duration * 1000)]
    # Create the noise volumes (using the default parameters)
    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction_tr=stimfunction_tr,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               )

    assert signal.shape == noise.shape, "The dimensions of signal and noise " \
                                        "the same"

    assert np.std(signal) < np.std(noise), "Noise was not created"

    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction_tr=stimfunction_tr,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               noise_dict={'sfnr': 10000, 'snr': 10000},
                               )

    system_noise = np.std(noise[mask > 0], 1).mean()

    assert system_noise <= 0.1, "Noise strength could not be manipulated"


def test_mask_brain():

    # Inputs for generate_signal
    dimensions = np.array([10, 10, 10])  # What is the size of the brain
    feature_size = [2]
    feature_type = ['cube']
    feature_coordinates = np.array(
        [[4, 4, 4]])
    signal_magnitude = [30]

    # Generate a volume representing the location and quality of the signal
    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=feature_type,
                                 feature_size=feature_size,
                                 signal_magnitude=signal_magnitude,
                                 )

    # Mask the volume to be the same shape as a brain
    mask, _ = sim.mask_brain(volume)
    brain = volume * mask

    assert np.sum(brain != 0) == np.sum(volume != 0), "Masking did not work"
    assert brain[0, 0, 0] == 0, "Masking did not work"
    assert brain[4, 4, 4] != 0, "Masking did not work"

    feature_coordinates = np.array(
        [[1, 1, 1]])

    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=feature_type,
                                 feature_size=feature_size,
                                 signal_magnitude=signal_magnitude,
                                 )

    # Mask the volume to be the same shape as a brain
    mask, _ = sim.mask_brain(volume)
    brain = volume * mask

    assert np.sum(brain != 0) < np.sum(volume != 0), "Masking did not work"


def test_calc_noise():

    # Inputs for functions
    onsets = [10, 30, 50, 70, 90]
    event_durations = [6]
    tr_duration = 2
    duration = 100
    tr_number = int(np.floor(duration / tr_duration))
    dimensions_tr = np.array([10, 10, 10, tr_number])

    # Preset the noise dict
    nd_orig = {'auto_reg_sigma': 0.6,
               'drift_sigma': 0.4,
               'snr': 30,
               'sfnr': 30,
               'max_activity': 1000,
               'fwhm': 4,
               }

    # Create the time course for the signal to be generated
    stimfunction = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             )

    # Mask the volume to be the same shape as a brain
    mask, template = sim.mask_brain(dimensions_tr, mask_threshold=0.2)
    stimfunction_tr = stimfunction[::int(tr_duration * 1000)]
    noise = sim.generate_noise(dimensions=dimensions_tr[0:3],
                               stimfunction_tr=stimfunction_tr,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               noise_dict=nd_orig,
                               )

    # Check that noise_system is being calculated correctly
    spatial_sd = 5
    temporal_sd = 5
    noise_system = sim._generate_noise_system(dimensions_tr,
                                              spatial_sd,
                                              temporal_sd)

    precision = abs(noise_system[0, 0, 0, :].std() - spatial_sd)
    assert precision < spatial_sd, 'noise_system calculated incorrectly'

    precision = abs(noise_system[:, :, :, 0].std() - temporal_sd)
    assert precision < spatial_sd, 'noise_system calculated incorrectly'

    # Calculate the noise
    nd_calc = sim.calc_noise(volume=noise,
                             mask=mask)

    # How precise are these estimates
    precision = abs(nd_calc['snr'] - nd_orig['snr'])
    assert precision < nd_orig['snr'], 'snr calculated incorrectly'

    precision = abs(nd_calc['sfnr'] - nd_orig['sfnr'])
    assert precision < nd_orig['sfnr'], 'sfnr calculated incorrectly'
