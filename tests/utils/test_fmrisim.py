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

    assert stimfunction.shape[0] == duration * 100, "stimfunc incorrect length"
    eventNumber = np.sum(event_durations * len(onsets)) * 100
    assert np.sum(stimfunction) == eventNumber, "Event number"

    # Create the signal function
    signal_function = sim.convolve_hrf(stimfunction=stimfunction,
                                       tr_duration=tr_duration,
                                       )

    stim_dur = stimfunction.shape[0] / (tr_duration * 100)
    assert signal_function.shape[0] == stim_dur, "The length did not change"

    # Test
    onsets = [0]
    tr_duration = 1
    event_durations = [1]
    stimfunction = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             )

    signal_function = sim.convolve_hrf(stimfunction=stimfunction,
                                       tr_duration=tr_duration,
                                       )

    max_response = np.where(signal_function != 0)[0].max()
    assert 25 < max_response <= 30, "HRF has the incorrect length"
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
    duration = 200

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
    mask, template = sim.mask_brain(signal,
                                    mask_self=None)

    assert min(mask[mask > 0]) > 0.1, "Mask thresholding did not work"
    assert len(np.unique(template) > 2), "Template creation did not work"

    stimfunction_tr = stimfunction[::int(tr_duration * 100)]
    # Create the noise volumes (using the default parameters)
    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction_tr=stimfunction_tr,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               iterations=[1, 0],
                               )

    assert signal.shape == noise.shape, "The dimensions of signal and noise " \
                                        "the same"

    noise_high = sim.generate_noise(dimensions=dimensions,
                                    stimfunction_tr=stimfunction_tr,
                                    tr_duration=tr_duration,
                                    template=template,
                                    mask=mask,
                                    noise_dict={'sfnr': 50, 'snr': 25},
                                    iterations=[1, 0],
                                    )

    noise_low = sim.generate_noise(dimensions=dimensions,
                                   stimfunction_tr=stimfunction_tr,
                                   tr_duration=tr_duration,
                                   template=template,
                                   mask=mask,
                                   noise_dict={'sfnr': 100, 'snr': 25},
                                   iterations=[1, 0],
                                   )

    system_high = np.std(noise_high[mask > 0], 1).mean()
    system_low = np.std(noise_low[mask > 0], 1).mean()

    assert system_low < system_high, "SFNR noise could not be manipulated"


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
    mask, _ = sim.mask_brain(dimensions, mask_self=None,)
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
    mask, _ = sim.mask_brain(dimensions, mask_self=None, )
    brain = volume * mask

    assert np.sum(brain != 0) < np.sum(volume != 0), "Masking did not work"


def test_calc_noise():

    # Inputs for functions
    onsets = [10, 30, 50, 70, 90]
    event_durations = [6]
    tr_duration = 2
    duration = 200
    temporal_res = 100
    tr_number = int(np.floor(duration / tr_duration))
    dimensions_tr = np.array([10, 10, 10, tr_number])

    # Preset the noise dict
    nd_orig = sim._noise_dict_update({})

    # Create the time course for the signal to be generated
    stimfunction = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             temporal_resolution=temporal_res,
                                             )

    # Mask the volume to be the same shape as a brain
    mask, template = sim.mask_brain(dimensions_tr, mask_self=None)
    stimfunction_tr = stimfunction[::int(tr_duration * temporal_res)]

    nd_orig['matched'] = 0
    noise = sim.generate_noise(dimensions=dimensions_tr[0:3],
                               stimfunction_tr=stimfunction_tr,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               noise_dict=nd_orig,
                               )

    # Check the spatial noise match
    nd_orig['matched'] = 1
    noise_matched = sim.generate_noise(dimensions=dimensions_tr[0:3],
                                       stimfunction_tr=stimfunction_tr,
                                       tr_duration=tr_duration,
                                       template=template,
                                       mask=mask,
                                       noise_dict=nd_orig,
                                       iterations=[50, 0]
                                       )

    # Calculate the noise parameters from this newly generated volume
    nd_new = sim.calc_noise(noise, mask, template)
    nd_matched = sim.calc_noise(noise_matched, mask, template)

    # Check the values are reasonable"
    assert nd_new['snr'] > 0, 'snr out of range'
    assert nd_new['sfnr'] > 0, 'sfnr out of range'
    assert nd_new['auto_reg_rho'][0] > 0, 'ar out of range'

    # Check that the fitting worked
    snr_diff = abs(nd_orig['snr'] - nd_new['snr'])
    snr_diff_match = abs(nd_orig['snr'] - nd_matched['snr'])
    assert snr_diff > snr_diff_match, 'snr fit incorrectly'

    # Check the temporal noise match
    nd_orig['matched'] = 1
    noise_matched = sim.generate_noise(dimensions=dimensions_tr[0:3],
                                       stimfunction_tr=stimfunction_tr,
                                       tr_duration=tr_duration,
                                       template=template,
                                       mask=mask,
                                       noise_dict=nd_orig,
                                       iterations=[0, 50]
                                       )

    nd_matched = sim.calc_noise(noise_matched, mask, template)

    sfnr_diff = abs(nd_orig['sfnr'] - nd_new['sfnr'])
    sfnr_diff_match = abs(nd_orig['sfnr'] - nd_matched['sfnr'])
    assert sfnr_diff > sfnr_diff_match, 'sfnr fit incorrectly'

    ar1_diff = abs(nd_orig['auto_reg_rho'][0] - nd_new['auto_reg_rho'][0])
    ar1_diff_match = abs(nd_orig['auto_reg_rho'][0] - nd_matched[
        'auto_reg_rho'][0])
    assert ar1_diff > ar1_diff_match, 'AR1 fit incorrectly'
