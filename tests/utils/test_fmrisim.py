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
import pytest
from itertools import product


def test_generate_signal():

    # Inputs for generate_signal
    dimensions = np.array([10, 10, 10])  # What is the size of the brain
    feature_size = [3]
    feature_type = ['cube']
    feature_coordinates = np.array([[5, 5, 5]])
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

    # Check feature size is correct
    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=['loop', 'cavity', 'sphere'],
                                 feature_size=[3],
                                 signal_magnitude=signal_magnitude)
    assert volume[5, 5, 5] == 0, "Loop is empty"
    assert volume[3, 3, 3] == 0, "Cavity is empty"
    assert volume[7, 7, 7] != 0, "Sphere is not empty"

    # Check feature size manipulation
    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=['loop', 'cavity', 'sphere'],
                                 feature_size=[1],
                                 signal_magnitude=signal_magnitude)
    assert volume[5, 6, 6] == 0, "Loop is too big"
    assert volume[3, 5, 5] == 0, "Cavity is too big"
    assert volume[7, 9, 9] == 0, "Sphere is too big"

    # Check that out of bounds feature coordinates are corrected
    feature_coordinates = np.array([0, 2, dimensions[2]])
    x, y, z = sim._insert_idxs(feature_coordinates, feature_size[0],
                               dimensions)
    assert x[1] - x[0] == 2, "x min not corrected"
    assert y[1] - y[0] == 3, "y was corrected when it shouldn't be"
    assert z[1] - z[0] == 1, "z max not corrected"

    # Check that signal patterns are created
    feature_coordinates = np.array([[5, 5, 5]])
    volume = sim.generate_signal(dimensions=dimensions,
                                 feature_coordinates=feature_coordinates,
                                 feature_type=feature_type,
                                 feature_size=feature_size,
                                 signal_magnitude=signal_magnitude,
                                 signal_constant=0,
                                 )
    assert volume[4:7, 4:7, 4:7].std() > 0, "Signal is constant"


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

    # Export a stimfunction
    sim.export_3_column(stimfunction,
                        'temp.txt',
                        )

    # Load in the stimfunction
    stimfunc_new = sim.generate_stimfunction(onsets=None,
                                             event_durations=None,
                                             total_time=duration,
                                             timing_file='temp.txt',
                                             )

    assert np.all(stimfunc_new == stimfunction), "Export/import failed"

    # Break the timing precision of the generation
    stimfunc_new = sim.generate_stimfunction(onsets=None,
                                             event_durations=None,
                                             total_time=duration,
                                             timing_file='temp.txt',
                                             temporal_resolution=0.5,
                                             )

    assert stimfunc_new.sum() == 0, "Temporal resolution not working right"

    # Set the duration to be too short so you should get an error
    onsets = [10, 30, 50, 70, 90]
    event_durations = [5]
    with pytest.raises(ValueError):
        sim.generate_stimfunction(onsets=onsets,
                                  event_durations=event_durations,
                                  total_time=89,
                                  )

    # Clip the event offset
    stimfunc_new = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=95,
                                             )
    assert stimfunc_new[-1] == 1, 'Event offset was not clipped'

    # Test exporting a group of participants to an epoch file
    cond_a = sim.generate_stimfunction(onsets=onsets,
                                       event_durations=event_durations,
                                       total_time=110,
                                       )

    cond_b = sim.generate_stimfunction(onsets=[x + 5 for x in onsets],
                                       event_durations=event_durations,
                                       total_time=110,
                                       )

    stimfunction_group = [np.hstack((cond_a, cond_b))] * 2
    sim.export_epoch_file(stimfunction_group,
                          'temp.txt',
                          tr_duration,
                          )

    # Check that convolve throws a warning when the shape is wrong
    sim.convolve_hrf(stimfunction=np.hstack((cond_a, cond_b)).T,
                     tr_duration=tr_duration,
                     temporal_resolution=1,
                     )


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

    # Check that you can compute signal change appropriately
    # Preset a bunch of things
    stimfunction_tr = stimfunction[::int(tr_duration * 100)]
    mask, template = sim.mask_brain(dimensions, mask_self=False)
    noise_dict = sim._noise_dict_update({})
    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction_tr=stimfunction_tr,
                               tr_duration=tr_duration,
                               template=template,
                               mask=mask,
                               noise_dict=noise_dict,
                               iterations=[0, 0]
                               )
    coords = feature_coordinates[0]
    noise_function_a = noise[coords[0], coords[1], coords[2], :]
    noise_function_a = noise_function_a.reshape(duration // tr_duration, 1)

    noise_function_b = noise[coords[0] + 1, coords[1], coords[2], :]
    noise_function_b = noise_function_b.reshape(duration // tr_duration, 1)

    # Check that the noise_function and signal_function must be the same size
    method = 'PSC'
    with pytest.raises(ValueError):
        sim.compute_signal_change(signal_function,
                                  noise_function_a.T,
                                  noise_dict,
                                  [0.5],
                                  method,
                                  )

    # Create the calibrated signal with PSC
    sig_a = sim.compute_signal_change(signal_function,
                                      noise_function_a,
                                      noise_dict,
                                      [0.5],
                                      method,
                                      )
    sig_b = sim.compute_signal_change(signal_function,
                                      noise_function_a,
                                      noise_dict,
                                      [1.0],
                                      method,
                                      )

    assert sig_b.max() / sig_a.max() == 2, 'PSC modulation failed'

    # Create the calibrated signal with SFNR
    method = 'SFNR'
    sig_a = sim.compute_signal_change(signal_function,
                                      noise_function_a,
                                      noise_dict,
                                      [0.5],
                                      method,
                                      )
    scaled_a = sig_a / (noise_function_a.mean() / noise_dict['sfnr'])
    sig_b = sim.compute_signal_change(signal_function,
                                      noise_function_b,
                                      noise_dict,
                                      [1.0],
                                      method,
                                      )
    scaled_b = sig_b / (noise_function_b.mean() / noise_dict['sfnr'])

    assert scaled_b.max() / scaled_a.max() == 2, 'SFNR modulation failed'

    # Create the calibrated signal with CNR_Amp/Noise-SD
    method = 'CNR_Amp/Noise-SD'
    sig_a = sim.compute_signal_change(signal_function,
                                      noise_function_a,
                                      noise_dict,
                                      [0.5],
                                      method,
                                      )
    scaled_a = sig_a / noise_function_a.std()
    sig_b = sim.compute_signal_change(signal_function,
                                      noise_function_b,
                                      noise_dict,
                                      [1.0],
                                      method,
                                      )
    scaled_b = sig_b / noise_function_b.std()

    assert scaled_b.max() / scaled_a.max() == 2, 'CNR_Amp modulation failed'

    # Create the calibrated signal with CNR_Amp/Noise-Var_dB
    method = 'CNR_Amp2/Noise-Var_dB'
    sig_a = sim.compute_signal_change(signal_function,
                                      noise_function_a,
                                      noise_dict,
                                      [0.5],
                                      method,
                                      )
    scaled_a = np.log(sig_a.max() / noise_function_a.std())
    sig_b = sim.compute_signal_change(signal_function,
                                      noise_function_b,
                                      noise_dict,
                                      [1.0],
                                      method,
                                      )
    scaled_b = np.log(sig_b.max() / noise_function_b.std())

    assert np.round(scaled_b / scaled_a) == 2, 'CNR_Amp dB modulation failed'

    # Create the calibrated signal with CNR_Signal-SD/Noise-SD
    method = 'CNR_Signal-SD/Noise-SD'
    sig_a = sim.compute_signal_change(signal_function,
                                      noise_function_a,
                                      noise_dict,
                                      [0.5],
                                      method,
                                      )
    scaled_a = sig_a.std() / noise_function_a.std()
    sig_b = sim.compute_signal_change(signal_function,
                                      noise_function_a,
                                      noise_dict,
                                      [1.0],
                                      method,
                                      )
    scaled_b = sig_b.std() / noise_function_a.std()

    assert (scaled_b / scaled_a) == 2, 'CNR signal modulation failed'

    # Create the calibrated signal with CNR_Amp/Noise-Var_dB
    method = 'CNR_Signal-Var/Noise-Var_dB'
    sig_a = sim.compute_signal_change(signal_function,
                                      noise_function_a,
                                      noise_dict,
                                      [0.5],
                                      method,
                                      )

    scaled_a = np.log(sig_a.std() / noise_function_a.std())
    sig_b = sim.compute_signal_change(signal_function,
                                      noise_function_b,
                                      noise_dict,
                                      [1.0],
                                      method,
                                      )
    scaled_b = np.log(sig_b.std() / noise_function_b.std())

    assert np.round(scaled_b / scaled_a) == 2, 'CNR signal dB modulation ' \
                                               'failed'

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

    # Check that there is an error if the number of signal voxels doesn't
    # match the number of non zero brain voxels
    with pytest.raises(IndexError):
        sig_vox = (volume > 0).sum()
        vox_pattern = np.tile(stimfunction, (1, sig_vox - 1))
        sim.apply_signal(signal_function=vox_pattern,
                         volume_signal=volume,
                         )


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

    # Check that you check for the appropriate template values
    with pytest.raises(ValueError):
        sim.generate_noise(dimensions=dimensions,
                           stimfunction_tr=stimfunction_tr,
                           tr_duration=tr_duration,
                           template=template * 2,
                           mask=mask,
                           noise_dict={},
                           )

    # Check that iterations does what it should
    sim.generate_noise(dimensions=dimensions,
                       stimfunction_tr=stimfunction_tr,
                       tr_duration=tr_duration,
                       template=template,
                       mask=mask,
                       noise_dict={},
                       iterations=[0, 0],
                       )

    sim.generate_noise(dimensions=dimensions,
                       stimfunction_tr=stimfunction_tr,
                       tr_duration=tr_duration,
                       template=template,
                       mask=mask,
                       noise_dict={},
                       iterations=None,
                       )

    # Test drift noise
    trs = 1000
    period = 100
    drift = sim._generate_noise_temporal_drift(trs,
                                               tr_duration,
                                               'sine',
                                               period,
                                               )

    # Check that the max frequency is the appropriate frequency
    power = abs(np.fft.fft(drift))[1:trs // 2]
    freq = np.linspace(1, trs // 2 - 1, trs // 2 - 1) / trs
    period_freq = np.where(freq == 1 / (period // tr_duration))
    max_freq = np.argmax(power)

    assert period_freq == max_freq, 'Max frequency is not where it should be'

    # Do the same but now with cosine basis functions, answer should be close
    drift = sim._generate_noise_temporal_drift(trs,
                                               tr_duration,
                                               'discrete_cos',
                                               period,
                                               )

    # Check that the appropriate frequency is peaky (may not be the max)
    power = abs(np.fft.fft(drift))[1:trs // 2]
    freq = np.linspace(1, trs // 2 - 1, trs // 2 - 1) / trs
    period_freq = np.where(freq == 1 / (period // tr_duration))[0][0]

    assert power[period_freq] > power[period_freq + 1], 'Power is low'
    assert power[period_freq] > power[period_freq - 1], 'Power is low'

    # Check it runs fine
    drift = sim._generate_noise_temporal_drift(50,
                                               tr_duration,
                                               'discrete_cos',
                                               period,
                                               )

    # Check it runs fine
    drift = sim._generate_noise_temporal_drift(300,
                                               tr_duration,
                                               'cos_power_drop',
                                               period,
                                               )

    # Check that when the TR is greater than the period it errors
    with pytest.raises(ValueError):
        sim._generate_noise_temporal_drift(30, 10, 'cos_power_drop', 5)

    # Test physiological noise (using unrealistic parameters so that it's easy)
    timepoints = list(np.linspace(0, (trs - 1) * tr_duration, trs))
    resp_freq = 0.2
    heart_freq = 1.17
    phys = sim._generate_noise_temporal_phys(timepoints,
                                             resp_freq,
                                             heart_freq,
                                             )

    # Check that the max frequency is the appropriate frequency
    power = abs(np.fft.fft(phys))[1:trs // 2]
    freq = np.linspace(1, trs // 2 - 1, trs // 2 - 1) / (trs * tr_duration)
    peaks = (power > (power.mean() + power.std()))  # Where are the peaks
    peak_freqs = freq[peaks]

    assert np.any(resp_freq == peak_freqs), 'Resp frequency not found'
    assert len(peak_freqs) == 2, 'Two peaks not found'

    # Test task noise
    sim._generate_noise_temporal_task(stimfunction_tr,
                                      motion_noise='gaussian',
                                      )
    sim._generate_noise_temporal_task(stimfunction_tr,
                                      motion_noise='rician',
                                      )

    # Test ARMA noise
    with pytest.raises(ValueError):
        noise_dict = {'fwhm': 4, 'auto_reg_rho': [1], 'ma_rho': [1, 1]}
        sim._generate_noise_temporal_autoregression(stimfunction_tr,
                                                    noise_dict,
                                                    dimensions,
                                                    mask,
                                                    )

    # Generate spatial noise
    vol = sim._generate_noise_spatial(np.array([10, 10, 10, trs]))
    assert len(vol.shape) == 3, 'Volume was not reshaped to ignore TRs'

    # Switch some of the noise types on
    noise_dict = dict(physiological_sigma=1, drift_sigma=1, task_sigma=1,
                      auto_reg_sigma=0)
    sim.generate_noise(dimensions=dimensions,
                       stimfunction_tr=stimfunction_tr,
                       tr_duration=tr_duration,
                       template=template,
                       mask=mask,
                       noise_dict=noise_dict,
                       iterations=[0, 0],
                       )


def test_generate_noise_spatial():

    # Set up the inputs
    dimensions = np.array([10, 5, 10])
    mask = np.ones(dimensions)
    vol = sim._generate_noise_spatial(dimensions, mask)

    # Run the analysis from _calc_FHWM but for th elast step of aggregating
    # across dimensions
    v_count = 0
    v_sum = 0
    v_sq = 0

    d_sum = [0.0, 0.0, 0.0]
    d_sq = [0.0, 0.0, 0.0]
    d_count = [0, 0, 0]

    # Pull out all the voxel coordinates
    coordinates = list(product(range(dimensions[0]),
                               range(dimensions[1]),
                               range(dimensions[2])))

    # Find the sum of squared error for the non-masked voxels in the brain
    for i in list(range(len(coordinates))):

        # Pull out this coordinate
        x, y, z = coordinates[i]

        # Is this within the mask?
        if mask[x, y, z] > 0:

            # Find the the volume sum and squared values
            v_count += 1
            v_sum += vol[x, y, z]
            v_sq += vol[x, y, z] ** 2

    # Get the volume variance
    v_var = (v_sq - ((v_sum ** 2) / v_count)) / (v_count - 1)

    for i in list(range(len(coordinates))):

        # Pull out this coordinate
        x, y, z = coordinates[i]

        # Is this within the mask?
        if mask[x, y, z] > 0:
            # For each xyz dimension calculate the squared
            # difference of this voxel and the next

            in_range = (x < dimensions[0] - 1)
            in_mask = in_range and (mask[x + 1, y, z] > 0)
            included = in_mask and (~np.isnan(vol[x + 1, y, z]))
            if included:
                d_sum[0] += vol[x, y, z] - vol[x + 1, y, z]
                d_sq[0] += (vol[x, y, z] - vol[x + 1, y, z]) ** 2
                d_count[0] += 1

            in_range = (y < dimensions[1] - 1)
            in_mask = in_range and (mask[x, y + 1, z] > 0)
            included = in_mask and (~np.isnan(vol[x, y + 1, z]))
            if included:
                d_sum[1] += vol[x, y, z] - vol[x, y + 1, z]
                d_sq[1] += (vol[x, y, z] - vol[x, y + 1, z]) ** 2
                d_count[1] += 1

            in_range = (z < dimensions[2] - 1)
            in_mask = in_range and (mask[x, y, z + 1] > 0)
            included = in_mask and (~np.isnan(vol[x, y, z + 1]))
            if included:
                d_sum[2] += vol[x, y, z] - vol[x, y, z + 1]
                d_sq[2] += (vol[x, y, z] - vol[x, y, z + 1]) ** 2
                d_count[2] += 1

    # Find the variance
    d_var = np.divide((d_sq - np.divide(np.power(d_sum, 2),
                                        d_count)), (np.add(d_count, -1)))

    o_var = np.divide(-1, (4 * np.log(1 - (0.5 * d_var / v_var))))
    fwhm3 = np.sqrt(o_var) * 2 * np.sqrt(2 * np.log(2))

    # Calculate the proportion of std relative to the mean
    std_proportion = np.nanstd(fwhm3) / np.nanmean(fwhm3)
    assert std_proportion < 0.25, 'Variance is inconsistent across dim'


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

    # Test that you can load the default
    dimensions = np.array([100, 100, 100])
    mask, template = sim.mask_brain(dimensions, mask_self=False)

    assert mask[20, 80, 50] == 0, 'Masking didn''t work'
    assert mask[25, 80, 50] == 1, 'Masking didn''t work'
    assert int(template[25, 80, 50] * 100) == 57, 'Template not correct'

    # Check that you can mask self
    mask_self, template_self = sim.mask_brain(template, mask_self=True)

    assert (template_self - template).sum() < 1e2, 'Mask self error'
    assert (mask_self - mask).sum() == 0, 'Mask self error'


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

    # Check that the dilation increases SNR
    no_dilation_snr = sim._calc_snr(noise_matched,
                                    mask,
                                    dilation=0,
                                    reference_tr=tr_duration,
                                    )

    assert nd_new['snr'] > no_dilation_snr, "Dilation did not increase SNR"

    # Check that template size is in bounds
    with pytest.raises(ValueError):
        sim.calc_noise(noise, mask, template * 2)

    # Check that Mask is set is checked
    with pytest.raises(ValueError):
        sim.calc_noise(noise, None, template)

    # Check that it can deal with missing noise parameters
    temp_nd = sim.calc_noise(noise, mask, template, noise_dict={})
    assert temp_nd['voxel_size'][0] == 1, 'Default voxel size not set'

    temp_nd = sim.calc_noise(noise, mask, template, noise_dict=None)
    assert temp_nd['voxel_size'][0] == 1, 'Default voxel size not set'

    # Check that the fitting worked
    snr_diff = abs(nd_orig['snr'] - nd_new['snr'])
    snr_diff_match = abs(nd_orig['snr'] - nd_matched['snr'])
    assert snr_diff > snr_diff_match, 'snr fit incorrectly'

    # Test that you can generate rician and exponential noise
    sim._generate_noise_system(dimensions_tr,
                               1,
                               1,
                               spatial_noise_type='exponential',
                               temporal_noise_type='rician',
                               )

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

    # Check that you can calculate ARMA for a single voxel
    vox = noise[5, 5, 5, :]
    arma = sim._calc_ARMA_noise(vox,
                                None,
                                sample_num=2,
                                )
    assert len(arma) == 2, "Two outputs not given by ARMA"


def test_gen_1D_gauss_shape():
    n_vox = 10
    res = 180
    rfs, centers = sim.generate_1d_gaussian_rfs(n_vox, res, (0, res-1))
    assert rfs.shape == (n_vox, res)
    assert centers.size == n_vox

    sim_data = sim.generate_1d_rf_responses(rfs, np.array([0, 10, 20]), res,
                                            (0, res-1))
    assert sim_data.shape == (n_vox, 3)


def test_gen_1d_gauss_range():
    res = 180
    range_values = (-10, res-11)
    rfs, centers = sim.generate_1d_gaussian_rfs(1, res, range_values,
                                                random_tuning=False)
    sim_data = sim.generate_1d_rf_responses(rfs, np.array([-10]), res,
                                            range_values, 0)
    assert sim_data[0, ] > 0
    range_values = (10, res+10)
    rfs, centers = sim.generate_1d_gaussian_rfs(1, res, range_values,
                                                random_tuning=False)
    sim_data = sim.generate_1d_rf_responses(rfs, np.array([10]), res,
                                            range_values, 0)
    assert sim_data[0, ] > 0


def test_gen_1D_gauss_even_spacing():
    n_vox = 9
    res = 180
    rfs, centers = sim.generate_1d_gaussian_rfs(n_vox, res, (0, res-1),
                                                random_tuning=False)
    assert np.all(centers == np.array([0, 19, 39, 59, 79, 99, 119, 139, 159]))
