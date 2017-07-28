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

"""fMRI Simulator

Simulate fMRI data for a single subject.

This code provides a set of functions necessary to produce realistic
simulations of fMRI data. There are two main steps: characterizing the
signal and generating the noise model, which are then combined to simulate
brain data. Tools are included to support the creation of different types
of signal, such as region specific differences in univariate
activity. To create the noise model the parameters can either be set
manually or can be estimated from real fMRI data with reasonable accuracy (
works best when fMRI data has not been preprocessed)

Functions:

generate_signal
Create a volume with activity, of a specified shape and either multivariate
or univariate pattern, in a specific region to represent the signal in the
neural data.

generate_stimfunction
Create a timecourse of the signal activation. This can be specified using event
onsets and durations from a timing file.

export_stimfunction:
Generate a three column timing file that can be used with software like FSL
to represent event event onsets and duration

double_gamma_hrf
Convolve the signal timecourse with the double gamma HRF to model the
expected evoked activity

apply_signal
Combine the signal volume with the HRF, thus giving the signal the temporal
properties of the HRF (such as smoothing and lag)

calc_noise
Estimate the noise properties of a given fMRI volume. Prominently, estimate
the smoothing and SFNR of the data

generate_noise
Create the noise for this run. This creates temporal, spatial task and white
noise. Various parameters can be tuned depending on need

mask_brain
Create a mask volume that has similar contrast as an fMRI image. Defaults to
use an MNI grey matter atlas but any image can be supplied to create an
estimate.

plot_brain
Display the brain, timepoint by timepoint, with above threshold voxels
highlighted against the outline of the brain.


 Authors:
 Cameron Ellis (Princeton) 2016-2017
 Chris Baldassano (Princeton) 2016-2017
"""
import logging

from itertools import product
import nitime.algorithms.autoregressive as ar
import math
import numpy as np
from pkg_resources import resource_stream
from scipy import stats
from scipy import signal
import scipy.ndimage as ndimage

__all__ = [
    "generate_signal",
    "generate_stimfunction",
    "export_stimfunction",
    "double_gamma_hrf",
    "apply_signal",
    "calc_noise",
    "generate_noise",
    "mask_brain",
    "plot_brain",
]

logger = logging.getLogger(__name__)


def _generate_feature(feature_type,
                      feature_size,
                      signal_magnitude,
                      thickness=1):
    """Generate features corresponding to signal

    Generate a single feature, that can be inserted into the signal volume

    Parameters
    ----------

    feature_type : str
        What shape signal is being inserted? Options are 'cube',
        'loop' (aka ring), 'cavity' (aka hollow sphere), 'sphere'.

    feature_size : int
        How big is the signal in diameter?

    signal_magnitude : float
        Set the signal size, a value of 1 means the signal is one standard
        deviation of the noise

    thickness : int
        How thick is the surface of the loop/cavity

    Returns
    ----------

    3 dimensional array
        The volume representing the signal

    """

    # If the size is equal to or less than 2 then all features are the same
    if feature_size <= 2:
        feature_type = 'cube'

    # What kind of signal is it?
    if feature_type == 'cube':

        # Preset the size of the signal
        signal = np.ones(np.power(feature_size, 3))

        # Reorganize the signal into a 3d matrix
        signal = signal.reshape([feature_size,
                                 feature_size,
                                 feature_size])

    elif feature_type == 'loop':

        # First make a cube of zeros
        signal = np.zeros(np.power(feature_size,
                                   3)).reshape([feature_size,
                                                feature_size,
                                                feature_size])

        # Make a mesh grid of the space
        seq = np.linspace(0, feature_size - 1,
                          feature_size)
        xx, yy = np.meshgrid(seq, seq)

        # Make a disk corresponding to the whole mesh grid
        xxmesh = (xx - ((feature_size - 1) / 2)) ** 2
        yymesh = (yy - ((feature_size - 1) / 2)) ** 2
        disk = xxmesh + yymesh

        # What are the limits of the rings being made
        outer_lim = disk[int((feature_size - 1) / 2), 0]
        inner_lim = disk[int((feature_size - 1) / 2), thickness]

        # What is the outer disk
        outer = disk <= outer_lim

        # What is the inner disk
        inner = disk <= inner_lim

        # Subtract the two disks to get a loop
        loop = outer != inner

        # If there is complete overlap then make the signal just the
        #  outer one
        if np.all(loop is False):
            loop = outer

        # store the loop
        signal[0:feature_size, 0:feature_size, int(np.round(feature_size /
                                                            2))] = loop

    elif feature_type == 'sphere' or feature_type == 'cavity':

        # Make a mesh grid of the space
        seq = np.linspace(0, feature_size - 1,
                          feature_size)
        xx, yy, zz = np.meshgrid(seq, seq, seq)

        # Make a disk corresponding to the whole mesh grid
        signal = ((xx - ((feature_size - 1) / 2)) ** 2 +
                  (yy - ((feature_size - 1) / 2)) ** 2 +
                  (zz - ((feature_size - 1) / 2)) ** 2)

        # What are the limits of the rings being made
        outer_lim = signal[int((feature_size - 1) / 2), int((feature_size -
                                                             1) / 2), 0]
        inner_lim = signal[int((feature_size - 1) / 2), int((feature_size -
                                                             1) / 2),
                           thickness]

        # Is the signal a sphere or a cavity?
        if feature_type == 'sphere':
            signal = signal <= outer_lim

        else:
            # Get the inner and outer sphere
            outer = signal <= outer_lim
            inner = signal <= inner_lim

            # Subtract the two disks to get a loop
            signal = outer != inner

            # If there is complete overlap then make the signal just the
            #  outer one
            if np.all(signal is False):
                signal = outer

    # Assign the signal magnitude
    signal = signal * signal_magnitude

    # Return the signal
    return signal


def _insert_idxs(feature_centre, feature_size, dimensions):
    """Returns the indices of where to put the signal into the signal volume

    Parameters
    ----------

    feature_centre : list, int
        List of coordinates for the centre location of the signal

    feature_size : list, int
        How big is the signal's diameter.

    dimensions : 3 length array, int
        What are the dimensions of the volume you wish to create


    Returns
    ----------
    x_idxs : tuple
        The x coordinates of where the signal is to be inserted

    y_idxs : tuple
        The y coordinates of where the signal is to be inserted

    z_idxs : tuple
        The z coordinates of where the signal is to be inserted

    """

    # Set up the indexes within which to insert the signal
    x_idx = [int(feature_centre[0] - (feature_size / 2)) + 1,
             int(feature_centre[0] - (feature_size / 2) +
                 feature_size) + 1]
    y_idx = [int(feature_centre[1] - (feature_size / 2)) + 1,
             int(feature_centre[1] - (feature_size / 2) +
                 feature_size) + 1]
    z_idx = [int(feature_centre[2] - (feature_size / 2)) + 1,
             int(feature_centre[2] - (feature_size / 2) +
                 feature_size) + 1]

    # Check for out of bounds
    # Min Boundary
    if 0 > x_idx[0]:
        x_idx[0] = 0
    if 0 > y_idx[0]:
        y_idx[0] = 0
    if 0 > z_idx[0]:
        z_idx[0] = 0

    # Max Boundary
    if dimensions[0] < x_idx[1]:
        x_idx[1] = dimensions[0]
    if dimensions[1] < y_idx[1]:
        y_idx[1] = dimensions[1]
    if dimensions[2] < z_idx[1]:
        z_idx[1] = dimensions[2]

    # Return the idxs for data
    return x_idx, y_idx, z_idx


def generate_signal(dimensions,
                    feature_coordinates,
                    feature_size,
                    feature_type,
                    signal_magnitude=[1],
                    signal_constant=1,
                    ):
    """Generate volume containing signal

    Generate signal, of a specific shape in specific regions, for a single
    volume. This will then be convolved with the HRF across time

    Parameters
    ----------

    dimensions : 1d array, ndarray
        What are the dimensions of the volume you wish to create

    feature_coordinates : multidimensional array
        What are the feature_coordinates of the signal being created.
        Be aware of clipping: features far from the centre of the
        brain will be clipped. If you wish to have multiple features
        then list these as a features x 3 array. To create a feature of
        a unique shape then supply all the individual
        feature_coordinates of the shape and set the feature_size to 1.

    feature_size : list, int
        How big is the signal. If feature_coordinates=1 then only one value is
        accepted, if feature_coordinates>1 then either one value must be
        supplied or m values

    feature_type : list, string
        What feature_type of signal is being inserted? Options are cube,
        loop, cavity, sphere. If feature_coordinates = 1 then
        only one value is accepted, if feature_coordinates > 1 then either
        one value must be supplied or m values

    signal_magnitude : list, float
        What is the (average) magnitude of the signal being generated? A
        value of 1 means that the signal is one standard deviation from the
        noise

    signal_constant : list, bool
        Is the signal constant across the feature (for univariate activity)
        or is it a random pattern of a given magnitude across the feature (for
        multivariate activity)

    Returns
    ----------
    volume_signal : 3 dimensional array, float
        Creates a single volume containing the signal

    """

    # Preset the volume
    volume_signal = np.zeros(dimensions)

    feature_quantity = round(feature_coordinates.shape[0])

    # If there is only one feature_size value then make sure to duplicate it
    # for all signals
    if len(feature_size) == 1:
        feature_size = feature_size * feature_quantity

    # Do the same for feature_type
    if len(feature_type) == 1:
        feature_type = feature_type * feature_quantity

    if len(signal_magnitude) == 1:
        signal_magnitude = signal_magnitude * feature_quantity

    # Iterate through the signals and insert in the data
    for signal_counter in list(range(0, feature_quantity)):

        # What is the centre of this signal
        if len(feature_size) > 1:
            feature_centre = np.asarray(feature_coordinates[signal_counter, ])
        else:
            feature_centre = np.asarray(feature_coordinates)[0]

        # Generate the feature to be inserted in the volume
        signal = _generate_feature(feature_type[signal_counter],
                                   feature_size[signal_counter],
                                   signal_magnitude[signal_counter],
                                   )

        # If the signal is a random noise pattern then multiply these ones by
        # a noise mask
        if signal_constant == 0:
            signal = signal * np.random.random([feature_size[signal_counter],
                                                feature_size[signal_counter],
                                                feature_size[signal_counter]])

        # Pull out the idxs for where to insert the data
        x_idx, y_idx, z_idx = _insert_idxs(feature_centre,
                                           feature_size[signal_counter],
                                           dimensions)

        # Insert the signal into the Volume
        volume_signal[x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], z_idx[0]:z_idx[
            1]] = signal

    return volume_signal


def generate_stimfunction(onsets,
                          event_durations,
                          total_time,
                          weights=[1],
                          timing_file=None,
                          temporal_resolution=1000.0,
                          ):
    """Return the function for the timecourse events

    When do stimuli onset, how long for and to what extent should you
    resolve the fMRI time course. There are two ways to create this, either
    by supplying onset, duration and weight information or by supplying a
    timing file (in the three column format used by FSL).

    Parameters
    ----------

    onsets : list, int
        What are the timestamps (in s) for when an event you want to
        generate onsets?

    event_durations : list, int
        What are the durations (in s) of the events you want to
        generate? If there is only one value then this will be assigned
        to all onsets

    total_time : int
        How long (in s) is the experiment in total.

    weights : list, float
        What is the weight for each event (how high is the box car)? If
        there is only one value then this will be assigned to all onsets

    timing_file : string
        The filename (with path) to a three column timing file (FSL) to
        make the events. Still requires total_time to work

    temporal_resolution : float
        How many elements per second are you modeling for the
        timecourse. This is useful when you want to model the HRF at an
        arbitrarily high resolution (and then downsample to your TR later).

    Returns
    ----------

    Iterable[float]
        The time course of stimulus evoked activation

    """

    # If the timing file is supplied then use this to acquire the
    if timing_file is not None:

        # Read in text file line by line
        with open(timing_file) as f:
            text = f.readlines()  # Pull out file as a an array

        # Preset
        onsets = list()
        event_durations = list()
        weights = list()

        # Pull out the onsets, weights and durations, set as a float
        for line in text:
            onset, duration, weight = line.strip().split()
            onsets.append(float(onset))
            event_durations.append(float(duration))
            weights.append(float(weight))

    # If only one duration is supplied then duplicate it for the length of
    # the onset variable
    if len(event_durations) == 1:
        event_durations = event_durations * len(onsets)

    if len(weights) == 1:
        weights = weights * len(onsets)

    # Generate the time course as empty, each element is a millisecond by
    # default
    stimfunction = [0] * int(round(total_time * temporal_resolution))

    # Cycle through the onsets
    for onset_counter in list(range(len(onsets))):
        # Adjust for the resolution
        onset_idx = int(np.floor(onsets[onset_counter] * temporal_resolution))

        # Adjust for the resolution
        offset_idx = int(np.floor((onsets[onset_counter] + event_durations[
            onset_counter])) * temporal_resolution)

        # For the appropriate number of indexes and duration, make this value 1
        idx_n = round(event_durations[onset_counter] * temporal_resolution)
        stimfunction[onset_idx:offset_idx] = [weights[onset_counter]] * idx_n

    # Shorten the data if it's too long
    if len(stimfunction) > total_time * temporal_resolution:
        stimfunction = stimfunction[0:int(total_time * temporal_resolution)]

    return stimfunction


def export_stimfunction(stimfunction,
                        filename,
                        temporal_resolution=1000.0
                        ):
    """ Output a tab separated three column timing file

    This produces a three column tab separated text file, with the three
    columns representing onset time (s), event duration (s) and weight,
    respectively. Useful if you want to run the simulated data through FEAT
    analyses

    Parameters
    ----------

    stimfunction : list
        The stimulus function describing the time course of events. For
        instance output from generate_stimfunction.

    filename : str
        The name of the three column text file to be output

    temporal_resolution : float
        How many elements per second are you modeling with the
        stimfunction?

    """

    # Iterate through the stim function
    stim_counter = 0
    event_counter = 0
    while stim_counter < len(stimfunction):

        # Is it an event?
        if stimfunction[stim_counter] != 0:

            # When did the event start?
            event_onset = str(stim_counter / temporal_resolution)

            # The weight of the stimulus
            weight = str(stimfunction[stim_counter])

            # Reset
            event_duration = 0

            # Is the event still ongoing?
            while stimfunction[stim_counter] != 0 & stim_counter <= len(
                    stimfunction):

                # Add one millisecond to each duration
                event_duration = event_duration + 1

                # Increment
                stim_counter = stim_counter + 1

            # How long was the event in seconds
            event_duration = str(event_duration / temporal_resolution)

            # Append this row to the data file
            with open(filename, "a") as file:
                file.write(event_onset + '\t' + event_duration + '\t' +
                           weight + '\n')

            # Increment the number of events
            event_counter = event_counter + 1

        # Increment
        stim_counter = stim_counter + 1


def double_gamma_hrf(stimfunction,
                     tr_duration,
                     response_delay=6,
                     undershoot_delay=12,
                     response_dispersion=0.9,
                     undershoot_dispersion=0.9,
                     response_scale=1,
                     undershoot_scale=0.035,
                     scale_function=1,
                     temporal_resolution=1000.0,
                     ):
    """Convolve the double gamma HRF with the timecourse evoked activity

    Parameters
    ----------

    stimfunction : list, bool
        What is the time course of events to be modelled in this
        experiment

    tr_duration : float
        How long (in s) between each volume onset

    response_delay : float
        How many seconds until the peak of the HRF

    undershoot_delay : float
        How many seconds until the trough of the HRF

    response_dispersion : float
        How wide is the rising peak dispersion

    undershoot_dispersion : float
        How wide is the undershoot dispersion

    response_scale : float
         How big is the response relative to the peak

    undershoot_scale :float
        How big is the undershoot relative to the trough

    scale_function : bool
        Do you want to scale the function to a range of 1

    temporal_resolution : float
        How many elements per second are you modeling for the stimfunction
    Returns
    ----------

    one dimensional array
        The time course of the HRF convolved with the stimulus function


    """

    hrf_length = 30  # How long is the HRF being created

    # How many seconds of the HRF will you model?
    hrf = [0] * int(hrf_length * temporal_resolution)

    # When is the peak of the two aspects of the HRF
    response_peak = response_delay * response_dispersion
    undershoot_peak = undershoot_delay * undershoot_dispersion

    for hrf_counter in list(range(len(hrf) - 1)):

        # Specify the elements of the HRF for both the response and undershoot
        resp_pow = math.pow((hrf_counter / temporal_resolution) /
                            response_peak, response_delay)
        resp_exp = math.exp(-((hrf_counter / temporal_resolution) -
                              response_peak) /
                            response_dispersion)

        response_model = response_scale * resp_pow * resp_exp

        undershoot_pow = math.pow((hrf_counter / temporal_resolution) /
                                  undershoot_peak,
                                  undershoot_delay)
        undershoot_exp = math.exp(-((hrf_counter / temporal_resolution) -
                                    undershoot_peak /
                                    undershoot_dispersion))

        undershoot_model = undershoot_scale * undershoot_pow * undershoot_exp

        # For this time point find the value of the HRF
        hrf[hrf_counter] = response_model - undershoot_model

    # Convolve the hrf that was created with the boxcar input
    signal_function = np.convolve(stimfunction, hrf)

    # Decimate the signal function so that it only has one element per TR
    decimate_interval = int(tr_duration * temporal_resolution)
    signal_function = signal_function[0::decimate_interval]

    # Cut off the HRF
    signal_function = signal_function[0:int((len(stimfunction) /
                                             tr_duration) /
                                            temporal_resolution)]

    # Scale the function so that the peak response is 1
    if scale_function == 1:
        signal_function = signal_function / np.max(signal_function)

    return signal_function


def apply_signal(signal_function,
                 volume_signal,
                 ):
    """Combine the signal volume with its timecourse

    Apply the convolution of the HRF and stimulus time course to the
    volume.

    Parameters
    ----------

    signal_function : list, float
        The one dimensional timecourse of the signal over time.
        Found by convolving the HRF with the stimulus time course.

    volume_signal : multi dimensional array, float
        The volume containing the signal to be convolved


    Returns
    ----------
    multidimensional array, float
        The convolved signal volume

    """

    # Preset volume
    signal = np.ndarray([volume_signal.shape[0], volume_signal.shape[
        1], volume_signal.shape[2], len(signal_function)])

    # Iterate through the brain, multiplying the volume by the HRF
    for tr_counter in list(range(0, len(signal_function))):
        signal[0:volume_signal.shape[0],
               0:volume_signal.shape[1],
               0:volume_signal.shape[2],
               tr_counter] = signal_function[tr_counter] * volume_signal

    return signal


def _calc_fwhm(volume,
               mask,
               voxel_size=[1.0, 1.0, 1.0],
               ):
    """ Calculate the FWHM of a volume
    Estimates the FWHM (mm) of a volume's non-masked voxels

    Parameters
    ----------

    volume : 3 dimensional array
        Functional data to have the FWHM measured.

    mask : 3 dimensional array
        A binary mask of the brain voxels in volume

    voxel_size : length 3 list, float
        Millimeters per voxel for x, y and z.

    Returns
    -------

    float, list
        Returns the FWHM of each TR in mm

    """

    # What are the dimensions of the volume
    dimensions = volume.shape

    # Iterate through the TRs, creating a FWHM for each TR

    # Preset
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
            v_sum += np.abs(volume[x, y, z])
            v_sq += volume[x, y, z] ** 2

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
            included = in_mask and (~np.isnan(volume[x + 1, y, z]))
            if included:
                d_sum[0] += volume[x, y, z] - volume[x + 1, y, z]
                d_sq[0] += (volume[x, y, z] - volume[x + 1, y, z]) ** 2
                d_count[0] += 1

            in_range = (y < dimensions[1] - 1)
            in_mask = in_range and (mask[x, y + 1, z] > 0)
            included = in_mask and (~np.isnan(volume[x, y + 1, z]))
            if included:
                d_sum[1] += volume[x, y, z] - volume[x, y + 1, z]
                d_sq[1] += (volume[x, y, z] - volume[x, y + 1, z]) ** 2
                d_count[1] += 1

            in_range = (z < dimensions[2] - 1)
            in_mask = in_range and (mask[x, y, z + 1] > 0)
            included = in_mask and (~np.isnan(volume[x, y, z + 1]))
            if included:
                d_sum[2] += volume[x, y, z] - volume[x, y, z + 1]
                d_sq[2] += (volume[x, y, z] - volume[x, y, z + 1]) ** 2
                d_count[2] += 1

    # Find the variance
    d_var = np.divide((d_sq - np.divide(np.power(d_sum, 2),
                                        d_count)), (np.add(d_count, -1)))

    o_var = np.divide(-1, (4 * np.log(1 - (0.5 * d_var / v_var))))
    fwhm3 = np.sqrt(o_var) * 2 * np.sqrt(2 * np.log(2))
    fwhm = np.prod(np.multiply(fwhm3, voxel_size)) ** (1 / 3)

    return fwhm


def _calc_temporal_noise(volume,
                         mask,
                         auto_reg_order=1,
                         ):
    """ Calculate the the temporal noise of a volume
    This calculates the variability of the volume over time, the SFNR of the
    volume and the proportion of variance over time that is due to
    autoregression and how much is due to scanner drift.

    Parameters
    ----------

    volume : 4d array, float
        Take a volume time series to extract the middle slice from the
        middle TR

    mask : 3d array, binary
        A binary mask the same size as the volume

    auto_reg_order : int
        What order of the autoregression do you want to pull out


    Returns
    -------

    float 
        The standard deviation of brain voxels   across time.

    float 
        The SFNR of the volume (mean brain activity divided by  temporal
        variability in the averaged non brain voxels)  

    float
        A sigma of the autoregression in the data

    float
        Sigma of the drift in the data

    """

    # What are the midpoints to be extracted
    mid_x_idx = int(np.ceil(volume.shape[0] / 2))
    mid_tr_idx = int(np.ceil(volume.shape[3] / 2))

    # Pull out the slices
    slice_volume = volume[mid_x_idx, :, :, mid_tr_idx]
    slice_mask = mask[mid_x_idx, :, :]

    # Divide by the mask (if the middle slice was low in grey matter mass
    # then you would have a lower mean signal by default)
    mean_signal = (slice_volume[slice_mask > 0]).mean()

    # What are the brain and non-brain voxels
    brain_voxels = volume[mask > 0]
    mask_voxels = volume[mask == 0]

    # What is the noise in the masked voxels (average the variance)
    mask_noise = ((np.std(mask_voxels, 1) ** 2).mean()) ** 0.5

    # Assume the background noise is a combination of random + drift. Thus
    # average all masked voxels and find the variation over time, this is the
    # variance due to drift.
    drift_noise = np.mean(mask_voxels, 0).std()

    # Subtract the drift variance from total mask noise to find the system
    # noise
    background_noise = np.sqrt(mask_noise ** 2 - drift_noise ** 2)

    # Find the noise to brain voxels
    temporal_noise = np.std(brain_voxels, 1).mean()

    # Convert temporal noise into percent signal change
    temporal_noise = temporal_noise / mean_signal * 100

    # Calculate sfnr and convert from memmap
    sfnr = float(mean_signal / background_noise)

    # Calculate the time course
    timecourse = np.mean(volume[mask > 0], 0)

    # Pull out the AR values (depends on order)
    auto_reg_sigma = ar.AR_est_YW(timecourse, auto_reg_order)[1]
    auto_reg_sigma = np.sqrt(auto_reg_sigma)

    # What is the size of the change in the time course
    drift_sigma = timecourse.std().tolist()

    return temporal_noise, sfnr, auto_reg_sigma, drift_sigma


def calc_noise(volume,
               mask=None,
               noise_dict=None,
               ):
    """ Calculates the noise properties of the volume supplied.
    This estimates what noise properties the volume has. For instance it
    determines the spatial smoothness, the autoregressive noise, system
    noise etc. Read the doc string for generate_noise to understand how
    these different types of noise interact.

    Parameters
    ----------

    volume : 4d numpy array, float
        Take in a functional volume (either the file name or the numpy
        array) to be used to estimate the noise properties of this

    mask : 3d numpy array, binary
        A binary mask of the brain, the same size as the volume

    Returns
    -------

    dict
        Return a dictionary of the calculated noise parameters of the provided
        dataset

    """

    # Create the mask if not supplied and set the mask size
    if mask is None:
        mask = np.ones(volume.shape)[:, :, :, 0]

    # Update noise dict if it is not yet created
    if noise_dict is None:
        noise_dict = {'voxel_size': [1.0, 1.0, 1.0]}
    elif 'voxel_size' not in noise_dict:
        noise_dict['voxel_size'] = [1.0, 1.0, 1.0]

    # What is the max activation of the mean of this voxel (allows you to
    # convert between the mask and the mean of the brain volume)
    noise_dict['max_activity'] = np.mean(volume, 3).max()

    # Since you are deriving the 'true' values then you want your noise to
    # be set to that level

    # Calculate the temporal variability of the volume
    temporal_noise, sfnr, auto_reg, drift = _calc_temporal_noise(volume, mask)
    noise_dict['temporal_noise'] = temporal_noise
    noise_dict['sfnr'] = sfnr

    # Calculate the fwhm on a subset of volumes
    if volume.shape[3] > 100:
        # Take only 100 shuffled TRs
        trs = np.arange(volume.shape[3])
        np.random.shuffle(trs)
        trs = trs[0:100]
    else:
        trs = list(range(0, volume.shape[3]))

    # Go through the trs and pull out the fwhm
    fwhm = [0] * len(trs)
    for tr in list(range(0, len(trs))):
        fwhm[tr] = _calc_fwhm(volume[:, :, :, trs[tr]],
                              mask,
                              noise_dict['voxel_size'],
                              )

    # Keep only the mean
    noise_dict['fwhm'] = np.mean(fwhm)

    # Calibrate for how sigma is originally calculated
    auto_reg_sigma = auto_reg / noise_dict['temporal_noise']

    # Total temporal noise, since these values only make sense relatively
    total_temporal_noise = auto_reg_sigma + drift

    # What proportion of noise is accounted for by these variables?
    noise_dict['auto_reg_sigma'] = auto_reg_sigma / total_temporal_noise
    noise_dict['drift_sigma'] = drift / total_temporal_noise

    # Return the noise dictionary
    return noise_dict


def _generate_noise_system(dimensions_tr,
                           noise_type='exponential',
                           ):
    """Generate the scanner noise

    Generate system noise, either rician or exponential, for the scanner.
    Low SNR scans tend to have rician noise whereas high SNR scans (>30) are
    better modelled by exponential noise. Generates a distribution with a SD
    of 1.

    Parameters
    ----------

    dimensions_tr : n length array, int
        What are the dimensions of the volume you wish to insert
        noise into. This can be a volume of any size

    noise_type : str
        String specifying the noise type. Rician is appropriate when the SNR is
        low but is insufficiently skewed to appropriately model high SNR data.

    Returns
    ----------

    system_noise : multidimensional array, float
        Create a volume with system noise

    """

    if noise_type == 'rician':
        # Generate the Rician noise (has an SD of 1)
        system_noise = stats.rice.rvs(b=0, loc=0, scale=1.527,
                                      size=dimensions_tr)

    elif noise_type == 'exponential':
        # Make an exponential distribution (has an SD of 1)
        system_noise = stats.expon.rvs(0, scale=1, size=dimensions_tr)

    return system_noise


def _generate_noise_temporal_task(stimfunction_tr,
                                  motion_noise='gaussian',
                                  ):
    """Generate the signal dependent noise

    Create noise specific to the signal, for instance there is variability
    in how the signal manifests on each event

    Parameters
    ----------

    stimfunction_tr : 1 Dimensional array
        This is the timecourse of the stimuli in this experiment,
        each element represents a TR

    motion_noise : str
        What type of noise will you generate? Can be gaussian or rician

    Returns
    ----------

    one dimensional array, float
        Generates the temporal task noise timecourse

    """

    # Make the noise to be added
    if motion_noise == 'gaussian':
        noise = stimfunction_tr * np.random.normal(0, 1, size=len(
            stimfunction_tr))
    elif motion_noise == 'rician':
        noise = stimfunction_tr * stats.rice.rvs(0, 1, size=len(
            stimfunction_tr))

    noise_task = stimfunction_tr + noise

    # Normalize
    noise_task = stats.zscore(noise_task)

    return noise_task


def _generate_noise_temporal_drift(trs,
                                   tr_duration,
                                   period=150,
                                   ):

    """Generate the drift noise

    Create a sinewave, of a given period and random phase, to represent the
    drift of the signal over time

    Parameters
    ----------

    trs : int
        How many volumes (aka TRs) are there

    tr_duration : float
        How long in seconds is each volume acqusition

    Returns
    ----------
    one dimensional array, float
        The drift timecourse of activity

    """

    # Calculate the cycles of the drift for a given function.
    cycles = trs * tr_duration / period

    # Create a sine wave with a given number of cycles and random phase
    timepoints = np.linspace(0, trs - 1, trs)
    phaseshift = np.pi * 2 * np.random.random()
    phase = (timepoints / (trs - 1) * cycles * 2 * np.pi) + phaseshift
    noise_drift = np.sin(phase)

    # Normalize
    noise_drift = stats.zscore(noise_drift)

    # Return noise
    return noise_drift


def _generate_noise_temporal_autoregression(timepoints,
                                            auto_reg_order=1,
                                            auto_reg_rho=[1],
                                            ):

    """Generate the autoregression noise
    Make a slowly drifting timecourse with the given autoregression parameters

    Parameters
    ----------

    timepoints : 1 Dimensional array
        What time points are sampled by a TR

    auto_reg_order : float
        How many timepoints ought to be taken into consideration for the
        autoregression function

    auto_reg_rho : float
        What is the scaling factor on the predictiveness of the previous
        time point

    Returns
    ----------
    one dimensional array, float
        Generates the autoregression noise timecourse

    """

    if len(auto_reg_rho) == 1:
        auto_reg_rho = auto_reg_rho * auto_reg_order  # Duplicate this so that
        # there is one
        #  for each value

    noise_autoregression = []
    for tr_counter in list(range(0, len(timepoints))):

        if tr_counter == 0:
            noise_autoregression.append(np.random.normal(0, 1))

        else:

            temp = []
            for pCounter in list(range(1, auto_reg_order + 1)):
                if tr_counter - pCounter >= 0:
                    past_trs = noise_autoregression[int(tr_counter - pCounter)]
                    past_reg = auto_reg_rho[pCounter - 1]
                    random = np.random.normal(0, 1)
                    temp.append(past_trs * past_reg + random)

                    noise_autoregression.append(np.mean(temp))

    # Normalize
    noise_autoregression = stats.zscore(noise_autoregression)

    return noise_autoregression


def _generate_noise_temporal_phys(timepoints,
                                  resp_freq=0.2,
                                  heart_freq=1.17,
                                  ):
    """Generate the physiological noise.
    Create noise representing the heart rate and respiration of the data

    Parameters
    ----------

    timepoints : 1 Dimensional array
        What time points are sampled by a TR

    resp_freq : float
        What is the frequency of respiration

    heart_freq : float
        What is the frequency of heart beat

    Returns
    ----------
    one dimensional array, float
        Generates the physiological temporal noise timecourse

    """

    noise_phys = []  # Preset
    for tr_counter in timepoints:
        # Calculate the radians for each variable at this given TR
        resp_radians = resp_freq * tr_counter * 2 * np.pi
        heart_radians = heart_freq * tr_counter * 2 * np.pi

        # Combine the two types of noise and append
        noise_phys.append(np.cos(resp_radians) + np.sin(heart_radians) +
                          np.random.normal(0, 1))

    # Normalize
    noise_phys = stats.zscore(noise_phys)

    return noise_phys


def _generate_noise_spatial(dimensions,
                            template=None,
                            mask=None,
                            fwhm=4.0,
                            ):
    """Generate code for Gaussian Random Fields.

    Adapted from code found here:
    http://andrewwalker.github.io/statefultransitions/post/gaussian-fields/
    with permission from the author:
    https://twitter.com/walkera101/status/785578499440377858. Original code
    comes from http://mathematica.stackexchange.com/questions/4829
    /efficiently-generating-n-d-gaussian-random-fields with a WTFPL (
    http://www.wtfpl.net).

    Parameters
    ----------

    dimensions : 3 length array, int
        What is the shape of the volume to be generated

    template : 3d array, float
        A continuous (0 -> 1) volume describing the likelihood a voxel is in
        the brain. This can be used to contrast the brain and non brain.

    mask : 3 dimensional array, binary
        The masked brain, thresholded to distinguish brain and non-brain

    fwhm : float
        What is the full width half max of the gaussian fields being created.
        This is converted into a sigma which is used in this function.
        However, this conversion was found empirically by testing values of
        sigma and how it relates to fwhm values. The relationship that would be
        found in such a test depends on the size of the brain (bigger brains
        can have bigger fwhm).
        However, small errors shouldn't matter too much since the fwhm
        generated here can only be approximate anyway: firstly, although the
        distribution that is being drawn from is set to this value,
        this will manifest differently on every draw. Secondly, because of
        the masking and dimensions of the generated volume, this does not
        behave simply- wrapping effects matter (the outputs are
        closer to the input value if you have no mask).
        Use _calc_fwhm on this volume alone if you have concerns about the
        accuracy of the fwhm.

    Returns
    ----------

    3d array, float
        Generates the spatial noise volume for these parameters

    """

    if len(dimensions) == 4:
        return

    def logfunc(x, a, b, c):
        """Solve for y given x for log function.

            Parameters
            ----------

            x : float
                x value of log function

            a : float
                x shift of function

            b : float
                rate of change

            c : float
                y shift of function

            Returns
            ----------

            float
                y value of log function
            """
        return (np.log(x + a) / np.log(b)) + c

    # Convert from fwhm to sigma (relationship discovered empirical, only an
    #  approximation up to sigma = 0 -> 5 which corresponds to fwhm = 0 -> 8,
    # relies on an assumption of brain size).
    spatial_sigma = logfunc(fwhm, -0.36778719, 2.10601011, 2.15439247)

    # Set up the input to the fast fourier transform
    def fftIndgen(n):
        a = list(range(0, int(n / 2 + 1)))
        b = list(range(1, int(n / 2)))
        b.reverse()
        b = [-i for i in b]
        return a + b

    # Take in an array of fft values and determine the amplitude for those
    # values
    def Pk2(idxs):

        # If all the indexes are zero then set the out put to zero
        if np.all(idxs == 0):
            return 0.0
        return np.sqrt(np.sqrt(np.sum(idxs ** 2)) ** (-1 * spatial_sigma))

    noise = np.fft.fftn(np.random.normal(size=dimensions))
    amplitude = np.zeros(dimensions)

    for x, fft_x in enumerate(fftIndgen(dimensions[0])):
        for y, fft_y in enumerate(fftIndgen(dimensions[1])):
            for z, fft_z in enumerate(fftIndgen(dimensions[2])):
                amplitude[x, y, z] = Pk2(np.array([fft_x, fft_y, fft_z]))

    # The output
    noise_spatial = np.fft.ifftn(noise * amplitude)

    # Mask or not, then z score
    if mask is not None and template is not None:

        # Mask the output
        noise_spatial = noise_spatial.real * template

        # Z score the specific to the brain
        noise_spatial[mask > 0] = stats.zscore(noise_spatial[mask > 0])
    else:
        noise_spatial = stats.zscore(noise_spatial.real)

    return noise_spatial


def _generate_noise_temporal(stimfunction_tr,
                             tr_duration,
                             dimensions,
                             template,
                             mask,
                             fwhm,
                             motion_sigma,
                             drift_sigma,
                             auto_reg_sigma,
                             physiological_sigma,
                             ):
    """Generate the temporal noise
    Generate the time course of the average brain voxel. To increase or
    decrease the amount of total noise change the temporal_noise noise_dict
    entry. To change the relative mixing of the noise components, change the
    sigma's specified below.

    Parameters
    ----------

    stimfunction_tr : 1 Dimensional array
        This is the timecourse of the stimuli in this experiment,
        each element represents a TR

    tr_duration : int
        How long is a TR, in seconds

    dimensions : 3 length array, int
        What is the shape of the volume to be generated

    template : 3d array, float
        A continuous (0 -> 1) volume describing the likelihood a voxel is in
        the brain. This can be used to contrast the brain and non brain.

    mask : 3 dimensional array, binary
        The masked brain, thresholded to distinguish brain and non-brain

    fwhm : float
        What is the full width half max of the gaussian fields being created
        to model spatial noise.

    motion_sigma : float
        How much noise is left over after pre-processing has been
        done. This is noise specifically on the task events

    drift_sigma : float

        What is the sigma on the size of the sine wave

    auto_reg_sigma : float, list
        How large is the sigma on the autocorrelation. Higher means more
        variable over time. If there are multiple entries then this is
        inferred as higher orders of the autoregression

    physiological_sigma : float

        How variable is the signal as a result of physiology,
        like heart beat and breathing

    Returns
    ----------

    one dimensional array, float
        Generates the temporal noise timecourse for these parameters

    """

    # Set up common parameters
    # How many TRs are there
    trs = len(stimfunction_tr)

    # What time points are sampled by a TR?
    timepoints = list(range(0, trs * tr_duration))[::tr_duration]

    noise_drift = _generate_noise_temporal_drift(trs,
                                                 tr_duration,
                                                 )

    noise_phys = _generate_noise_temporal_phys(timepoints,
                                               )

    noise_autoregression = _generate_noise_temporal_autoregression(timepoints,
                                                                   )

    # Generate the volumes that will differ depending on the type of noise
    # that it will be used for
    volume_drift = np.ones(dimensions)

    volume_phys = _generate_noise_spatial(dimensions=dimensions,
                                          template=template,
                                          mask=mask,
                                          fwhm=fwhm,
                                          )

    volume_autoreg = _generate_noise_spatial(dimensions=dimensions,
                                             template=template,
                                             mask=mask,
                                             fwhm=fwhm,
                                             )

    # Multiply the noise by the spatial volume
    noise_drift_volume = np.multiply.outer(volume_drift, noise_drift)
    noise_phys_volume = np.multiply.outer(volume_phys, noise_phys)
    noise_autoregression_volume = np.multiply.outer(volume_autoreg,
                                                    noise_autoregression)

    # Sum the noise (it would have been nice to just add all of them in a
    # single line but this was causing formatting problems)
    noise_temporal = noise_drift_volume * drift_sigma
    noise_temporal = noise_temporal + (noise_phys_volume * physiological_sigma)
    noise_temporal = noise_temporal + (noise_autoregression_volume *
                                       auto_reg_sigma)

    # Only do this if you are making motion variance
    if motion_sigma != 0 and np.sum(stimfunction_tr) > 0:
        # Make each noise type
        noise_task = _generate_noise_temporal_task(stimfunction_tr,
                                                   )
        volume_task = _generate_noise_spatial(dimensions=dimensions,
                                              template=template,
                                              mask=mask,
                                              fwhm=fwhm,
                                              )
        noise_task_volume = np.multiply.outer(volume_task, noise_task)
        noise_temporal = noise_temporal + (noise_task_volume * motion_sigma)

    # Finally, z score each voxel so things mix nicely
    noise_temporal = stats.zscore(noise_temporal, 3)

    return noise_temporal


def mask_brain(volume,
               template_name=None,
               mask_threshold=None,
               mask_self=0,
               ):
    """ Mask the simulated volume
    This creates a mask specifying the likelihood (kind of) a voxel is
    part of the brain. All values are bounded to the range of 0 to 1. An
    appropriate threshold to isolate brain voxels is >0.2

    Parameters
    ----------

    volume : multidimensional array
        Either numpy array of a volume or a tuple describing the dimensions
        of the mask to be created

    template_name : str
        What is the path to the template to be loaded? If empty then it
        defaults to an MNI152 grey matter mask.

    mask_threshold : float
        What is the threshold (0 -> 1) for including a voxel in the mask? If
        None then the program will try and identify the last wide peak in a
        histogram of the template (assumed to be the brain voxels) and takes
        the minima before that peak as the threshold. Won't work when the
        data is not bimodal.

    mask_self : bool
        If set to true then it makes a mask from the volume supplied (by
        averaging across time points and changing the range).

    Returns
    ----------

    mask : 3 dimensional array, binary
        The masked brain, thresholded to distinguish brain and non-brain

    template : 3 dimensional array, float
        A continuous (0 -> 1) volume describing the likelihood a voxel is in
        the brain. This can be used to contrast the brain and non brain.

    """

    # If the volume supplied is a 1d array then output a volume of the
    # supplied dimensions
    if len(volume.shape) == 1:
        volume = np.ones(volume)

    # Load in the mask
    if mask_self is True:
        mask_raw = volume
    elif template_name is None:
        mask_raw = np.load(resource_stream(__name__, "grey_matter_mask.npy"))
    else:
        mask_raw = np.load(template_name)

    # Make the masks 3d
    if len(mask_raw.shape) == 3:
        mask_raw = np.array(mask_raw)
    elif len(mask_raw.shape) == 4 and mask_raw.shape[3] == 1:
        mask_raw = np.array(mask_raw[:, :, :, 0])
    else:
        mask_raw = np.mean(mask_raw, 3)

    # Find the max value (so you can calulate these as proportions)
    mask_max = mask_raw.max()

    # Make sure the mask values range from 0 to 1 (make out of max of volume
    #  so that this is invertible later)
    mask_raw = mask_raw / mask_max

    # If there is only one brain volume then make this a forth dimension
    if len(volume.shape) == 3:
        temp = np.zeros([volume.shape[0], volume.shape[1], volume.shape[2], 1])
        temp[:, :, :, 0] = volume
        volume = temp

    # Reshape the mask to be the size as the brain
    brain_dim = volume.shape
    mask_dim = mask_raw.shape

    zoom_factor = (brain_dim[0] / mask_dim[0],
                   brain_dim[1] / mask_dim[1],
                   brain_dim[2] / mask_dim[2],
                   )

    # Scale the mask according to the input brain
    # You might get a warning but ignore it
    template = ndimage.zoom(mask_raw, zoom_factor, order=2)

    # If the mask threshold is not supplied then guess it is a minima
    # between the two peaks of the bimodal distribution of voxel activity
    if mask_threshold is None:

        # Make the histogram
        template_vector = template.reshape(brain_dim[0] * brain_dim[1] *
                                           brain_dim[2])
        template_hist = np.histogram(template_vector, 100)

        # Identify the last peak
        peak = signal.argrelmax(template_hist[0], order=5)[0].max()

        # What is the threshold
        minima = template_hist[0][0:peak].min()
        minima_idx = np.where(template_hist[0] == minima)
        mask_threshold = template_hist[1][minima_idx][0]

    # Mask the template based on the threshold
    mask = np.zeros(template.shape)
    mask[template > mask_threshold] = 1

    return mask, template


def _noise_dict_update(noise_dict):
    """
    Update the noise dictionary parameters with default values, in case any
    were missing

    Parameters
    ----------

    noise_dict : dict
        A dictionary specifying the types of noise in this experiment. The
        noise types interact in important ways. First, all noise types
        ending with sigma (e.g. motion sigma) are mixed together in
        _generate_temporal_noise. These values describe the proportion of
        mixing of these elements. However critically, temporal_noise is the
        parameter that describes how much noise these components contribute
        to the brain.

    Returns
    -------

    Updated dictionary

    """

    # Check what noise is in the dictionary and add if necessary. Numbers
    # determine relative proportion of noise

    if 'temporal_noise' not in noise_dict:
        noise_dict['temporal_noise'] = 5
    if 'motion_sigma' not in noise_dict:
        noise_dict['motion_sigma'] = 0
    if 'drift_sigma' not in noise_dict:
        noise_dict['drift_sigma'] = 0.45
    if 'auto_reg_sigma' not in noise_dict:
        noise_dict['auto_reg_sigma'] = 0.45
    if 'physiological_sigma' not in noise_dict:
        noise_dict['physiological_sigma'] = 0.1
    if 'sfnr' not in noise_dict:
        noise_dict['sfnr'] = 30
    if 'max_activity' not in noise_dict:
        noise_dict['max_activity'] = 1000
    if 'voxel_size' not in noise_dict:
        noise_dict['voxel_size'] = [1.0, 1.0, 1.0]
    if 'fwhm' not in noise_dict:
        noise_dict['fwhm'] = 4

    return noise_dict


def generate_noise(dimensions,
                   stimfunction_tr,
                   tr_duration,
                   template,
                   mask=None,
                   noise_dict=None,
                   ):
    """ Generate the noise to be added to the signal.
    Default noise parameters will create a noise volume with a standard
    deviation of 0.1 (where the signal defaults to a value of 1). This has
    built into estimates of how different types of noise mix. All noise
    values can be set by the user or estimated with calc_noise.

    Parameters
    ----------

    dimensions : nd array
        What is the shape of the volume to be generated

    stimfunction_tr :  Iterable, list
        When do the stimuli events occur. Each element is a TR

    tr_duration : float
        What is the duration, in seconds, of each TR?

    template : 3d array, float
        A continuous (0 -> 1) volume describing the likelihood a voxel is in
        the brain. This can be used to contrast the brain and non brain.

    mask : 3d array, binary
        The mask of the brain volume, distinguishing brain from non-brain

    noise_dict : dictionary, float
        This is a dictionary which describes the noise parameters of the
        data. If there are no other variables provided then it will use default
        values

    Returns
    ----------

    multidimensional array, float
        Generates the noise volume for these parameters

    """

    # Change to be an empty dictionary if it is None
    if noise_dict is None:
        noise_dict = {}

    # Take in the noise dictionary and determine whether
    noise_dict = _noise_dict_update(noise_dict)

    # What are the dimensions of the volume, including time
    dimensions_tr = (dimensions[0],
                     dimensions[1],
                     dimensions[2],
                     len(stimfunction_tr))

    # Get the mask of the brain and set it to be 3d
    if mask is None:
        mask = np.ones(dimensions)

    # Generate the noise
    noise_temporal = _generate_noise_temporal(stimfunction_tr=stimfunction_tr,
                                              tr_duration=tr_duration,
                                              dimensions=dimensions,
                                              mask=mask,
                                              template=template,
                                              fwhm=noise_dict[
                                                 'fwhm'],
                                              motion_sigma=noise_dict[
                                                 'motion_sigma'],
                                              drift_sigma=noise_dict[
                                                 'drift_sigma'],
                                              auto_reg_sigma=noise_dict[
                                                 'auto_reg_sigma'],
                                              physiological_sigma=noise_dict[
                                                 'physiological_sigma'],
                                              )

    # Create the base (this inverts the process to make the template)
    base = template * noise_dict['max_activity']

    # Set the amount of background based on the SFNR value

    # What are the midpoints to be extracted
    mid_x_idx = int(np.ceil(base.shape[0] / 2))

    # Pull out the slices
    slice_volume = base[mid_x_idx, :, :]
    slice_mask = mask[mid_x_idx, :, :]

    # What is the mean signal of the non masked voxels in this slice?
    mean_signal = (slice_volume[slice_mask > 0]).mean()

    # Set up the machine noise
    noise_system = _generate_noise_system(dimensions_tr=dimensions_tr)

    # What is the standard deviation of the background activity
    # (N.B. You need to subtract the mean system noise from this number
    # since system_sigma * noise_system.mean() will later be added to the base)
    system_sigma = mean_signal / (noise_dict['sfnr'] - noise_system.mean())

    # Increase the size of the system noise based on the SFNR
    noise_system *= system_sigma

    # Convert temporal noise (in percent) to real numbers
    abs_change = noise_dict['temporal_noise'] * mean_signal / 100

    # Reshape the base (to be the same size as the volume to be created)
    base = base.reshape(dimensions[0], dimensions[1], dimensions[2], 1)
    base = np.ones(dimensions_tr) * base

    # Sum up the noise of the brain
    noise = base + (noise_temporal * abs_change) + noise_system

    # Reject negative values (only happens outside of the brain
    noise[noise < 0] = 0

    return noise


def plot_brain(fig,
               brain,
               mask=None,
               percentile=99,
               ):
    """ Display the brain that has been generated with a given threshold
    Will display the voxels above the given percentile and then a shadow of
    all voxels in the mask

    Parameters
    ----------

    fig : matplotlib object
        The figure to be displayed, generated from matplotlib. import
        matplotlib.pyplot as plt; fig = plt.figure()

    brain : 3d array
        This is a 3d array with the neural data

    mask : 3d array
        A binary mask describing the location that you want to specify as

    percentile : float
        What percentage of voxels will be included? Based on the values
        supplied

    Returns
    ----------
    ax : matplotlib object
        Object with the information to be plotted

    """

    ax = fig.add_subplot(111, projection='3d')

    # Threshold the data
    threshold = np.percentile(brain.reshape(np.prod(brain.shape[0:3])),
                              percentile)

    # How many voxels exceed a threshold
    brain_threshold = np.where(np.abs(brain) > threshold)

    # Clear the way
    ax.clear()

    ax.set_xlim(0, brain.shape[0])
    ax.set_ylim(0, brain.shape[1])
    ax.set_zlim(0, brain.shape[2])

    # If a mask is provided then plot this
    if mask is not None:
        mask_threshold = np.where(np.abs(mask) > 0)
        ax.scatter(mask_threshold[0],
                   mask_threshold[1],
                   mask_threshold[2],
                   zdir='z',
                   c='black',
                   s=10,
                   alpha=0.01)

    # Plot the volume
    ax.scatter(brain_threshold[0],
               brain_threshold[1],
               brain_threshold[2],
               zdir='z',
               c='red',
               s=20)

    return ax
