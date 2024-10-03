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
onsets and durations from a timing file. This is the time course before
convolution and therefore can be at any temporal precision.

export_3_column:
Generate a three column timing file that can be used with software like FSL
to represent event event onsets and duration

export_epoch_file:
Generate an epoch file from the time course which can be used as an input to
brainiak functions

convolve_hrf
Convolve the signal timecourse with the  HRF to model the expected evoked
activity

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

compute_signal_change
Convert the signal function into useful metric units according to metrics
used by others (Welvaert & Rosseel, 2013)

 Authors:
 Cameron Ellis (Princeton & Yale) 2016-2018
 Chris Baldassano (Princeton) 2016-2017
 Mingbo Cai (Princeton) 2017
"""
import logging

from itertools import product
from statsmodels.tsa.arima.model import ARIMA
import math
import numpy as np
# See pyflakes issue #248
# https://github.com/PyCQA/pyflakes/issues/248
from numpy.linalg import LinAlgError

from scipy import stats
from scipy import signal
import scipy.ndimage as ndimage
import copy
from scipy import optimize

from importlib.resources import files

__all__ = [
    "apply_signal",
    "calc_noise",
    "compute_signal_change",
    "convolve_hrf",
    "export_3_column",
    "export_epoch_file",
    "generate_signal",
    "generate_stimfunction",
    "generate_noise",
    "mask_brain",
    "generate_1d_gaussian_rfs",
    "generate_1d_rf_responses",
]

logger = logging.getLogger(__name__)


def _generate_feature(feature_type,
                      feature_size,
                      signal_magnitude,
                      thickness=1):
    """Generate features corresponding to signal

    Generate a single feature, that can be inserted into the signal volume.
    A feature is a region of activation with a specific shape such as cube
    or ring

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

    signal : 3 dimensional array
        The volume representing the signal

    """

    # If the size is equal to or less than 2 then all features are the same
    if feature_size <= 2:
        feature_type = 'cube'

    # What kind of signal is it?
    if feature_type == 'cube':

        # Preset the size of the signal
        signal = np.ones((feature_size, feature_size, feature_size))

    elif feature_type == 'loop':

        # First make a cube of zeros
        signal = np.zeros((feature_size, feature_size, feature_size))

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

        # Check if the loop is a disk
        if np.all(inner is False):
            logger.warning('Loop feature reduces to a disk because the loop '
                           'is too thick')

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

            # Check if the cavity is a sphere
            if np.all(inner is False):
                logger.warning('Cavity feature reduces to a sphere because '
                               'the cavity is too thick')

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
    for signal_counter in range(feature_quantity):

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
                          temporal_resolution=100.0,
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

    stim_function : 1 by timepoint array, float
        The time course of stimulus evoked activation. This has a temporal
        resolution of temporal resolution / 1.0 elements per second

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

            # Check if the onset is more precise than the temporal resolution
            upsampled_onset = float(onset) * temporal_resolution

            # Because of float precision, the upsampled values might
            # not round as expected .
            # E.g. float('1.001') * 1000 = 1000.99
            if np.allclose(upsampled_onset, np.round(upsampled_onset)) == 0:
                warning = 'Your onset: ' + str(onset) + ' has more decimal ' \
                                                        'points than the ' \
                                                        'specified temporal ' \
                                                        'resolution can ' \
                                                        'resolve. This means' \
                                                        ' that events might' \
                                                        ' be missed. ' \
                                                        'Consider increasing' \
                                                        ' the temporal ' \
                                                        'resolution.'
                logger.warning(warning)

            onsets.append(float(onset))
            event_durations.append(float(duration))
            weights.append(float(weight))

    # If only one duration is supplied then duplicate it for the length of
    # the onset variable
    if len(event_durations) == 1:
        event_durations = event_durations * len(onsets)

    if len(weights) == 1:
        weights = weights * len(onsets)

    # Check files
    if np.max(onsets) > total_time:
        raise ValueError('Onsets outside of range of total time.')

    # Generate the time course as empty, each element is a millisecond by
    # default
    stimfunction = np.zeros((int(round(total_time * temporal_resolution)), 1))

    # Cycle through the onsets
    for onset_counter in list(range(len(onsets))):
        # Adjust for the resolution
        onset_idx = int(np.floor(onsets[onset_counter] * temporal_resolution))

        # Adjust for the resolution
        offset_idx = int(np.floor((onsets[onset_counter] + event_durations[
            onset_counter]) * temporal_resolution))

        # Store the weights
        stimfunction[onset_idx:offset_idx, 0] = [weights[onset_counter]]

    return stimfunction


def export_3_column(stimfunction,
                    filename,
                    temporal_resolution=100.0
                    ):
    """ Output a tab separated three column timing file

    This produces a three column tab separated text file, with the three
    columns representing onset time (s), event duration (s) and weight,
    respectively. Useful if you want to run the simulated data through FEAT
    analyses. In a way, this is the reverse of generate_stimfunction

    Parameters
    ----------

    stimfunction : timepoint by 1 array
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
    while stim_counter < stimfunction.shape[0]:

        # Is it an event?
        if stimfunction[stim_counter, 0] != 0:

            # When did the event start?
            event_onset = str(stim_counter / temporal_resolution)

            # The weight of the stimulus
            weight = str(stimfunction[stim_counter, 0])

            # Reset
            event_duration = 0

            # Is the event still ongoing?
            while stimfunction[stim_counter, 0] != 0 & stim_counter <= \
                    stimfunction.shape[0]:

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


def export_epoch_file(stimfunction,
                      filename,
                      tr_duration,
                      temporal_resolution=100.0
                      ):
    """ Output an epoch file, necessary for some inputs into brainiak

    This takes in the time course of stimulus events and outputs the epoch
    file used in Brainiak. The epoch file is a way to structure the timing
    information in fMRI that allows you to flexibly input different stimulus
    sequences. This is a list with each entry a 3d matrix corresponding to a
    participant. The dimensions of the 3d matrix are condition by epoch by
    time. For the i-th condition, if its k-th epoch spans time points t_m to
    t_n-1, then [i, k, t_m:t_n] are 1 in the epoch file.

    Parameters
    ----------

    stimfunction : list of timepoint by condition arrays
        The stimulus function describing the time course of events. Each
        list entry is from a different participant, each row is a different
        timepoint (with the given temporal precision), each column is a
        different condition. export_epoch_file is looking for differences in
        the value of stimfunction to identify the start and end of an
        epoch. If epochs in stimfunction are coded with the same weight and
        there is no time between blocks then export_epoch_file won't be able to
        label them as different epochs

    filename : str
        The name of the epoch file to be output

    tr_duration : float
        How long is each TR in seconds

    temporal_resolution : float
        How many elements per second are you modeling with the
        stimfunction?

    """

    # Cycle through the participants, different entries in the list
    epoch_file = [0] * len(stimfunction)
    for ppt_counter in range(len(stimfunction)):

        # What is the time course for the participant (binarized)
        stimfunction_ppt = np.abs(stimfunction[ppt_counter]) > 0

        # Down sample the stim function
        stride = tr_duration * temporal_resolution
        stimfunction_downsampled = stimfunction_ppt[::int(stride), :]

        # Calculates the number of event onsets. This uses changes in value
        # to reflect different epochs. This might be false in some cases (the
        # weight is non-uniform over an epoch or there is no break between
        # identically weighted epochs).
        epochs = 0  # Preset
        conditions = stimfunction_ppt.shape[1]
        for condition_counter in range(conditions):

            weight_change = (np.diff(stimfunction_downsampled[:,
                                     condition_counter], 1, 0) != 0)

            # If the first or last events are 'on' then make these
            # represent a epoch change
            if stimfunction_downsampled[0, condition_counter] == 1:
                weight_change[0] = True
            if stimfunction_downsampled[-1, condition_counter] == 1:
                weight_change[-1] = True

            epochs += int(np.max(np.sum(weight_change, 0)) / 2)

        # Get other information
        trs = stimfunction_downsampled.shape[0]

        # Make a timing file for this participant
        epoch_file[ppt_counter] = np.zeros((conditions, epochs, trs))

        # Cycle through conditions
        epoch_counter = 0  # Reset and count across conditions
        tr_counter = 0
        while tr_counter < stimfunction_downsampled.shape[0]:

            for condition_counter in range(conditions):

                # Is it an event?
                if tr_counter < stimfunction_downsampled.shape[0] and \
                                stimfunction_downsampled[
                                    tr_counter, condition_counter] == 1:
                    # Add a one for this TR
                    epoch_file[ppt_counter][condition_counter,
                                            epoch_counter, tr_counter] = 1

                    # Find the next non event value
                    end_idx = np.where(stimfunction_downsampled[tr_counter:,
                                       condition_counter] == 0)[
                        0][0]
                    tr_idxs = list(range(tr_counter, tr_counter + end_idx))

                    # Add ones to all the trs within this event time frame
                    epoch_file[ppt_counter][condition_counter,
                                            epoch_counter, tr_idxs] = 1

                    # Start from this index
                    tr_counter += end_idx

                    # Increment
                    epoch_counter += 1

                # Increment the counter
                tr_counter += 1

        # Convert to boolean
        epoch_file[ppt_counter] = epoch_file[ppt_counter].astype('bool')

    # Save the file
    np.save(filename, epoch_file)


def _double_gamma_hrf(response_delay=6,
                      undershoot_delay=12,
                      response_dispersion=0.9,
                      undershoot_dispersion=0.9,
                      response_scale=1,
                      undershoot_scale=0.035,
                      temporal_resolution=100.0,
                      ):
    """Create the double gamma HRF with the timecourse evoked activity.
    Default values are based on Glover, 1999 and Walvaert, Durnez,
    Moerkerke, Verdoolaege and Rosseel, 2011

    Parameters
    ----------

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

    hrf : multi dimensional array
        A double gamma HRF to be used for convolution.

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

    return hrf


def convolve_hrf(stimfunction,
                 tr_duration,
                 hrf_type='double_gamma',
                 scale_function=True,
                 temporal_resolution=100.0,
                 ):
    """ Convolve the specified hrf with the timecourse.
    The output of this is a downsampled convolution of the stimfunction and
    the HRF function. If temporal_resolution is 1 / tr_duration then the
    output will be the same length as stimfunction. This time course assumes
    that slice time correction has occurred and all slices have been aligned
    to the middle time point in the TR.

    Be aware that if scaling is on and event durations are less than the
    duration of a TR then the hrf may or may not come out as anticipated.
    This is because very short events would evoke a small absolute response
    after convolution  but if there are only short events and you scale then
    this will look similar to a convolution with longer events. In general
    scaling is useful, which is why it is the default, but be aware of this
    edge case and if it is a concern, set the scale_function to false.

    Parameters
    ----------

    stimfunction : timepoint by feature array
        What is the time course of events to be modelled in this
        experiment. This can specify one or more timecourses of events.
        The events can be weighted or binary

    tr_duration : float
        How long (in s) between each volume onset

    hrf_type : str or list
        Takes in a string describing the hrf that ought to be created.
        Can instead take in a vector describing the HRF as it was
        specified by any function. The default is 'double_gamma' in which
        an initial rise and an undershoot are modelled.

    scale_function : bool
        Do you want to scale the function to a range of 1

    temporal_resolution : float
        How many elements per second are you modeling for the stimfunction
    Returns
    ----------

    signal_function : timepoint by timecourse array
        The time course of the HRF convolved with the stimulus function.
        This can have multiple time courses specified as different
        columns in this array.

    """

    # Check if it is timepoint by feature
    if stimfunction.shape[0] < stimfunction.shape[1]:
        logger.warning('Stimfunction may be the wrong shape')

    if np.any(np.sum(abs(stimfunction), 0) == 0):
        logger.warning('stimfunction contains voxels of all zeros, will nan')

    # How will stimfunction be resized
    stride = int(temporal_resolution * tr_duration)
    duration = int(stimfunction.shape[0] / stride)

    # Generate the hrf to use in the convolution
    if hrf_type == 'double_gamma':
        hrf = _double_gamma_hrf(temporal_resolution=temporal_resolution)
    elif isinstance(hrf_type, list):
        hrf = hrf_type

    # How many timecourses are there
    list_num = stimfunction.shape[1]

    # Create signal functions for each list in the stimfunction
    for list_counter in range(list_num):

        # Perform the convolution
        signal_temp = np.convolve(stimfunction[:, list_counter], hrf)

        # Down sample the signal function so that it only has one element per
        # TR. This assumes that all slices are collected at the same time,
        # which is often the result of slice time correction. In other
        # words, the output assumes slice time correction
        signal_temp = signal_temp[:duration * stride]
        signal_vox = signal_temp[int(stride / 2)::stride]

        # Scale the function so that the peak response is 1
        if scale_function:
            signal_vox = signal_vox / np.max(signal_vox)

        # Add this function to the stack
        if list_counter == 0:
            signal_function = np.zeros((len(signal_vox), list_num))

        signal_function[:, list_counter] = signal_vox

    return signal_function


def apply_signal(signal_function,
                 volume_signal,
                 ):
    """Combine the signal volume with its timecourse

    Apply the convolution of the HRF and stimulus time course to the
    volume.

    Parameters
    ----------

    signal_function : timepoint by timecourse array, float
        The timecourse of the signal over time. If there is only one column
        then the same timecourse is applied to all non-zero voxels in
        volume_signal. If there is more than one column then each column is
        paired with a non-zero voxel in the volume_signal (a 3d numpy array
        generated in generate_signal).

    volume_signal : multi dimensional array, float
        The volume containing the signal to be convolved with the same
        dimensions as the output volume. The elements in volume_signal
        indicate how strong each signal in signal_function are modulated by
        in the output volume


    Returns
    ----------
    signal : multidimensional array, float
        The convolved signal volume with the same 3d as volume signal and
        the same 4th dimension as signal_function

    """

    # How many timecourses are there within the signal_function
    timepoints = signal_function.shape[0]
    timecourses = signal_function.shape[1]

    # Preset volume
    signal = np.zeros([volume_signal.shape[0], volume_signal.shape[
        1], volume_signal.shape[2], timepoints])

    # Find all the non-zero voxels in the brain
    idxs = np.where(volume_signal != 0)
    if timecourses == 1:
        # If there is only one time course supplied then duplicate it for
        # every voxel
        signal_function = np.tile(signal_function, (1, len(idxs[0])))

    elif len(idxs[0]) != timecourses:
        raise IndexError('The number of non-zero voxels in the volume and '
                         'the number of timecourses does not match. Aborting')

    # For each coordinate with a non zero voxel, fill in the timecourse for
    # that voxel
    for idx_counter in range(len(idxs[0])):
        x = idxs[0][idx_counter]
        y = idxs[1][idx_counter]
        z = idxs[2][idx_counter]

        # Pull out the function for this voxel
        signal_function_temp = signal_function[:, idx_counter]

        # Multiply the voxel value by the function timecourse
        signal[x, y, z, :] = volume_signal[x, y, z] * signal_function_temp

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

    fwhm : float, list
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


def _calc_sfnr(volume,
               mask,
               ):
    """ Calculate the the SFNR of a volume
    Calculates the Signal to Fluctuation Noise Ratio, the mean divided
    by the detrended standard deviation of each brain voxel. Based on
    Friedman and Glover, 2006

    Parameters
    ----------

    volume : 4d array, float
        Take a volume time series

    mask : 3d array, binary
        A binary mask the same size as the volume

    Returns
    -------

    snr : float
        The SFNR of the volume

    """

    # Make a matrix of brain voxels by time
    brain_voxels = volume[mask > 0]

    # Take the means of each voxel over time
    mean_voxels = np.nanmean(brain_voxels, 1)

    # Detrend (second order polynomial) the voxels over time and then
    # calculate the standard deviation.
    order = 2
    seq = np.linspace(1, brain_voxels.shape[1], brain_voxels.shape[1])
    detrend_poly = np.polyfit(seq, brain_voxels.transpose(), order)

    # Detrend for each voxel
    detrend_voxels = np.zeros(brain_voxels.shape)
    for voxel in range(brain_voxels.shape[0]):
        trend = detrend_poly[0, voxel] * seq ** 2 + detrend_poly[1, voxel] * \
                                                   seq + detrend_poly[2, voxel]
        detrend_voxels[voxel, :] = brain_voxels[voxel, :] - trend

    std_voxels = np.nanstd(detrend_voxels, 1)

    # Calculate the sfnr of all voxels across the brain
    sfnr_voxels = mean_voxels / std_voxels

    # Return the average sfnr
    return np.mean(sfnr_voxels)


def _calc_snr(volume,
              mask,
              dilation=5,
              reference_tr=None,
              ):
    """ Calculate the the SNR of a volume
    Calculates the Signal to  Noise Ratio, the mean of brain voxels
    divided by the standard deviation across non-brain voxels. Specify a TR
    value to calculate the mean and standard deviation for that TR. To
    calculate the standard deviation of non-brain voxels we can subtract
    any baseline structure away first, hence getting at deviations due to the
    system noise and not something like high baseline values in non-brain
    parts of the body.

    Parameters
    ----------

    volume : 4d array, float
        Take a volume time series

    mask : 3d array, binary
        A binary mask the same size as the volume

    dilation : int
        How many binary dilations do you want to perform on the mask to
        determine the non-brain voxels. If you increase this the SNR
        increases and the non-brain voxels (after baseline subtraction) more
        closely resemble a gaussian

    reference_tr : int or list
        Specifies the TR to calculate the SNR for. If multiple are supplied
        then it will use the average of them.

    Returns
    -------

    snr : float
        The SNR of the volume

    """

    # If no TR is specified then take all of them
    if reference_tr is None:
        reference_tr = list(range(volume.shape[3]))

    # Dilate the mask in order to ensure that non-brain voxels are far from
    # the brain
    if dilation > 0:
        mask_dilated = ndimage.binary_dilation(mask, iterations=dilation)
    else:
        mask_dilated = mask

    # Make a matrix of brain and non_brain voxels, selecting the timepoint/s
    brain_voxels = volume[mask > 0][:, reference_tr]
    nonbrain_voxels = (volume[:, :, :, reference_tr]).astype('float64')

    # If you have multiple TRs
    if len(brain_voxels.shape) > 1:
        brain_voxels = np.mean(brain_voxels, 1)
        nonbrain_voxels = np.mean(nonbrain_voxels, 3)

    nonbrain_voxels = nonbrain_voxels[mask_dilated == 0]

    # Take the means of each voxel over time
    mean_voxels = np.nanmean(brain_voxels)

    # Find the standard deviation of the voxels
    std_voxels = np.nanstd(nonbrain_voxels)

    # Return the snr
    return mean_voxels / std_voxels


def _calc_ARMA_noise(volume,
                     mask,
                     auto_reg_order=1,
                     ma_order=1,
                     sample_num=100,
                     ):
    """ Calculate the the ARMA noise of a volume
    This calculates the autoregressive and moving average noise of the volume
    over time by sampling brain voxels and averaging them.

    Parameters
    ----------

    volume : 4d array or 1d array, float
        Take a volume time series to extract the middle slice from the
        middle TR. Can also accept a one dimensional time course (mask input
        is then ignored).

    mask : 3d array, binary
        A binary mask the same size as the volume

    auto_reg_order : int
        What order of the autoregression do you want to estimate

    sample_num : int
        How many voxels would you like to sample to calculate the AR values.
        The AR distribution of real data is approximately exponential maxing
        at 1. From analyses across a number of participants, to get less
        than 3% standard deviation of error from the true mean it is
        necessary to sample at least 100 voxels.

    Returns
    -------

    auto_reg_rho : list of floats
        Rho of a specific order for the autoregression noise in the data

    na_rho : list of floats
        Moving average of a specific order for the data

    """

    # Pull out the non masked voxels
    if len(volume.shape) > 1:
        brain_timecourse = volume[mask > 0]
    else:
        # If a 1 dimensional input is supplied then reshape it to make the
        # timecourse
        brain_timecourse = volume.reshape(1, len(volume))

    # Identify some brain voxels to assess
    voxel_idxs = list(range(brain_timecourse.shape[0]))
    np.random.shuffle(voxel_idxs)

    # If there are more samples than voxels, take all of the voxels
    if len(voxel_idxs) < sample_num:
        sample_num = len(voxel_idxs)

    auto_reg_rho_all = np.zeros((sample_num, auto_reg_order))
    ma_all = np.zeros((sample_num, ma_order))
    for voxel_counter in range(sample_num):

        # Get the timecourse and demean it
        timecourse = brain_timecourse[voxel_idxs[voxel_counter], :]
        demeaned_timecourse = timecourse - timecourse.mean()

        # Pull out the ARMA values (depends on order)
        try:
            model = ARIMA(demeaned_timecourse,
                          order=[auto_reg_order, 0, ma_order])
            model_fit = model.fit()
            params = model_fit.params
        except (ValueError, LinAlgError):
            params = np.ones(auto_reg_order + ma_order + 1) * np.nan

        auto_reg_rho_all[voxel_counter, :] = params[1:auto_reg_order + 1]
        ma_all[voxel_counter, :] = params[auto_reg_order + 1:-1]

    # Average all of the values and then convert them to a list
    auto_reg_rho = np.nanmean(auto_reg_rho_all, 0).tolist()
    ma_rho = np.nanmean(ma_all, 0).tolist()

    # Return the coefficients
    return auto_reg_rho, ma_rho


def calc_noise(volume,
               mask,
               template,
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
    template : 3d array, float
        A continuous (0 -> 1) volume describing the likelihood a voxel is in
        the brain. This can be used to contrast the brain and non brain.

    noise_dict : dict
        The initialized dictionary of the calculated noise parameters of the
        provided dataset (usually it is only the voxel size)
    Returns
    -------

    noise_dict : dict
        Return a dictionary of the calculated noise parameters of the provided
        dataset

    """

    # Check the inputs
    if template.max() > 1.1:
        raise ValueError('Template out of range')

    # Create the mask if not supplied and set the mask size
    if mask is None:
        raise ValueError('Mask not supplied')

    # Update noise dict if it is not yet created
    if noise_dict is None:
        noise_dict = {'voxel_size': [1.0, 1.0, 1.0]}
    elif 'voxel_size' not in noise_dict:
        noise_dict['voxel_size'] = [1.0, 1.0, 1.0]

    # What is the max activation of the mean of this voxel (allows you to
    # convert between the mask and the mean of the brain volume)
    noise_dict['max_activity'] = np.nanmax(np.mean(volume, 3))

    # Calculate the temporal variability of the volume
    noise_dict['auto_reg_rho'], noise_dict['ma_rho'] = _calc_ARMA_noise(
        volume, mask)

    # Set it such that all of the temporal variability will be accounted for
    #  by the AR component
    noise_dict['auto_reg_sigma'] = 1

    # Preset these values to be zero, as in you are not attempting to
    # simulate them
    noise_dict['physiological_sigma'] = 0
    noise_dict['task_sigma'] = 0
    noise_dict['drift_sigma'] = 0

    # Calculate the sfnr
    noise_dict['sfnr'] = _calc_sfnr(volume,
                                    mask,
                                    )

    # Calculate the fwhm on a subset of volumes
    if volume.shape[3] > 100:
        # Take only 100 shuffled TRs
        trs = np.random.choice(volume.shape[3], size=100, replace=False)
    else:
        trs = list(range(0, volume.shape[3]))

    # Go through the trs and pull out the fwhm
    fwhm = [0] * len(trs)
    for tr in range(len(trs)):
        fwhm[tr] = _calc_fwhm(volume[:, :, :, trs[tr]],
                              mask,
                              noise_dict['voxel_size'],
                              )

    # Keep only the mean
    noise_dict['fwhm'] = np.mean(fwhm)
    noise_dict['snr'] = _calc_snr(volume,
                                  mask,
                                  )

    # Return the noise dictionary
    return noise_dict


def _generate_noise_system(dimensions_tr,
                           spatial_sd,
                           temporal_sd,
                           spatial_noise_type='gaussian',
                           temporal_noise_type='gaussian',
                           ):
    """Generate the scanner noise

    Generate system noise, either rician, gaussian or exponential, for the
    scanner. Generates a distribution with a SD of 1. If you look at the
    distribution of non-brain voxel intensity in modern scans you will see
    it is rician. However, depending on how you have calculated the SNR and
    whether the template is being used you will want to use this function
    differently: the voxels outside the brain tend to be stable over time and
    usually reflect structure in the MR signal (e.g. the
    baseline MR of the head coil or skull). Hence the template captures this
    rician noise structure. If you are adding the machine noise to the
    template, as is done in generate_noise, then you are likely doubling up
    on the addition of machine noise. In such cases, machine noise seems to
    be better modelled by gaussian noise on top of this rician structure.

    Parameters
    ----------

    dimensions_tr : n length array, int
        What are the dimensions of the volume you wish to insert
        noise into. This can be a volume of any size

    spatial_sd : float
        What is the standard deviation in space of the noise volume to be
        generated

    temporal_sd : float
        What is the standard deviation in time of the noise volume to be
        generated

    noise_type : str
        String specifying the noise type. If you aren't specifying the noise
        template then Rician is the appropriate model of noise. However,
        if you are subtracting the template, as is default, then you should
        use gaussian. (If the dilation parameter of _calc_snr is <10 then
        gaussian is only an approximation)

    Returns
    ----------

    system_noise : multidimensional array, float
        Create a volume with system noise

    """
    def noise_volume(dimensions,
                     noise_type,
                     ):

        if noise_type == 'rician':
            # Generate the Rician noise (has an SD of 1)
            noise = stats.rice.rvs(b=0, loc=0, scale=1.527, size=dimensions)
        elif noise_type == 'exponential':
            # Make an exponential distribution (has an SD of 1)
            noise = stats.expon.rvs(0, scale=1, size=dimensions)
        elif noise_type == 'gaussian':
            noise = np.random.randn(np.prod(dimensions)).reshape(dimensions)

        # Return the noise
        return noise

    # Get just the xyz coordinates
    dimensions = np.asarray([dimensions_tr[0],
                             dimensions_tr[1],
                             dimensions_tr[2],
                             1])

    # Generate noise
    spatial_noise = noise_volume(dimensions, spatial_noise_type)
    temporal_noise = noise_volume(dimensions_tr, temporal_noise_type)

    # Make the system noise have a specific spatial variability
    spatial_noise *= spatial_sd

    # Set the size of the noise
    temporal_noise *= temporal_sd

    # The mean in time of system noise needs to be zero, so subtract the
    # means of the temporal noise in time
    temporal_noise_mean = np.mean(temporal_noise, 3).reshape(dimensions[0],
                                                             dimensions[1],
                                                             dimensions[2],
                                                             1)
    temporal_noise = temporal_noise - temporal_noise_mean

    # Save the combination
    system_noise = spatial_noise + temporal_noise

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

    noise_task : one dimensional array, float
        Generates the temporal task noise timecourse

    """

    # Make the noise to be added
    stimfunction_tr = stimfunction_tr != 0
    if motion_noise == 'gaussian':
        noise = stimfunction_tr * np.random.normal(0, 1,
                                                   size=stimfunction_tr.shape)
    elif motion_noise == 'rician':
        noise = stimfunction_tr * stats.rice.rvs(0, 1,
                                                 size=stimfunction_tr.shape)

    noise_task = stimfunction_tr + noise

    # Normalize
    noise_task = stats.zscore(noise_task).flatten()

    return noise_task


def _generate_noise_temporal_drift(trs,
                                   tr_duration,
                                   basis="cos_power_drop",
                                   period=150,
                                   ):

    """Generate the drift noise

    Create a trend (either sine or discrete_cos), of a given period and random
    phase, to represent the drift of the signal over time

    Parameters
    ----------

    trs : int
        How many volumes (aka TRs) are there

    tr_duration : float
        How long in seconds is each volume acqusition

    basis : str
        What is the basis function for the drift. Could be made of discrete
        cosines (for longer run durations, more basis functions are
        created) that either have equal power ('discrete_cos') or the power
        diminishes such that 99% of the power is below a specified frequency
        ('cos_power_drop'). Alternatively, this drift could simply be a sine
        wave ('sine')

    period : int
        When the basis function is 'cos_power_drop' this is the period over
        which no power of the drift exceeds (i.e. the power of the drift
        asymptotes at this period). However for the other basis functions,
        this is simply how many seconds is the period of oscillation of the
        drift

    Returns
    ----------
    noise_drift : one dimensional array, float
        The drift timecourse of activity

    """

    # Calculate drift differently depending on the basis function
    if basis == 'discrete_cos':

        # Specify each tr in terms of its phase with the given period
        timepoints = np.linspace(0, trs - 1, trs)
        timepoints = ((timepoints * tr_duration) / period) * 2 * np.pi

        # Specify the other timing information
        duration = trs * tr_duration
        basis_funcs = int(np.floor(duration / period))  # How bases do you have

        if basis_funcs == 0:
            err_msg = 'Too few timepoints (' + str(trs) + ') to accurately ' \
                                                          'model drift'
            logger.warning(err_msg)
            basis_funcs = 1

        noise_drift = np.zeros((timepoints.shape[0], basis_funcs))
        for basis_counter in list(range(1, basis_funcs + 1)):

            # What steps do you want to take for this basis function
            timepoints_basis = (timepoints/basis_counter) + (np.random.rand()
                                                             * np.pi * 2)

            # Store the drift from this basis func
            noise_drift[:, basis_counter - 1] = np.cos(timepoints_basis)

        # Average the drift
        noise_drift = np.mean(noise_drift, 1)

    elif basis == 'sine':

        # Calculate the cycles of the drift for a given function.
        cycles = trs * tr_duration / period

        # Create a sine wave with a given number of cycles and random phase
        timepoints = np.linspace(0, trs - 1, trs)
        phaseshift = np.pi * 2 * np.random.random()
        phase = (timepoints / (trs - 1) * cycles * 2 * np.pi) + phaseshift
        noise_drift = np.sin(phase)

    elif basis == 'cos_power_drop':

        # Make a vector counting each TR
        timepoints = np.linspace(0, trs - 1, trs) * tr_duration

        # Specify the other timing information
        duration = trs * tr_duration

        # How bases do you have? This is to adhere to Nyquist
        basis_funcs = int(trs)

        noise_drift = np.zeros((timepoints.shape[0], basis_funcs))
        for basis_counter in list(range(1, basis_funcs + 1)):
            # What steps do you want to take for this basis function
            random_phase = np.random.rand() * np.pi * 2

            timepoint_phase = (timepoints / duration * np.pi * basis_counter)

            # In radians, what is the value for each time point
            timepoints_basis = timepoint_phase + random_phase

            # Store the drift from this basis func
            noise_drift[:, basis_counter - 1] = np.cos(timepoints_basis)

        def power_drop(r, L, F, tr_duration):
            # Function to return the drop rate for the power of basis functions
            # In other words, how much should the weight of each basis function
            # reduce in order to make the power you retain of the period's
            # frequency be 99% of the total power of the highest frequency, as
            # defined by the DCT.
            # For an example where there are 20 time points, there will be 20
            # basis functions in the DCT. If the period of the signal you wish
            # to simulate is such that 99% of the power should drop off after
            # the equivalent of 5 of these basis functions, then the way this
            # code works is it finds the rate at which power must drop off for
            # all of the 20 basis functions such that by the 5th one, there is
            # only 1% of the power remaining.
            # r is the power reduction rate which should be between 0 and 1
            # L is the duration of the run in seconds
            # F is period of the cycle in seconds It is assumed that this will
            # be greater than the tr_duration, or else this will not work
            # tr_duration is the duration of each TR in seconds

            # Check the TR duration
            if F < tr_duration:
                msg = 'Period %0.0f > TR duration %0.0f' % ((F, tr_duration))
                raise ValueError(msg)

            percent_retained = 0.99  # What is the percentage of power retained

            # Compare the power at the period frequency (in the numerator) with
            # the power at the frequency of the DCT, AKA the highest possible
            # frequency in the data (in the denominator)
            numerator = 1 - r ** (2 * L / F)  # Power of this period
            denominator = 1 - r ** (2 * L / tr_duration)  # Power of DCT freq.

            # Calculate the retained power
            power_drop = abs((numerator / denominator) - percent_retained)
            return power_drop

        # Solve for power reduction rate.
        # This assumes that r is between 0 and 1
        # Takes the duration and period as arguments
        sol = optimize.minimize_scalar(power_drop,
                                       bounds=(0, 1),
                                       method='Bounded',
                                       args=(duration, period, tr_duration))

        # Pull out the solution
        r = sol.x

        # Weight the basis functions based on the power drop off
        basis_weights = r ** np.arange(basis_funcs)

        # Weigh the basis functions
        weighted_basis_funcs = np.multiply(noise_drift, basis_weights)

        # Average the drift
        noise_drift = np.mean(weighted_basis_funcs, 1)

    # Normalize so the sigma is 1
    noise_drift = stats.zscore(noise_drift)

    # Return noise
    return noise_drift


def _generate_noise_temporal_autoregression(timepoints,
                                            noise_dict,
                                            dimensions,
                                            mask,
                                            ):

    """Generate the autoregression noise
    Make a slowly drifting timecourse with the given autoregression
    parameters. This can take in both AR and MA components

    Parameters
    ----------

    timepoints : 1 Dimensional array
        What time points are sampled by a TR

    noise_dict : dict
        A dictionary specifying the types of noise in this experiment. The
        noise types interact in important ways. First, autoregressive,
        physiological and task-based noise types are mixed together in
        _generate_temporal_noise. The parameter values for 'auto_reg_sigma',
        'physiological_sigma' and 'task_sigma' describe the proportion of
        mixing of these elements, respectively. However critically, 'SFNR' is
        the parameter that controls how much noise these components contribute
        to the brain. 'auto_reg_rho' and 'ma_rho' set parameters for the
        autoregressive noise being simulated. Second, drift noise is added to
        this, according to the size of 'drift_sigma'. Thirdly, system noise is
        added based on the 'SNR' parameter. Finally, 'fwhm' is used to estimate
        the smoothness of the noise being inserted. If 'matched' is set to
        true, then it will fit the parameters to match the participant as best
        as possible.

        Variables defined as follows:

        snr [float]: Ratio of MR signal to the spatial noise
        sfnr [float]: Ratio of the MR signal to the temporal noise. This is the
        total variability that the following sigmas 'sum' to:

        task_sigma [float]: Size of the variance of task specific noise
        auto_reg_sigma [float]: Size of the variance of autoregressive
        noise. This is an ARMA process where the AR and MA components can be
        separately specified
        physiological_sigma [float]: Size of the variance of physiological
        noise

        drift_sigma [float]: Size of the variance of drift noise

        auto_reg_rho [list]: The coefficients of the autoregressive
        components you are modeling
        ma_rho [list]:The coefficients of the moving average components you
        are modeling
        max_activity [float]: The max value of the averaged brain in order
        to reference the template
        voxel_size [list]: The mm size of the voxels
        fwhm [float]: The gaussian smoothing kernel size (mm)
        matched [bool]: Specify whether you are fitting the noise parameters

        The volumes of brain noise that are generated have smoothness
        specified by 'fwhm'

    dimensions : 3 length array, int
        What is the shape of the volume to be generated

    mask : 3 dimensional array, binary
        The masked brain, thresholded to distinguish brain and non-brain

    Returns
    ----------
    noise_autoregression : one dimensional array, float
        Generates the autoregression noise timecourse

    """

    # Pull out the relevant noise parameters
    auto_reg_rho = noise_dict['auto_reg_rho']
    ma_rho = noise_dict['ma_rho']

    # Specify the order based on the number of rho supplied
    auto_reg_order = len(auto_reg_rho)
    ma_order = len(ma_rho)

    # This code assumes that the AR order is higher than the MA order
    if ma_order > auto_reg_order:
        msg = 'MA order (%d) is greater than AR order (%d). Cannot run.' % (
            ma_order, auto_reg_order)
        raise ValueError(msg)

    # Generate a random variable at each time point that is a decayed value
    # of the previous time points
    noise_autoregression = np.zeros((dimensions[0], dimensions[1],
                                     dimensions[2], len(timepoints)))
    err_vols = np.zeros((dimensions[0], dimensions[1], dimensions[2],
                         len(timepoints)))
    for tr_counter in range(len(timepoints)):

        # Create a brain shaped volume with appropriate smoothing properties
        noise = _generate_noise_spatial(dimensions=dimensions,
                                        mask=mask,
                                        fwhm=noise_dict['fwhm'],
                                        )

        # Store all of the noise volumes
        err_vols[:, :, :, tr_counter] = noise

        if tr_counter == 0:
            noise_autoregression[:, :, :, tr_counter] = noise

        else:

            # Preset the volume to collect the AR estimated process
            AR_vol = np.zeros((dimensions[0], dimensions[1], dimensions[2]))

            # Iterate through both the AR and MA values
            for pCounter in list(range(1, auto_reg_order + 1)):
                past_TR = int(tr_counter - pCounter)

                if tr_counter - pCounter >= 0:

                    # Pull out a previous TR
                    past_vols = noise_autoregression[:, :, :, past_TR]

                    # Add the discounted previous volume
                    AR_vol += past_vols * auto_reg_rho[pCounter - 1]

                    # If the MA order has at least this many coefficients
                    # then consider the error terms
                    if ma_order >= pCounter:

                        # Pull out a previous TR
                        past_noise = err_vols[:, :, :, past_TR]

                        # Add the discounted previous noise
                        AR_vol += past_noise * ma_rho[pCounter - 1]

            noise_autoregression[:, :, :, tr_counter] = AR_vol + noise

    # Z score the data so that all of the standard deviations of the voxels
    # are one (but the ARMA coefs are unchanged)
    noise_autoregression = stats.zscore(noise_autoregression, 3)

    return noise_autoregression


def _generate_noise_temporal_phys(timepoints,
                                  resp_freq=0.2,
                                  heart_freq=1.17,
                                  ):
    """Generate the physiological noise.
    Create noise representing the heart rate and respiration of the data.
    Default values based on Walvaert, Durnez, Moerkerke, Verdoolaege and
    Rosseel, 2011

    Parameters
    ----------

    timepoints : 1 Dimensional array
        What time points, in seconds, are sampled by a TR

    resp_freq : float
        What is the frequency of respiration (in Hz)

    heart_freq : float
        What is the frequency of heart beat (in Hz)

    Returns
    ----------
    noise_phys : one dimensional array, float
        Generates the physiological temporal noise timecourse

    """

    resp_phase = (np.random.rand(1) * 2 * np.pi)[0]
    heart_phase = (np.random.rand(1) * 2 * np.pi)[0]

    # Find the rate for each timepoint
    resp_rate = (resp_freq * 2 * np.pi)
    heart_rate = (heart_freq * 2 * np.pi)

    # Calculate the radians for each variable at this
    # given TR
    resp_radians = np.multiply(timepoints, resp_rate) + resp_phase
    heart_radians = np.multiply(timepoints, heart_rate) + heart_phase

    # Combine the two types of noise and append
    noise_phys = np.cos(resp_radians) + np.sin(heart_radians)

    # Normalize
    noise_phys = stats.zscore(noise_phys)

    return noise_phys


def _generate_noise_spatial(dimensions,
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
        What is the shape of the volume to be generated. This code
        compresesses the range if the x and y dimensions are not equivalent.
        This fixes this by upsampling and then downsampling the volume.

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

    noise_spatial : 3d array, float
        Generates the spatial noise volume for these parameters

    """

    # Check the input is correct
    if len(dimensions) == 4:
        logger.warning('4 dimensions have been supplied, only using 3')
        dimensions = dimensions[0:3]

    # If the dimensions are wrong then upsample now
    if dimensions[0] != dimensions[1] or dimensions[1] != dimensions[2]:
        max_dim = np.max(dimensions)
        new_dim = (max_dim, max_dim, max_dim)
    else:
        new_dim = dimensions

    def _logfunc(x, a, b, c):
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

    def _fftIndgen(n):
        """# Specify the fft coefficents

        Parameters
        ----------

        n : int
            Dim size to estimate over

        Returns
        ----------

        array of ints
            fft indexes
        """

        # Pull out the ascending and descending indexes
        ascending = np.linspace(0, int(n / 2), int(n / 2 + 1))
        elements = int(np.ceil(n / 2 - 1))  # Round up so that len(output)==n
        descending = np.linspace(-elements, -1, elements)

        return np.concatenate((ascending, descending))

    def _Pk2(idxs, sigma):
        """# Specify the amplitude given the fft coefficents

        Parameters
        ----------

        idxs : 3 by voxel array int
            fft indexes

        sigma : float
            spatial sigma

        Returns
        ----------

        amplitude : 3 by voxel array
            amplitude of the fft coefficients
        """

        # The first set of idxs ought to be zero so make the first value
        # zero to avoid a divide by zero error
        amp_start = np.array((0))

        # Compute the amplitude of the function for a series of indices
        amp_end = np.sqrt(np.sqrt(np.sum(idxs[:, 1:] ** 2, 0)) ** (-1 * sigma))
        amplitude = np.append(amp_start, amp_end)

        # Return the output
        return amplitude

    # Convert from fwhm to sigma (relationship discovered empirical, only an
    #  approximation up to sigma = 0 -> 5 which corresponds to fwhm = 0 -> 8,
    # relies on an assumption of brain size).
    spatial_sigma = _logfunc(fwhm, -0.36778719, 2.10601011, 2.15439247)

    noise = np.fft.fftn(np.random.normal(size=new_dim))

    # Create a meshgrid of the object
    fft_vol = np.meshgrid(_fftIndgen(new_dim[0]), _fftIndgen(new_dim[1]),
                          _fftIndgen(new_dim[2]))

    # Reshape the data into a vector
    fft_vec = np.asarray((fft_vol[0].flatten(), fft_vol[1].flatten(), fft_vol[
        2].flatten()))

    # Compute the amplitude for each element in the grid
    amp_vec = _Pk2(fft_vec, spatial_sigma)

    # Reshape to be a brain volume
    amplitude = amp_vec.reshape(new_dim)

    # The output
    noise_fft = (np.fft.ifftn(noise * amplitude)).real

    # Fix the dimensionality of the data (if necessary)
    noise_spatial = noise_fft[:dimensions[0], :dimensions[1], :dimensions[2]]

    # Mask or not, then z score
    if mask is not None:

        # Mask the output
        noise_spatial *= mask

        # Z score the specific to the brain
        noise_spatial[mask > 0] = stats.zscore(noise_spatial[mask > 0])
    else:
        # Take the grand mean/std and use for z scoring
        grand_mean = (noise_spatial).mean()
        grand_std = (noise_spatial).std()

        noise_spatial = (noise_spatial - grand_mean) / grand_std

    return noise_spatial


def _generate_noise_temporal(stimfunction_tr,
                             tr_duration,
                             dimensions,
                             template,
                             mask,
                             noise_dict
                             ):
    """Generate the temporal noise
    Generate the time course of the average brain voxel. To change the
    relative mixing of the noise components, change the sigma's specified
    below.

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

    noise_dict : dict
        A dictionary specifying the types of noise in this experiment. The
        noise types interact in important ways. First, autoregressive,
        physiological and task-based noise types are mixed together in
        _generate_temporal_noise. The parameter values for 'auto_reg_sigma',
        'physiological_sigma' and 'task_sigma' describe the proportion of
        mixing of these elements, respectively. However critically, 'SFNR' is
        the parameter that controls how much noise these components contribute
        to the brain. 'auto_reg_rho' and 'ma_rho' set parameters for the
        autoregressive noise being simulated. Second, drift noise is added to
        this, according to the size of 'drift_sigma'. Thirdly, system noise is
        added based on the 'SNR' parameter. Finally, 'fwhm' is used to estimate
        the smoothness of the noise being inserted. If 'matched' is set to
        true, then it will fit the parameters to match the participant as best
        as possible.

        Variables defined as follows:

        snr [float]: Ratio of MR signal to the spatial noise
        sfnr [float]: Ratio of the MR signal to the temporal noise. This is the
        total variability that the following sigmas 'sum' to:

        task_sigma [float]: Size of the variance of task specific noise
        auto_reg_sigma [float]: Size of the variance of autoregressive
        noise. This is an ARMA process where the AR and MA components can be
        separately specified
        physiological_sigma [float]: Size of the variance of physiological
        noise

        drift_sigma [float]: Size of the variance of drift noise

        auto_reg_rho [list]: The coefficients of the autoregressive
        components you are modeling
        ma_rho [list]:The coefficients of the moving average components you
        are modeling
        max_activity [float]: The max value of the averaged brain in order
        to reference the template
        voxel_size [list]: The mm size of the voxels
        fwhm [float]: The gaussian smoothing kernel size (mm)
        matched [bool]: Specify whether you are fitting the noise parameters

        The volumes of brain noise that are generated have smoothness
        specified by 'fwhm'

    Returns
    ----------

    noise_temporal : one dimensional array, float
        Generates the temporal noise timecourse for these parameters

    """

    # Set up common parameters
    # How many TRs are there
    trs = len(stimfunction_tr)

    # What time points are sampled by a TR?
    timepoints = list(np.linspace(0, (trs - 1) * tr_duration, trs))

    # Preset the volume
    noise_volume = np.zeros((dimensions[0], dimensions[1], dimensions[2], trs))

    # Generate the physiological noise
    if noise_dict['physiological_sigma'] != 0:

        # Calculate the physiological time course
        noise = _generate_noise_temporal_phys(timepoints,
                                              )

        # Create a brain shaped volume with similar smoothing properties
        volume = _generate_noise_spatial(dimensions=dimensions,
                                         mask=mask,
                                         fwhm=noise_dict['fwhm'],
                                         )

        # Combine the volume and noise
        noise_volume += np.multiply.outer(volume, noise) * noise_dict[
            'physiological_sigma']

    # Generate the AR noise
    if noise_dict['auto_reg_sigma'] != 0:

        # Calculate the AR time course volume
        noise = _generate_noise_temporal_autoregression(timepoints,
                                                        noise_dict,
                                                        dimensions,
                                                        mask,
                                                        )

        # Combine the volume and noise
        noise_volume += noise * noise_dict['auto_reg_sigma']

    # Generate the task related noise
    if noise_dict['task_sigma'] != 0 and np.sum(stimfunction_tr) > 0:

        # Calculate the task based noise time course
        noise = _generate_noise_temporal_task(stimfunction_tr,
                                              )

        # Create a brain shaped volume with similar smoothing properties
        volume = _generate_noise_spatial(dimensions=dimensions,
                                         mask=mask,
                                         fwhm=noise_dict['fwhm'],
                                         )
        # Combine the volume and noise
        noise_volume += np.multiply.outer(volume, noise) * noise_dict[
            'task_sigma']

    # Finally, z score each voxel so things mix nicely
    noise_volume = stats.zscore(noise_volume, 3)

    # If it is a nan it is because you just divided by zero (since some
    # voxels are zeros in the template)
    noise_volume[np.isnan(noise_volume)] = 0

    return noise_volume


def mask_brain(volume,
               template_name=None,
               mask_threshold=None,
               mask_self=True,
               ):
    """ Mask the simulated volume
    This creates a mask specifying the approximate likelihood that a voxel is
    part of the brain. All values are bounded to the range of 0 to 1. An
    appropriate threshold to isolate brain voxels is >0.2. Critically,
    the data that should be used to create a template shouldn't already be
    masked/skull stripped. If it is then it will give in accurate estimates
    of non-brain noise and corrupt estimations of SNR.

    Parameters
    ----------

    volume : multidimensional array
        Either numpy array of a volume or a tuple describing the dimensions
        of the mask to be created

    template_name : str
        What is the path to the template to be loaded? If empty then it
        defaults to an MNI152 grey matter mask. This is ignored if mask_self
        is True.

    mask_threshold : float
        What is the threshold (0 -> 1) for including a voxel in the mask? If
        None then the program will try and identify the last wide peak in a
        histogram of the template (assumed to be the brain voxels) and takes
        the minima before that peak as the threshold. Won't work when the
        data is not bimodal.

    mask_self : bool or None
        If set to true then it makes a mask from the volume supplied (by
        averaging across time points and changing the range). If it is set
        to false then it will use the template_name as an input.

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
        rf = files('brainiak').joinpath(
            'utils/sim_parameters/grey_matter_mask.npy')
        with rf.open('rb') as f:
            mask_raw = np.load(f)
    else:
        mask_raw = np.load(template_name)

    # Make the masks 3dremove_baseline
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
    # You might get a warning if the zoom_factor is not an integer but you
    # can safely ignore that.
    template = ndimage.zoom(mask_raw, zoom_factor, order=2)
    template[template < 0] = 0

    # If the mask threshold is not supplied then guess it is a minima
    # between the two peaks of the bimodal distribution of voxel activity
    if mask_threshold is None:

        # How many bins on either side of a peak will be compared
        order = 5

        # Make the histogram
        template_vector = template.reshape(brain_dim[0] * brain_dim[1] *
                                           brain_dim[2])
        template_hist = np.histogram(template_vector, 100)

        # Zero pad the values
        binval = np.concatenate([np.zeros((order,)), template_hist[0]])
        bins = np.concatenate([np.zeros((order,)), template_hist[1]])

        # Identify the first two peaks
        peaks = signal.argrelmax(binval, order=order)[0][0:2]

        # What is the minima between peaks
        minima = binval[peaks[0]:peaks[1]].min()

        # What is the index of the last idx with this min value (since if
        # zero, there may be many)
        minima_idx = (np.where(binval[peaks[0]:peaks[1]] == minima) + peaks[
            0])[-1]

        # Convert the minima into a threshold
        mask_threshold = bins[minima_idx][0]

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
        noise types interact in important ways. First, autoregressive,
        physiological and task-based noise types are mixed together in
        _generate_temporal_noise. The parameter values for 'auto_reg_sigma',
        'physiological_sigma' and 'task_sigma' describe the proportion of
        mixing of these elements, respectively. However critically, 'SFNR' is
        the parameter that controls how much noise these components contribute
        to the brain. 'auto_reg_rho' and 'ma_rho' set parameters for the
        autoregressive noise being simulated. Second, drift noise is added to
        this, according to the size of 'drift_sigma'. Thirdly, system noise is
        added based on the 'SNR' parameter. Finally, 'fwhm' is used to estimate
        the smoothness of the noise being inserted. If 'matched' is set to
        true, then it will fit the parameters to match the participant as best
        as possible.

        Variables defined as follows:

        snr [float]: Ratio of MR signal to the spatial noise
        sfnr [float]: Ratio of the MR signal to the temporal noise. This is the
        total variability that the following sigmas 'sum' to:

        task_sigma [float]: Size of the variance of task specific noise
        auto_reg_sigma [float]: Size of the variance of autoregressive
        noise. This is an ARMA process where the AR and MA components can be
        separately specified
        physiological_sigma [float]: Size of the variance of physiological
        noise

        drift_sigma [float]: Size of the variance of drift noise

        auto_reg_rho [list]: The coefficients of the autoregressive
        components you are modeling
        ma_rho [list]:The coefficients of the moving average components you
        are modeling
        max_activity [float]: The max value of the averaged brain in order
        to reference the template
        voxel_size [list]: The mm size of the voxels
        fwhm [float]: The gaussian smoothing kernel size (mm)
        matched [bool]: Specify whether you are fitting the noise parameters

        The volumes of brain noise that are generated have smoothness
        specified by 'fwhm'

    Returns
    -------

    noise_dict : dict
        Updated dictionary

    """
    # Create the default dictionary
    default_dict = {'task_sigma': 0, 'drift_sigma': 0, 'auto_reg_sigma': 1,
                    'auto_reg_rho': [0.5], 'ma_rho': [0.0],
                    'physiological_sigma': 0, 'sfnr': 90, 'snr': 50,
                    'max_activity': 1000, 'voxel_size': [1.0, 1.0, 1.0],
                    'fwhm': 4, 'matched': 1}

    # Check what noise is in the dictionary and add if necessary. Numbers
    # determine relative proportion of noise
    for default_key in default_dict:
        if default_key not in noise_dict:
            noise_dict[default_key] = default_dict[default_key]

    return noise_dict


def _fit_spatial(noise,
                 noise_temporal,
                 drift_noise,
                 mask,
                 template,
                 spatial_sd,
                 temporal_sd,
                 noise_dict,
                 fit_thresh,
                 fit_delta,
                 iterations,
                 ):
    """
    Fit the noise model to match the SNR of the data

    Parameters
    ----------

    noise : multidimensional array, float
        Initial estimate of the noise

    noise_temporal : multidimensional array, float
        The temporal noise that was generated by _generate_temporal_noise

    drift_noise : multidimensional array, float
        The drift noise generated by _generate_noise_temporal_drift

    tr_duration : float
        What is the duration, in seconds, of each TR?

    template : 3d array, float
        A continuous (0 -> 1) volume describing the likelihood a voxel is in
        the brain. This can be used to contrast the brain and non brain.

    mask : 3d array, binary
        The mask of the brain volume, distinguishing brain from non-brain

    spatial_sd : float
        What is the standard deviation in space of the noise volume to be
        generated

    temporal_sd : float
        What is the standard deviation in time of the noise volume to be
        generated

    noise_dict : dict
        A dictionary specifying the types of noise in this experiment. The
        noise types interact in important ways. First, autoregressive,
        physiological and task-based noise types are mixed together in
        _generate_temporal_noise. The parameter values for 'auto_reg_sigma',
        'physiological_sigma' and 'task_sigma' describe the proportion of
        mixing of these elements, respectively. However critically, 'SFNR' is
        the parameter that controls how much noise these components contribute
        to the brain. 'auto_reg_rho' and 'ma_rho' set parameters for the
        autoregressive noise being simulated. Second, drift noise is added to
        this, according to the size of 'drift_sigma'. Thirdly, system noise is
        added based on the 'SNR' parameter. Finally, 'fwhm' is used to estimate
        the smoothness of the noise being inserted. If 'matched' is set to
        true, then it will fit the parameters to match the participant as best
        as possible.

        Variables defined as follows:

        snr [float]: Ratio of MR signal to the spatial noise
        sfnr [float]: Ratio of the MR signal to the temporal noise. This is the
        total variability that the following sigmas 'sum' to:

        task_sigma [float]: Size of the variance of task specific noise
        auto_reg_sigma [float]: Size of the variance of autoregressive
        noise. This is an ARMA process where the AR and MA components can be
        separately specified
        physiological_sigma [float]: Size of the variance of physiological
        noise

        drift_sigma [float]: Size of the variance of drift noise

        auto_reg_rho [list]: The coefficients of the autoregressive
        components you are modeling
        ma_rho [list]:The coefficients of the moving average components you
        are modeling
        max_activity [float]: The max value of the averaged brain in order
        to reference the template
        voxel_size [list]: The mm size of the voxels
        fwhm [float]: The gaussian smoothing kernel size (mm)
        matched [bool]: Specify whether you are fitting the noise parameters

        The volumes of brain noise that are generated have smoothness
        specified by 'fwhm'

    fit_thresh : float
        What proportion of the target parameter value is sufficient error to
        warrant finishing fit search.

    fit_delta : float
        How much are the parameters attenuated during the fitting process,
        in terms of the proportion of difference between the target parameter
        and the actual parameter

    iterations : int
        The first element is how many steps of fitting the SFNR and SNR
        values will be performed. Usually converges after < 5. The second
        element is the number of iterations for the AR fitting. This is much
        more time consuming (has to make a new timecourse on each iteration)
        so be careful about setting this appropriately.

    Returns
    -------

    noise : multidimensional array, float
        Generates the noise volume given these parameters

    """

    # Pull out information that is needed
    dim_tr = noise.shape
    base = template * noise_dict['max_activity']
    base = base.reshape(dim_tr[0], dim_tr[1], dim_tr[2], 1)
    mean_signal = (base[mask > 0]).mean()
    target_snr = noise_dict['snr']

    # Iterate through different parameters to fit SNR and SFNR
    spat_sd_orig = np.copy(spatial_sd)
    iteration = 0
    for iteration in list(range(iterations)):

        # Calculate the new metrics
        new_snr = _calc_snr(noise, mask)

        # Calculate the difference between the real and simulated data
        diff_snr = abs(new_snr - target_snr) / target_snr

        # If the AR is sufficiently close then break the loop
        if diff_snr < fit_thresh:
            logger.info('Terminated SNR fit after ' + str(
                iteration) + ' iterations.')
            break

        # Convert the SFNR and SNR
        spat_sd_new = mean_signal / new_snr

        # Update the variable
        spatial_sd -= ((spat_sd_new - spat_sd_orig) * fit_delta)

        # Prevent these going out of range
        if spatial_sd < 0 or np.isnan(spatial_sd):
            spatial_sd = 10e-3

        # Set up the machine noise
        noise_system = _generate_noise_system(dimensions_tr=dim_tr,
                                              spatial_sd=spatial_sd,
                                              temporal_sd=temporal_sd,
                                              )

        # Sum up the noise of the brain
        noise = base + drift_noise + noise_system
        noise += (noise_temporal * temporal_sd)  # Add the brain specific noise

        # Reject negative values (only happens outside of the brain)
        noise[noise < 0] = 0

    # Failed to converge
    if iterations == 0:
        logger.info('No fitting iterations were run')
    elif iteration == iterations:
        logger.warning('SNR failed to converge.')

    # Return the updated noise
    return noise, spatial_sd


def _fit_temporal(noise,
                  mask,
                  template,
                  stimfunction_tr,
                  tr_duration,
                  spatial_sd,
                  temporal_proportion,
                  temporal_sd,
                  drift_noise,
                  noise_dict,
                  fit_thresh,
                  fit_delta,
                  iterations,
                  ):
    """
    Fit the noise model to match the SFNR and AR of the data

    Parameters
    ----------

    noise : multidimensional array, float
        Initial estimate of the noise

    mask : 3d array, binary
        The mask of the brain volume, distinguishing brain from non-brain

    template : 3d array, float
        A continuous (0 -> 1) volume describing the likelihood a voxel is in
        the brain. This can be used to contrast the brain and non brain.

    stimfunction_tr :  Iterable, list
        When do the stimuli events occur. Each element is a TR

    tr_duration : float
        What is the duration, in seconds, of each TR?

    spatial_sd : float
        What is the standard deviation in space of the noise volume to be
        generated

    temporal_proportion, float
        What is the proportion of the temporal variance (as specified by the
        SFNR noise parameter) that is accounted for by the system noise. If
        this number is high then all of the temporal variability is due to
        system noise, if it is low then all of the temporal variability is
        due to brain variability.

    temporal_sd : float
        What is the standard deviation in time of the noise volume to be
        generated

    drift_noise : multidimensional array, float
        The drift noise generated by _generate_noise_temporal_drift

    noise_dict : dict
        A dictionary specifying the types of noise in this experiment. The
        noise types interact in important ways. First, autoregressive,
        physiological and task-based noise types are mixed together in
        _generate_temporal_noise. The parameter values for 'auto_reg_sigma',
        'physiological_sigma' and 'task_sigma' describe the proportion of
        mixing of these elements, respectively. However critically, 'SFNR' is
        the parameter that controls how much noise these components contribute
        to the brain. 'auto_reg_rho' and 'ma_rho' set parameters for the
        autoregressive noise being simulated. Second, drift noise is added to
        this, according to the size of 'drift_sigma'. Thirdly, system noise is
        added based on the 'SNR' parameter. Finally, 'fwhm' is used to estimate
        the smoothness of the noise being inserted. If 'matched' is set to
        true, then it will fit the parameters to match the participant as best
        as possible.


        Variables defined as follows:

        snr [float]: Ratio of MR signal to the spatial noise
        sfnr [float]: Ratio of the MR signal to the temporal noise. This is the
        total variability that the following sigmas 'sum' to:

        task_sigma [float]: Size of the variance of task specific noise
        auto_reg_sigma [float]: Size of the variance of autoregressive
        noise. This is an ARMA process where the AR and MA components can be
        separately specified
        physiological_sigma [float]: Size of the variance of physiological
        noise

        drift_sigma [float]: Size of the variance of drift noise

        auto_reg_rho [list]: The coefficients of the autoregressive
        components you are modeling
        ma_rho [list]:The coefficients of the moving average components you
        are modeling
        max_activity [float]: The max value of the averaged brain in order
        to reference the template
        voxel_size [list]: The mm size of the voxels
        fwhm [float]: The gaussian smoothing kernel size (mm)
        matched [bool]: Specify whether you are fitting the noise parameters

        The volumes of brain noise that are generated have smoothness
        specified by 'fwhm'

    fit_thresh : float
        What proportion of the target parameter value is sufficient error to
        warrant finishing fit search.

    fit_delta : float
        How much are the parameters attenuated during the fitting process,
        in terms of the proportion of difference between the target parameter
        and the actual parameter

    iterations : list, int
        The first element is how many steps of fitting the SFNR and SNR
        values will be performed. Usually converges after < 5. The second
        element is the number of iterations for the AR fitting. This is much
        more time consuming (has to make a new timecourse on each iteration)
        so be careful about setting this appropriately.

    Returns
    -------

    noise : multidimensional array, float
        Generates the noise volume given these parameters

    """

    # Pull out the
    dim_tr = noise.shape
    dim = dim_tr[0:3]
    base = template * noise_dict['max_activity']
    base = base.reshape(dim[0], dim[1], dim[2], 1)
    mean_signal = (base[mask > 0]).mean()

    # Iterate through different parameters to fit SNR and SFNR
    temp_sd_orig = np.copy(temporal_sd)

    # Make a copy of the dictionary so it can be modified
    new_nd = copy.deepcopy(noise_dict)

    # What SFNR do you want
    target_sfnr = noise_dict['sfnr']

    # What AR do you want?
    target_ar = noise_dict['auto_reg_rho'][0]

    # Iterate through different MA parameters to fit AR
    for iteration in list(range(iterations)):

        # If there are iterations left to perform then recalculate the
        # metrics and try again

        # Calculate the new SFNR
        new_sfnr = _calc_sfnr(noise, mask)

        # Calculate the AR
        new_ar, _ = _calc_ARMA_noise(noise,
                                     mask,
                                     len(noise_dict['auto_reg_rho']),
                                     len(noise_dict['ma_rho']),
                                     )

        # Calculate the difference between the real and simulated data
        sfnr_diff = abs(new_sfnr - target_sfnr) / target_sfnr

        # Calculate the difference in the first AR component
        ar_diff = new_ar[0] - target_ar

        # If the SFNR and AR is sufficiently close then break the loop
        if (abs(ar_diff) / target_ar) < fit_thresh and sfnr_diff < fit_thresh:
            msg = 'Terminated AR fit after ' + str(iteration) + ' iterations.'
            logger.info(msg)
            break

        # Otherwise update the noise metrics. Get the new temporal noise value
        temp_sd_new = mean_signal / new_sfnr
        temporal_sd -= ((temp_sd_new - temp_sd_orig) * fit_delta)

        # Prevent these going out of range
        if temporal_sd < 0 or np.isnan(temporal_sd):
            temporal_sd = 10e-3

        # Set the new system noise
        temp_sd_system_new = np.sqrt((temporal_sd ** 2) * temporal_proportion)

        # Get the new AR value
        new_nd['auto_reg_rho'][0] -= (ar_diff * fit_delta)

        # Don't let the AR coefficient exceed 1
        if new_nd['auto_reg_rho'][0] >= 1:
            new_nd['auto_reg_rho'][0] = 0.99

        # Generate the noise. The appropriate
        noise_temporal = _generate_noise_temporal(stimfunction_tr,
                                                  tr_duration,
                                                  dim,
                                                  template,
                                                  mask,
                                                  new_nd,
                                                  )

        # Set up the machine noise
        noise_system = _generate_noise_system(dimensions_tr=dim_tr,
                                              spatial_sd=spatial_sd,
                                              temporal_sd=temp_sd_system_new,
                                              )

        # Sum up the noise of the brain
        noise = base + drift_noise + noise_system
        noise += (noise_temporal * temporal_sd)  # Add the brain specific noise

        # Reject negative values (only happens outside of the brain)
        noise[noise < 0] = 0

    # Failed to converge
    if iterations == 0:
        logger.info('No fitting iterations were run')
    elif iteration == iterations:
        logger.warning('AR failed to converge.')

    # Return the updated noise
    return noise


def generate_noise(dimensions,
                   stimfunction_tr,
                   tr_duration,
                   template,
                   mask=None,
                   noise_dict=None,
                   temporal_proportion=0.5,
                   iterations=None,
                   fit_thresh=0.05,
                   fit_delta=0.5,
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

    noise_dict : dict
        A dictionary specifying the types of noise in this experiment. The
        noise types interact in important ways. First, autoregressive,
        physiological and task-based noise types are mixed together in
        _generate_temporal_noise. The parameter values for 'auto_reg_sigma',
        'physiological_sigma' and 'task_sigma' describe the proportion of
        mixing of these elements, respectively. However critically, 'SFNR' is
        the parameter that controls how much noise these components contribute
        to the brain. 'auto_reg_rho' and 'ma_rho' set parameters for the
        autoregressive noise being simulated. Second, drift noise is added to
        this, according to the size of 'drift_sigma'. Thirdly, system noise is
        added based on the 'SNR' parameter. Finally, 'fwhm' is used to estimate
        the smoothness of the noise being inserted. If 'matched' is set to
        true, then it will fit the parameters to match the participant as best
        as possible.

        Variables defined as follows:

        snr [float]: Ratio of MR signal to the spatial noise
        sfnr [float]: Ratio of the MR signal to the temporal noise. This is the
        total variability that the following sigmas 'sum' to:

        task_sigma [float]: Size of the variance of task specific noise
        auto_reg_sigma [float]: Size of the variance of autoregressive
        noise. This is an ARMA process where the AR and MA components can be
        separately specified
        physiological_sigma [float]: Size of the variance of physiological
        noise

        drift_sigma [float]: Size of the variance of drift noise

        auto_reg_rho [list]: The coefficients of the autoregressive
        components you are modeling
        ma_rho [list]:The coefficients of the moving average components you
        are modeling
        max_activity [float]: The max value of the averaged brain in order
        to reference the template
        voxel_size [list]: The mm size of the voxels
        fwhm [float]: The gaussian smoothing kernel size (mm)
        matched [bool]: Specify whether you are fitting the noise parameters

        The volumes of brain noise that are generated have smoothness
        specified by 'fwhm'

    temporal_proportion, float
        What is the proportion of the temporal variance (as specified by the
        SFNR noise parameter) that is accounted for by the system noise. If
        this number is high then all of the temporal variability is due to
        system noise, if it is low then all of the temporal variability is
        due to brain variability.

    iterations : list, int
        The first element is how many steps of fitting the SFNR and SNR values
        will be performed. Usually converges after < 5. The second element
        is the number of iterations for the AR fitting. This is much more
        time consuming (has to make a new timecourse on each iteration) so
        be careful about setting this appropriately.

    fit_thresh : float
        What proportion of the target parameter value is sufficient error to
        warrant finishing fit search.

    fit_delta : float
        How much are the parameters attenuated during the fitting process,
        in terms of the proportion of difference between the target
        parameter and the actual parameter


    Returns
    ----------

    noise : multidimensional array, float
        Generates the noise volume for these parameters

    """

    # Check the input data
    if template.max() > 1.1:
        raise ValueError('Template out of range')

    # Change to be an empty dictionary if it is None
    if noise_dict is None:
        noise_dict = {}

    # Take in the noise dictionary and add any missing information
    noise_dict = _noise_dict_update(noise_dict)

    # How many iterations will you perform? If unspecified it will set
    # values based on whether you are trying to match noise specifically to
    # this participant or just get in the ball park
    if iterations is None:
        if noise_dict['matched'] == 1:
            iterations = [20, 20]
        else:
            iterations = [0, 0]

    if abs(noise_dict['auto_reg_rho'][0]) - abs(noise_dict['ma_rho'][0]) < 0.1:
        logger.warning('ARMA coefs are close, may have trouble fitting')

    # What are the dimensions of the volume, including time
    dimensions_tr = (dimensions[0],
                     dimensions[1],
                     dimensions[2],
                     len(stimfunction_tr))

    # Get the mask of the brain and set it to be 3d
    if mask is None:
        mask = np.ones(dimensions)

    # Create the base (this inverts the process to make the template)
    base = template * noise_dict['max_activity']

    # Reshape the base (to be the same size as the volume to be created)
    base = base.reshape(dimensions[0], dimensions[1], dimensions[2], 1)
    base = np.ones(dimensions_tr) * base

    # What is the mean signal of the non masked voxels in this template?
    mean_signal = (base[mask > 0]).mean()

    # Generate the temporal noise
    noise_temporal = _generate_noise_temporal(stimfunction_tr=stimfunction_tr,
                                              tr_duration=tr_duration,
                                              dimensions=dimensions,
                                              template=template,
                                              mask=mask,
                                              noise_dict=noise_dict,
                                              )

    # Generate the drift noise
    if noise_dict['drift_sigma'] != 0:
        # Calculate the drift time course
        noise = _generate_noise_temporal_drift(len(stimfunction_tr),
                                               tr_duration,
                                               )
        # Create a volume with the drift properties
        volume = np.ones(dimensions[:3])

        # Combine the volume and noise
        drift_noise = np.multiply.outer(volume, noise) * noise_dict[
            'drift_sigma']
    else:
        # If there is no drift, then just make this zeros (in 4d)
        drift_noise = np.zeros(dimensions_tr)

    # Convert SFNR into the size of the standard deviation of temporal
    # variability
    temporal_sd = (mean_signal / noise_dict['sfnr'])

    # Calculate the temporal sd of the system noise (as opposed to the noise
    #  attributed to the functional variability).
    temporal_sd_system = np.sqrt((temporal_sd ** 2) * temporal_proportion)

    # What is the standard deviation of the background activity
    spat_sd = mean_signal / noise_dict['snr']
    spatial_sd = np.sqrt((spat_sd ** 2) * (1 - temporal_proportion))

    # Set up the machine noise
    noise_system = _generate_noise_system(dimensions_tr=dimensions_tr,
                                          spatial_sd=spatial_sd,
                                          temporal_sd=temporal_sd_system,
                                          )

    # Sum up the noise of the brain
    noise = base + drift_noise + noise_system
    noise += (noise_temporal * temporal_sd)  # Add the brain specific noise

    # Reject negative values (only happens outside of the brain)
    noise[noise < 0] = 0

    # Fit the SNR
    noise, spatial_sd = _fit_spatial(noise,
                                     noise_temporal,
                                     drift_noise,
                                     mask,
                                     template,
                                     spatial_sd,
                                     temporal_sd_system,
                                     noise_dict,
                                     fit_thresh,
                                     fit_delta,
                                     iterations[0],
                                     )

    # Fit the SFNR and AR noise
    noise = _fit_temporal(noise,
                          mask,
                          template,
                          stimfunction_tr,
                          tr_duration,
                          spatial_sd,
                          temporal_proportion,
                          temporal_sd,
                          drift_noise,
                          noise_dict,
                          fit_thresh,
                          fit_delta,
                          iterations[1],
                          )

    # Return the noise
    return noise


def compute_signal_change(signal_function,
                          noise_function,
                          noise_dict,
                          magnitude,
                          method='PSC',
                          ):
    """ Rescale the signal to be a given magnitude, based on a specified
    metric (e.g. percent signal change). Metrics are heavily inspired by
    Welvaert & Rosseel (2013). The rescaling is based on the maximal
    activity in the timecourse. Importantly, all values within the
    signal_function are scaled to have a min of -1 or max of 1, meaning that
    the voxel value will be the same as the magnitude.

    Parameters
    ----------


    signal_function : timepoint by voxel array
        The signal time course to be altered. This can have
        multiple time courses specified as different columns in this
        array. Conceivably you could use the output of
        generate_stimfunction as the input but the temporal variance
        will be incorrect. Critically, different values across voxels are
        considered relative to each other, not independently. E.g., if the
        voxel has a peak signal twice as high as another voxel's, then this
        means that the signal after these transformations will still be
        twice as high (according to the metric) in the first voxel relative
        to the second

    noise_function : timepoint by voxel numpy array
        The time course of noise (a voxel created from generate_noise)
        for each voxel specified in signal_function. This is necessary
        for computing the mean evoked activity and the noise variability

    noise_dict : dict
        A dictionary specifying the types of noise in this experiment. The
        noise types interact in important ways. First, autoregressive,
        physiological and task-based noise types are mixed together in
        _generate_temporal_noise. The parameter values for 'auto_reg_sigma',
        'physiological_sigma' and 'task_sigma' describe the proportion of
        mixing of these elements, respectively. However critically, 'SFNR' is
        the parameter that controls how much noise these components contribute
        to the brain. 'auto_reg_rho' and 'ma_rho' set parameters for the
        autoregressive noise being simulated. Second, drift noise is added to
        this, according to the size of 'drift_sigma'. Thirdly, system noise is
        added based on the 'SNR' parameter. Finally, 'fwhm' is used to estimate
        the smoothness of the noise being inserted. If 'matched' is set to
        true, then it will fit the parameters to match the participant as best
        as possible.

        Variables defined as follows:

        snr [float]: Ratio of MR signal to the spatial noise
        sfnr [float]: Ratio of the MR signal to the temporal noise. This is the
        total variability that the following sigmas 'sum' to:

        task_sigma [float]: Size of the variance of task specific noise
        auto_reg_sigma [float]: Size of the variance of autoregressive
        noise. This is an ARMA process where the AR and MA components can be
        separately specified
        physiological_sigma [float]: Size of the variance of physiological
        noise

        drift_sigma [float]: Size of the variance of drift noise

        auto_reg_rho [list]: The coefficients of the autoregressive
        components you are modeling
        ma_rho [list]:The coefficients of the moving average components you
        are modeling
        max_activity [float]: The max value of the averaged brain in order
        to reference the template
        voxel_size [list]: The mm size of the voxels
        fwhm [float]: The gaussian smoothing kernel size (mm)
        matched [bool]: Specify whether you are fitting the noise parameters

        The volumes of brain noise that are generated have smoothness
        specified by 'fwhm'

    magnitude : list of floats
        This specifies the size, in terms of the metric choosen below,
        of the signal being generated. This can be a single number,
        and thus apply to all signal timecourses, or it can be array and
        thus different for each voxel.

    method : str
        Select the procedure used to calculate the signal magnitude,
        some of which are based on the definitions outlined in Welvaert &
        Rosseel (2013):
        - 'SFNR': Change proportional to the temporal variability,
        as represented by the (desired) SFNR
        - 'CNR_Amp/Noise-SD': Signal magnitude relative to the temporal
        noise
        - 'CNR_Amp2/Noise-Var_dB': Same as above but converted to decibels
        - 'CNR_Signal-SD/Noise-SD': Standard deviation in signal
        relative to standard deviation in noise
        - 'CNR_Signal-Var/Noise-Var_dB': Same as above but converted to
        decibels
        - 'PSC': Calculate the percent signal change based on the
        average activity of the noise (mean / 100 * magnitude)


    Returns
    ----------
    signal_function_scaled : 4d numpy array
        The new signal volume with the appropriately set signal change

    """

    # If you have only one magnitude value, duplicate the magnitude for each
    #  timecourse you have
    assert type(magnitude) is list, '"magnitude" should be a list of floats'
    if len(magnitude) == 1:
        magnitude *= signal_function.shape[1]

    # Check that the signal_function and noise_function are the same size
    if signal_function.shape != noise_function.shape:
        msg = 'noise_function is not the same size as signal_function'
        raise ValueError(msg)

    # Scale all signals that to have a range of -1 to 1. This is
    # so that any values less than this will be scaled appropriately
    signal_function /= np.max(np.abs(signal_function))

    # Iterate through the timecourses and calculate the metric
    signal_function_scaled = np.zeros(signal_function.shape)
    for voxel_counter in range(signal_function.shape[1]):

        # Pull out the values for this voxel
        sig_voxel = signal_function[:, voxel_counter]
        noise_voxel = noise_function[:, voxel_counter]
        magnitude_voxel = magnitude[voxel_counter]

        # Calculate the maximum signal amplitude (likely to be 1,
        # but not necessarily)
        max_amp = np.max(np.abs(sig_voxel))

        # Calculate the scaled time course using the specified method
        if method == 'SFNR':

            # How much temporal variation is there, relative to the mean
            # activity
            temporal_var = noise_voxel.mean() / noise_dict['sfnr']

            # Multiply the timecourse by the variability metric
            new_sig = sig_voxel * (temporal_var * magnitude_voxel)

        elif method == 'CNR_Amp/Noise-SD':

            # What is the standard deviation of the noise
            noise_std = np.std(noise_voxel)

            # Multiply the signal timecourse by the the CNR and noise (
            # rearranging eq.)
            new_sig = sig_voxel * (magnitude_voxel * noise_std)

        elif method == 'CNR_Amp2/Noise-Var_dB':

            # What is the standard deviation of the noise
            noise_std = np.std(noise_voxel)

            # Rearrange the equation to compute the size of signal change in
            #  decibels
            scale = (10 ** (magnitude_voxel / 20)) * noise_std / max_amp

            new_sig = sig_voxel * scale

        elif method == 'CNR_Signal-SD/Noise-SD':

            # What is the standard deviation of the signal and noise
            sig_std = np.std(sig_voxel)
            noise_std = np.std(noise_voxel)

            # Multiply the signal timecourse by the the CNR and noise (
            # rearranging eq.)
            new_sig = sig_voxel * ((magnitude_voxel / max_amp) * noise_std
                                   / sig_std)

        elif method == 'CNR_Signal-Var/Noise-Var_dB':
            # What is the standard deviation of the signal and noise
            sig_std = np.std(sig_voxel)
            noise_std = np.std(noise_voxel)

            # Rearrange the equation to compute the size of signal change in
            #  decibels
            scale = (10 ** (magnitude_voxel / 20)) * noise_std / (max_amp *
                                                                  sig_std)

            new_sig = sig_voxel * scale

        elif method == 'PSC':

            # What is the average activity divided by percentage
            scale = ((noise_voxel.mean() / 100) * magnitude_voxel)
            new_sig = sig_voxel * scale

        signal_function_scaled[:, voxel_counter] = new_sig

    # Return the scaled time course
    return signal_function_scaled


def generate_1d_gaussian_rfs(n_voxels, feature_resolution, feature_range,
                             rf_size=15, random_tuning=True, rf_noise=0.):
    """
    Creates a numpy matrix of Gaussian-shaped voxel receptive fields (RFs)
    along one dimension. Can specify whether they are evenly tiled or randomly
    tuned along the axis. RF range will be between 0 and 1.

    Parameters
    ----------

    n_voxels : int
        Number of voxel RFs to create.

    feature_resolution : int
        Number of points along the feature axis.

    feature_range : tuple (numeric)
        A tuple indicating the start and end values of the feature range. e.g.
        (0, 359) for motion directions.

    rf_size : numeric
        Width of the Gaussian receptive field. Should be given in units of the
        feature dimension. e.g., 15 degrees wide in motion direction space.

    random_tuning : boolean [default True]
        Indicates whether or not the voxels are randomly tuned along the 1D
        feature axis or whether tuning is evenly spaced.

    rf_noise : float [default 0.]
        Amount of uniform noise to add to the Gaussian RF. This will cause the
        generated responses to be distorted by the same uniform noise for a
        given voxel.

    Returns
    ----------

    voxel_rfs : 2d numpy array (float)
        The receptive fields in feature space. Dimensions are n_voxels by
        feature_resolution.

    voxel_tuning : 1d numpy array (float)
        The centers of the voxel RFs, in feature space.

    """
    range_start, range_stop = feature_range
    if random_tuning:
        # Voxel selectivity is random
        voxel_tuning = np.floor((np.random.rand(n_voxels) * range_stop)
                                + range_start).astype(int)
    else:
        # Voxel selectivity is evenly spaced along the feature axis
        voxel_tuning = np.linspace(range_start, range_stop, n_voxels + 1)
        voxel_tuning = voxel_tuning[0:-1]
        voxel_tuning = np.floor(voxel_tuning).astype(int)
    gaussian = signal.windows.gaussian(feature_resolution, rf_size)
    voxel_rfs = np.zeros((n_voxels, feature_resolution))
    for i in range(0, n_voxels):
        voxel_rfs[i, :] = np.roll(gaussian, voxel_tuning[i] -
                                  ((feature_resolution // 2) - 1))
    voxel_rfs += np.random.rand(n_voxels, feature_resolution) * rf_noise
    voxel_rfs = voxel_rfs / np.max(voxel_rfs, axis=1)[:, None]

    return voxel_rfs, voxel_tuning


def generate_1d_rf_responses(rfs, trial_list, feature_resolution,
                             feature_range, trial_noise=0.25):
    """
    Generates trial-wise data for a given set of receptive fields (RFs) and
    a 1d array of features presented across trials.

    Parameters
    ----------

    voxel_rfs : 2d numpy array (float)
        The receptive fields in feature space. Dimensions must be n_voxels
        by feature_resolution.

    trial_list : 1d numpy array (numeric)
        The feature value of the stimulus presented on individual trials.
        Array size be n_trials.

    feature_resolution : int
        Number of points along the feature axis.

    feature_range : tuple (numeric)
        A tuple indicating the start and end values of the feature range. e.g.
        (0, 359) for motion directions.

    trial_noise : float [default 0.25]
        Amount of uniform noise to inject into the synthetic data. This is
        generated independently for every trial and voxel.

    Returns
    ----------

    trial_data : 2d numpy array (float)
        The synthetic data for each voxel and trial. Dimensions are n_voxels by
        n_trials.

    """
    range_start, range_stop = feature_range
    stim_axis = np.linspace(range_start, range_stop,
                            feature_resolution)
    if range_start > 0:
        trial_list = trial_list + range_start
    elif range_start < 0:
        trial_list = trial_list - range_start
    one_hot = np.eye(feature_resolution)
    indices = [np.argmin(abs(stim_axis - x)) for x in trial_list]
    stimulus_mask = one_hot[:, indices]
    trial_data = rfs @ stimulus_mask
    trial_data += np.random.rand(rfs.shape[0], trial_list.size) * \
        (trial_noise * np.max(trial_data))

    return trial_data
