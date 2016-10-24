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
simulations of neural data.

Steps:

generate_signal
Specify the volume (or volumes) which represent the signal in the neural data.

generate_stimfunction
Create a function to describe the stimulus onsets/durations

double_gamma_hrf
Convolve the stimulus function with the HRF to model when events are expected.

apply_signal
Combine the volume and the HRF

generate_noise
Create the noise for this run. This creates temporal, task and white noise.
Various parameters can be tuned depending on need

mask_brain
Mask the volume to look like a volume. Based on MNI standard space

plot_brain
Show the brain as it unfolds over time with a given opacity.


 Authors: Cameron Ellis (Princeton) 2016
"""
import logging

import numpy as np
import scipy.ndimage as ndimage
import math
from scipy import stats
from pkg_resources import resource_stream

__all__ = [
    "generate_signal",
    "generate_stimfunction",
    "double_gamma_hrf",
    "apply_signal",
    "generate_noise",
    "mask_brain",
    "plot_brain",
]

logger = logging.getLogger(__name__)


def _generate_feature(feature_type, feature_size, signal_magnitude):
    """Generate features corresponding to signal

    Generate signal in specific regions of the brain with for a single
    volume. This will then be convolved with the HRF across time

    Parameters
    ----------

    feature_type : str
        What feature_type of signal is being inserted? Options are cube,
        loop, cavity, sphere.

    feature_size : int
        How big is the signal?

    signal_magnitude : float
        What is the value of the signal voxels in the feature

    Returns
    ----------

    3 dimensional array
        The volume representing the signal to be outputed

    """

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
        disk = ((xx - ((feature_size - 1) / 2)) ** 2
                + (yy - ((feature_size - 1) / 2)) ** 2)

        # What is the outer disk
        outer = disk < (feature_size - 1)

        # What is the inner disk
        inner = disk < (feature_size - 1) / 2

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

        # Is the signal a sphere or a cavity?
        if feature_type == 'sphere':
            signal = signal < (feature_size - 1)

        else:
            # Get the inner and outer sphere
            outer = signal < (feature_size - 1)
            inner = signal < (feature_size - 1) / 2

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
    """Returns the indexes of where to put the signal into Volume_Static

    Parameters
    ----------

    feature_centre : list, int
        List of coordinates for the centre location of the signal

    feature_size : list, int
        How big is the signal.

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
    x_idx = [int(feature_centre[0] - (feature_size / 2)),
             int(feature_centre[0] - (feature_size / 2) +
                 feature_size)]
    y_idx = [int(feature_centre[1] - (feature_size / 2)),
             int(feature_centre[1] - (feature_size / 2) +
                 feature_size)]
    z_idx = [int(feature_centre[2] - (feature_size / 2)),
             int(feature_centre[2] - (feature_size / 2) +
                 feature_size)]

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
                    signal_magnitude,
                    signal_constant=1,
                    ):
    """Generate volume containing signal

    Generate signal in specific regions of the brain with for a single
    volume. This will then be convolved with the HRF across time

    Parameters
    ----------

    dimensions : 3 length array, int
        What are the dimensions of the volume you wish to create

    feature_coordinates : multidimensional array, int
        What are the feature_coordinates of the signal being created.
        Be aware of clipping: features far from the centre of the
        brain will be clipped. If you wish to have multiple features
        then list these as an features x 3 array. To create a signal of
        a specific shape then supply all the individual
        feature_coordinates and set the feature_size to 1.

    feature_size : list, int
        How big is the signal. If m=1 then only one value is accepted,
        if m>1 then either one value must be supplied or m values

    feature_type : list, string
        What feature_type of signal is being inserted? Options are cube,
        loop, cavity, sphere. If features = 1 then
        only one value is accepted, if features > 1 then either one value
        must be supplied or m values

    signal_magnitude : list, float
        What is the (average) magnitude of the signal being generated?

    signal_constant : list, bool
        Is the signal constant or is it a random pattern (with the same
        average magnitude)

    Returns
    ----------
    volume_static : 3 dimensional array, float
        Creates a single volume containing the signal

    """

    # Preset the volume
    volume_static = np.zeros(dimensions)

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
        volume_static[x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], z_idx[0]:z_idx[
            1]] = signal

    return volume_static


def generate_stimfunction(onsets,
                          event_durations,
                          total_time,
                          tr_duration,
                          ):
    """Return the function for the onset of events

    When do stimuli onset, how long for and to what extent should you
    resolve the fMRI time course

    Parameters
    ----------

        onsets : list, int
            What are the timestamps for when an event you want to generate
            onsets?

        event_durations : list, int
            What are the durations of the events you want to generate? If
            there is only one value then this will be assigned to all onsets

        total_time : int
            How long is the experiment in total.

        tr_duration : float
            What is the TR duration, related to the precision of the boxcar.

    Returns
    ----------

    Iterable[bool]
        The time course of stimulus related activation

    """

    # If only one duration is supplied then duplicate it for the length of
    # the onset variable
    if len(event_durations) == 1:
        event_durations = event_durations * len(onsets)

    if (event_durations[0] / tr_duration).is_integer() is False:
        logging.warning('Event durations are not a multiple of the TR '
                        'duration, rounding down.')

    if (onsets[0] / tr_duration).is_integer() is False:
        logging.warning('Onsets are not a multiple of the TR duration, '
                        'rounding down.')

    # Generate the time course as empty
    stimfunction = [0] * round(total_time / tr_duration)

    # Cycle through the onsets
    for onset_counter in list(range(0, len(onsets))):
        # Adjust for the resolution
        onset_idx = int(np.floor(onsets[onset_counter] / tr_duration))

        # Adjust for the resolution
        offset_idx = int(np.floor((onsets[onset_counter] + event_durations[
            onset_counter]) / tr_duration))

        # For the appropriate indexes and duration, make this value 1
        idx_number = round(event_durations[onset_counter] / tr_duration)
        stimfunction[onset_idx:offset_idx] = idx_number * [1]

    # Remove any indexes that are too long
    if len(stimfunction) > total_time / tr_duration:
        stimfunction = stimfunction[0:int(total_time / tr_duration)]

    return stimfunction


def double_gamma_hrf(stimfunction,
                     response_delay=6,
                     undershoot_delay=12,
                     response_dispersion=0.9,
                     undershoot_dispersion=0.9,
                     undershoot_scale=0.035):
    """Return a double gamma HRF

    Parameters
    ----------
        stimfunction : list, bool
            What is the time course of events to be modelled in this
            experiment

        response_delay : float
            How many seconds until the peak of the HRF

        undershoot_delay : float
            How many seconds until the trough of the HRF

        response_dispersion : float
            How wide is the rising peak dispersion

        undershoot_dispersion : float
            How wide is the undershoot dispersion

        undershoot_scale :float
            How big is the undershoot relative to the peak

    Returns
    ----------

        one dimensional array
            The time course of the HRF convolved with the stimulus function


    """

    hrf_length = 30  # How long is the HRF being created

    hrf = [0] * hrf_length  # How many seconds of the HRF will you model?

    for hrf_counter in list(range(0, hrf_length - 1)):
        # When is the peak of the two aspects of the HRF
        response_peak = response_delay * response_dispersion
        undershoot_peak = undershoot_delay * undershoot_dispersion

        # Specify the elements of the HRF for both the response and undershoot
        resp_pow = math.pow(hrf_counter / response_peak, response_delay)
        resp_exp = math.exp(-(hrf_counter - response_peak) /
                            response_dispersion)

        response_model = resp_pow * resp_exp

        undershoot_pow = math.pow(hrf_counter / undershoot_peak,
                                  undershoot_delay)
        undershoot_exp = math.exp(-(hrf_counter - undershoot_peak
                                    / undershoot_dispersion))

        undershoot_model = undershoot_scale * undershoot_pow * undershoot_exp

        # For this time point find the value of the HRF
        hrf[hrf_counter] = response_model - undershoot_model

    # Convolve the hrf that was created with the boxcar input
    signal_function = np.convolve(stimfunction, hrf)

    # Cut off the HRF
    signal_function = signal_function[0:len(stimfunction)]

    return signal_function


def apply_signal(signal_function,
                 volume_static):
    """Apply the convolution and stimfunction

    Apply the convolution of the HRF and stimulus time course to the
    volume.

    Parameters
    ----------
        signal_function : list, float
            The one dimensional timecourse of the signal over time.
            Found by convolving the HRF with the stimulus time course.

        volume_static : multi dimensional array, float


    Returns
    ----------
    multidimensional array, float
        Generates the spatial noise volume for these parameters

        """

    # Preset volume
    signal = np.ndarray([volume_static.shape[0], volume_static.shape[
        1], volume_static.shape[2], len(signal_function)])

    # Iterate through the brain, multiplying the volume by the HRF
    for tr_counter in list(range(0, len(signal_function))):
        signal[0:volume_static.shape[0],
               0:volume_static.shape[1],
               0:volume_static.shape[2],
               tr_counter] = signal_function[tr_counter] * volume_static

    return signal


def _generate_noise_system(dimensions,
                           sigma=1.5,
                           ):
    """Generate the scanner noise

    Generate the noise that is typical of a scanner. This is comprised
    of two types of noise, Rician and Gaussian

    Parameters
    ----------
        dimensions : n length array, int
            What are the dimensions of the volume you wish to insert
            noise into. This can be a volume of any size

        sigma : float
            What is the standard deviation of this noise?

    Returns
    ----------
        system_noise : multidimensional array, float
            Create a volume with system noise


        """

    # Generate the Rician noise
    noise_rician = stats.rice.rvs(1, sigma, size=dimensions)

    # Apply the gaussian noise
    noise_gaussian = np.random.normal(0, sigma, size=dimensions)

    # Combine these two noise types
    noise_system = noise_rician + noise_gaussian

    return noise_system


def _generate_noise_temporal_task(stimfunction,
                                  motion_sigma,
                                  motion_noise='gaussian',
                                  ):
    """Generate the signal dependent noise

    This noise depends on things like the signal or the timing of the
    experiment.

    Parameters
    ----------

    stimfunction : 1 Dimensional array
        This is the timecourse of the stimuli in this experiment

    motion_sigma : float

        How much noise is left over after pre-processing has been
        done. This is noise specifically on the task events

    motion_noise : str
        What type of noise will you generate? Can be gaussian or rician

    Returns
    ----------
    one dimensional array, float
        Generates the temporal task noise timecourse


    """

    noise_task = []
    if motion_noise == 'gaussian':
        noise_task = stimfunction + (stimfunction *
                                     np.random.normal(0,
                                                      motion_sigma,
                                                      size=len(stimfunction)))
    elif motion_noise == 'rician':
        noise_task = stimfunction + (stimfunction *
                                     stats.rice.rvs(0,
                                                    motion_sigma,
                                                    size=len(stimfunction)))

    return noise_task


def _generate_noise_temporal_drift(trs,
                                   tr_duration,
                                   drift_sigma,
                                   timepoints,
                                   ):

    """Generate the drift noise

    According to AFNI (https://afni.nimh.nih.gov/pub/dist/doc/
    program_help/3dDeconvolve.html) the appropriate order of the
    polynomial to fit for temporal drift is calculated as follows

    Parameters
    ----------

    trs : int
        How many TRs are there

    tr_duration : float
        How long is each TR

    drift_sigma : float
        How large are the coefficients controlling drift

    timepoints : 1 Dimensional array
        What time points are sampled by a TR

    Returns
    ----------
    one dimensional array, float
        Generates the autoregression noise timecourse

    """

    # Calculate the coefficients of the drift for a given function
    degree = round(trs * tr_duration / 150) + 1
    coefficients = np.random.normal(0, drift_sigma, size=degree)

    # What are the values of this drift
    noise_drift = np.polyval(coefficients, timepoints)

    # Return noise
    return noise_drift


def _generate_noise_temporal_autoregression(auto_reg_rho,
                                            auto_reg_order,
                                            auto_seg_sigma,
                                            timepoints,
                                            ):

    """Generate the autoregression noise

    Parameters
    ----------

    auto_reg_sigma : float


    auto_reg_order : float


    auto_reg_rho : float

    timepoints : 1 Dimensional array
        What time points are sampled by a TR

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
            noise_autoregression.append(np.random.normal(0, auto_seg_sigma))

        else:

            temp = []
            for pCounter in list(range(1, auto_reg_order + 1)):
                if tr_counter - pCounter >= 0:
                    past_trs = noise_autoregression[int(tr_counter - pCounter)]
                    past_reg = auto_reg_rho[pCounter - 1]
                    random = np.random.normal(0, auto_seg_sigma)
                    temp.append(past_trs * past_reg + random)

                    noise_autoregression.append(np.mean(temp))

    return noise_autoregression


def _generate_noise_temporal_phys(timepoints,
                                  physiological_sigma,
                                  resp_freq=0.2,
                                  heart_freq=1.17,
                                  ):
    """Generate the physiological noise.

    Parameters
    ----------

    timepoints : 1 Dimensional array
        What time points are sampled by a TR

    physiological_sigma : float
        How variable is the signal as a result of physiology,
        like heart beat and breathing

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
                          np.random.normal(0, physiological_sigma))

    return noise_phys


def _generate_noise_temporal(stimfunction,
                             tr_duration,
                             z_score=1,
                             motion_sigma=15,
                             drift_sigma=15,
                             auto_seg_sigma=15,
                             auto_reg_order=1,
                             auto_reg_rho=[1],
                             physiological_sigma=15,
                             ):
    """Generate the signal dependent noise

    This noise depends on things like the signal or the timing of the
    experiment.

    Parameters
    ----------

    stimfunction : 1 Dimensional array
        This is the timecourse of the stimuli in this experiment

    tr_duration : int
        How long is a TR, in seconds

    z_score : bool
        Is the data to be normalized

    motion_sigma : float

        How much noise is left over after pre-processing has been
        done. This is noise specifically on the task events

    drift_sigma : float

        What is the sigma on the distribution that coefficients are
        randomly sampled from

    auto_reg_sigma : float


    auto_reg_order : float


    auto_reg_rho : float


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
    trs = len(stimfunction)

    # What time points are sampled by a TR?
    timepoints = list(range(0, trs * tr_duration))[::tr_duration]

    # Make each noise type
    noise_task = _generate_noise_temporal_task(stimfunction,
                                               motion_sigma,
                                               )

    noise_drift = _generate_noise_temporal_drift(trs,
                                                 tr_duration,
                                                 drift_sigma,
                                                 timepoints,
                                                 )

    noise_phys = _generate_noise_temporal_phys(timepoints,
                                               physiological_sigma,
                                               )

    noise_autoregression = _generate_noise_temporal_autoregression(
        auto_reg_rho, auto_reg_order, auto_seg_sigma, timepoints, )

    # Do you want to z score it?
    if z_score == 1:
        noise_task = stats.zscore(noise_task)
        noise_phys = stats.zscore(noise_phys)
        noise_drift = stats.zscore(noise_drift)
        noise_autoregression = stats.zscore(noise_autoregression)

    # add the noise (it would have been nice to just add all of them in a
    # single line but this was causing formatting problems)
    noise_temporal = noise_task
    noise_temporal = noise_temporal + noise_phys
    noise_temporal = noise_temporal + noise_drift
    noise_temporal = noise_temporal + noise_autoregression

    return noise_temporal


def _generate_noise_spatial(dimensions,
                            sigma=-4.0,
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
    dimensions : tuple
        What is the shape of the volume to be generated

    sigma : float
        What is the size of the standard deviation in the gaussian random
        fields to generated

    Returns
    ----------

    multidimensional array, float
        Generates the spatial noise volume for these parameters
    """

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
        return np.sqrt((np.sqrt(np.sum(idxs ** 2))) ** sigma)

    noise = np.fft.fft2(np.random.normal(size=dimensions))
    amplitude = np.zeros(dimensions)

    for x, fft_x in enumerate(fftIndgen(dimensions[0])):
        for y, fft_y in enumerate(fftIndgen(dimensions[1])):
            for z, fft_z in enumerate(fftIndgen(dimensions[2])):
                amplitude[x, y, z] = Pk2(np.array([fft_x, fft_y, fft_z]))

    # The output
    noise_spatial = np.fft.ifft2(noise * amplitude)

    return noise_spatial.real


def generate_noise(dimensions,
                   stimfunction,
                   tr_duration,
                   noise_strength=[1]):
    """ Generate the noise to be added to the signal

    Parameters
    ----------
    dimensions : tuple
        What is the shape of the volume to be generated

    stimfunction :  Iterable, bool
        When do the stimuli events occur

    tr_duration : float
        What is the duration, in seconds, of each TR?

    noise_strength : list, float
        Either a one element list or a list of length 3 which varies
        the size of the noise for temporal noise, spatial noise and
        system noise

    Returns
    ----------

    multidimensional array, float
        Generates the noise volume for these parameters


    """

    # Duplicate the noise strength if only one is supplied
    if len(noise_strength) == 1:
        noise_strength = noise_strength * 3

    # What are the dimensions of the volume, including time
    dimensions_tr = (dimensions[0],
                     dimensions[1],
                     dimensions[2],
                     len(stimfunction))

    # Generate the noise
    noise_temporal = _generate_noise_temporal(stimfunction=stimfunction,
                                              tr_duration=tr_duration,
                                              ) * noise_strength[0]

    noise_spatial = _generate_noise_spatial(dimensions=dimensions,
                                            ) * noise_strength[1]

    noise_system = _generate_noise_system(dimensions=dimensions_tr,
                                          ) * noise_strength[2]

    # Find the outer product and add white noise
    noise = np.multiply.outer(noise_spatial, noise_temporal) + noise_system

    return noise


def mask_brain(volume,
               mask_name=None):
    """ Mask the simulated volume

    Parameters
    ----------

    volume : multidimensional array
        The volume that has been simulated

    mask_name : str
        What is the path to the mask to be loaded?

    Returns
    ----------
    brain : multidimensional array, float
        The masked brain
    """

    # Load in the mask
    if mask_name is None:
        mask_raw = np.load(resource_stream(__name__, "grey_matter_mask.npy"))
    else:
        mask_raw = np.load(mask_name)

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
                   1)

    # Scale the mask according to the input brain
    # You might get a warning but ignore it
    mask = ndimage.zoom(mask_raw, zoom_factor, order=2)

    # Anything that is in the bottom 1% (presumably the exterior of the
    # brain, make zero)
    mask[mask < 0.01] = 0

    # Mask out the non grey matter regions
    brain = volume * mask

    return brain


def plot_brain(fig,
               brain,
               percentile=99,
               ):
    """ Display the brain that has been generated with a given threshold

    Parameters
    ----------

    fig : matplotlib object
        The figure to be displayed, generated from matplotlib. import
        matplotlib.pyplot as plt; fig = plt.figure()

    brain : multidimensional array
        This is a 3d or 4d array with the neural data

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

    brain_all = np.where(np.abs(brain) > 0)

    # Clear the way
    ax.clear()

    ax.set_xlim(0, brain.shape[0])
    ax.set_ylim(0, brain.shape[1])
    ax.set_zlim(0, brain.shape[2])

    ax.scatter(brain_all[0],
               brain_all[1],
               brain_all[2],
               zdir='z',
               c='black',
               s=10,
               alpha=0.01)

    ax.scatter(brain_threshold[0],
               brain_threshold[1],
               brain_threshold[2],
               zdir='z',
               c='red',
               s=20)

    return ax
