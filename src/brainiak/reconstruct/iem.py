#  Copyright 2018 David Huberdeau & Peter Kok
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
#
# Copyright 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inverted Encoding Model (IEM)

    Method to decode and reconstruct features from data.

    The implementation is roughly based on the following publications:

    [Kok2013] "1.Kok, P., Brouwer, G. J., Gerven, M. A. J. van &
    Lange, F. P. de. Prior Expectations Bias Sensory Representations
    in Visual Cortex. J. Neurosci. 33, 16275–16284 (2013).

    [Brouwer2011] "2.Brouwer, G. J. & Heeger, D. J. Cross-orientation
    suppression in human visual cortex. J. Neurophysiol. 106(5):
    2108-2119 (2011).

    [Brouwer2009] "3.Brouwer, G. J. & Heeger, D. J.
    Decoding and Reconstructing Color from Responses in Human Visual
    Cortex. J. Neurosci. 29, 13992–14003 (2009).

    This implementation uses a set of sinusoidal
    basis functions to represent the set of possible feature values.
    A feature value is some characteristic of a stimulus, e.g. the
    angular location of a target along a horizontal line. This code was
    written to give some flexibility compared to the specific instances
    in Kok, 2013 & in Brouwer, 2009. Users can set the number of basis
    functions, or channels, and the range of possible feature values.

    There are separate classes for reconstructing feature values in a
    1-dimensional (1D) space or in a 2-dimensional (2D) space.
"""

# Authors: David Huberdeau (Yale University) &
# Peter Kok (Yale University), 2018 &
# Vy Vo (Intel Corp., UCSD), 2019

import logging
import warnings
import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from ..utils.utils import circ_dist

__all__ = ["InvertedEncoding1D",
           "InvertedEncoding2D"]

logger = logging.getLogger(__name__)
MAX_CONDITION_CHECK = 9000


class InvertedEncoding1D(BaseEstimator):
    """Basis function-based reconstruction method

    Inverted encoding models (alternatively known as forward models) are used
    to reconstruct a feature represented in some N-dimensional space, here 1D,
    (e.g. color of a stimulus) from patterns across voxels in functional data.
    The model uses n_channels number of idealized basis functions and assumes
    that the transformation from stimulus feature (e.g. color) to basis
    function is one- to-one and invertible. The response of a voxel is
    expressed as the weighted sum of basis functions. In this implementation,
    basis functions were half-wave rectified sinusoid functions raised to a
    power set by the user (e.g. 6).

    The model:
    Inverted encoding models reconstruct a stimulus feature from
    patterns of BOLD activity by relating the activity in each
    voxel, B, to the values of hypothetical channels (or basis
    functions), C, according to Equation 1 below.

    (1)     B = W*C

    where W is a weight matrix that represents the relationship
    between BOLD activity and Channels. W must be estimated from
    training data; this implementation (and most described in the
    literature) uses linear regression to estimate W as in Equation
    2 below [note: inv() represents matrix inverse or
    pseudo-inverse].

    (2)     W_est = B_train*inv(C_train)

    The weights in W_est (short for "estimated") represent the
    contributions of each channel to the response of each voxel.
    Estimated channel responses can be computed given W_est and
    new voxel activity represented in matrix B_exp (short for
    "experiment") through inversion of Equation 1:

    (3)     C_est = inv(W_est)*B_exp

    Given estimated channel responses, C_est, it is straightforward
    to obtain the reconstructed feature value by summing over
    channels multiplied by their channel responses and taking the
    argmax (i.e. the feature associated with the maximum value).

    Using this model:
    Use fit() to estimate the weights of the basis functions given
    input data (e.g. beta values from fMRI data). This function
    will execute equation 2 above.

    Use predict() to compute predicted stimulus values
    from new functional data. This function computes estimated
    channel responses, as in equation 3, then computes summed
    channel output and finds the argmax (within the stimulus
    feature space) associated with those responses.

    Use score() to compute a measure of the error of the prediction
    based on known stimuli.

    This implementation assumes a circular (or half-
    circular) feature domain. Future implementations might
    generalize the feature input space, and increase the
    possible dimensionality.

    Parameters
    ----------
    n_channels: int, default 5. Number of channels
        The number of channels, or basis functions, to be used in
        the inverted encoding model.

    channel_exp: int, default 6. Basis function exponent.
        The exponent of the sinuoidal basis functions, which
        establishes the width of the functions.

    stimulus_mode: str, default 'halfcircular' (other option is
        'circular'). Describes the feature domain.

    range_start: double, default 0. Lowest value of domain.
        Beginning value of range of independent variable
        (usually degrees).

    range_stop: double, default 180. Highest value of domain.
        Ending value of range of independent variable
        (usually degrees).

    channel_density: int, default 180. Number of points in the
        feature domain.

    stimulus_resolution: double, default None will set the stimulus
        resolution to be identical to the channel density. This sets
        the resolution at which the stimuli were presented (e.g. a
        spatial position with some width has a lower stimulus
        resolution).

    Attributes
    ----------
    channels_: [n_channels, channel density] NumPy 2D array
        matrix defining channel values

    W_: sklearn.linear_model model containing weight matrix that
        relates estimated channel responses to response amplitude
        data

    See get_params() for the rest of the attributes.
    """

    def __init__(self, n_channels=6, channel_exp=5,
                 stimulus_mode='halfcircular', range_start=0., range_stop=180.,
                 channel_density=180, stimulus_resolution=None):
        self.n_channels = n_channels
        self.channel_exp = channel_exp
        self.stimulus_mode = stimulus_mode
        self.range_start = range_start
        self.range_stop = range_stop
        self.channel_density = channel_density
        self.channel_domain = np.linspace(range_start, range_stop - 1,
                                          channel_density)
        if stimulus_resolution is None:
            self.stim_res = channel_density
        else:
            self.stim_res = stimulus_resolution
        self._check_params()

    def _check_params(self):
        if self.range_start >= self.range_stop:
            raise ValueError("range_start {} must be less than "
                             "{} range_stop.".format(self.range_start,
                                                     self.range_stop))
        if self.stimulus_mode == 'halfcircular':
            if (self.range_stop - self.range_start) != 180.:
                raise ValueError("For half-circular feature spaces,"
                                 "the range must be 180 degrees, "
                                 "not {}".
                                 format(self.range_stop - self.range_start))
        elif self.stimulus_mode == 'circular':
            if (self.range_stop - self.range_start) != 360.:
                raise ValueError("For circular feature spaces, the"
                                 " range must be 360 degrees"
                                 "not {}".
                                 format(self.range_stop - self.range_start))
        if self.n_channels < 2:
            raise ValueError("Insufficient number of channels.")
        if not np.isin(self.stimulus_mode, ['circular', 'halfcircular']):
            raise ValueError("Stimulus mode must be one of these: "
                             "'circular', 'halfcircular'")

    def fit(self, X, y):
        """Use data and feature variable labels to fit an IEM

        Parameters
        ----------
        X: numpy matrix of voxel activation data. [observations, voxels]
            Should contain the beta values for each observation or
            trial and each voxel of training data.
        y: numpy array of response variable. [observations]
            Should contain the feature for each observation in X.
        """
        # Check that data matrix is well conditioned:
        if np.linalg.cond(X) > MAX_CONDITION_CHECK:
            logger.error("Data is singular.")
            raise ValueError("Data matrix is nearly singular.")
        if X.shape[0] < self.n_channels:
            logger.error("Not enough observations. Cannot calculate "
                         "pseudoinverse.")
            raise ValueError("Fewer observations (trials) than "
                             "channels. Cannot compute pseudoinverse.")
        # Check that the data matrix is the right size
        shape_data = np.shape(X)
        shape_labels = np.shape(y)
        if len(shape_data) != 2:
            raise ValueError("Data matrix has too many or too few "
                             "dimensions.")
        else:
            if shape_data[0] != shape_labels[0]:
                raise ValueError("Mismatched data samples and label samples")

        # Define the channels (or basis set)
        self.channels_, channel_centers = self._define_channels()
        logger.info("Defined channels centered at {} degrees.".format(
            np.rad2deg(channel_centers)))
        # Create a matrix of channel activations for every observation.
        # (i.e., C1 in Brouwer & Heeger 2009.)
        C = self._define_trial_activations(y)
        # Solve for W in B = WC
        self.W_ = X.transpose() @ np.linalg.pinv(C.transpose())
        if np.linalg.cond(self.W_) > MAX_CONDITION_CHECK:
            logger.error("Weight matrix is nearly singular.")
            raise ValueError("Weight matrix is nearly singular.")

        return self

    def predict(self, X):
        """Use test data to predict the feature

        Parameters
        ----------
            X: numpy matrix of voxel activation from test trials
            [observations, voxels]. Used to predict feature
            associated with the given observation.

        Returns
        -------
            model_prediction: numpy array of estimated feature values.
        """
        # Check that the data matrix is the right size
        shape_data = np.shape(X)
        if len(shape_data) != 2:
            raise ValueError("Data matrix has too many or too few "
                             "dimensions.")

        model_prediction = self._predict_features(X)

        return model_prediction

    def score(self, X, y):
        """Calculate error measure of prediction. Default measurement
        is R^2, the coefficient of determination.

        Parameters
        ----------
            X: numpy matrix of voxel activation from new data
                [observations,voxels]
            y: numpy array of responses. [observations]

        Returns
        -------
            score_value: the error measurement between the actual
                feature and predicted features.
        """
        pred_features = self.predict(X)
        if self.stimulus_mode == 'halfcircular':
            # multiply features by 2. otherwise doesn't wrap properly
            pred_features = pred_features * 2
            y = y * 2

        ssres = (circ_dist(np.deg2rad(y),
                           np.deg2rad(pred_features)) ** 2).sum()
        sstot = (circ_dist(np.deg2rad(y),
                           np.ones(y.size) * scipy.stats.circmean(
                               np.deg2rad(y))) ** 2).sum()
        score_value = (1 - ssres / sstot)

        return score_value

    def get_params(self, deep: bool = True):
        """Returns model parameters.

        Returns
        -------
        params: parameter of this object
        """
        return {"n_channels": self.n_channels, "channel_exp": self.channel_exp,
                "stimulus_mode": self.stimulus_mode,
                "range_start": self.range_start, "range_stop": self.range_stop,
                "channel_domain": self.channel_domain,
                "stim_res": self.stim_res}

    def set_params(self, **parameters):
        """Sets model parameters after initialization.

        Parameters
        ----------
            parameters: structure with parameters and change values
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        setattr(self, "channel_domain",
                np.linspace(self.range_start, self.range_stop - 1,
                            self.channel_density))
        self._check_params()
        return self

    def _define_channels(self):
        """Define basis functions (aka channels).

        Returns
        -------
            channels: numpy matrix of basis functions. dimensions are
                [n_channels, function resolution].
            channel_centers: numpy array of the centers of each channel
        """
        channel_centers = np.linspace(np.deg2rad(self.range_start),
                                      np.deg2rad(self.range_stop),
                                      self.n_channels + 1)
        channel_centers = channel_centers[0:-1]
        # make sure channels are not bimodal if using 360 deg space
        if self.stimulus_mode == 'circular':
            domain = self.channel_domain * 0.5
            centers = channel_centers * 0.5
        elif self.stimulus_mode == 'halfcircular':
            domain = self.channel_domain
            centers = channel_centers

        # define exponentiated function
        channels = np.asarray(
            [np.cos(np.deg2rad(domain) - cx) ** self.channel_exp for cx in
             centers])
        # half-wave rectification preserving circularity
        channels = abs(channels)

        return channels, channel_centers

    def _define_trial_activations(self, stimuli):
        """Defines a numpy matrix of predicted channel responses for
        each trial/observation.

        Parameters

            stimuli: numpy array of the feature values for each
                observation

        Returns
        -------
            C: matrix of predicted channel responses. dimensions are
                number of observations by stimulus resolution
        """
        stim_axis = np.linspace(self.range_start, self.range_stop - 1,
                                self.stim_res)
        if self.range_start > 0:
            stimuli = stimuli + self.range_start
        elif self.range_start < 0:
            stimuli = stimuli - self.range_start
        one_hot = np.eye(self.stim_res)
        indices = [np.argmin(abs(stim_axis - x)) for x in stimuli]
        stimulus_mask = one_hot[indices, :]
        if self.channel_density != self.stim_res:
            if self.channel_density % self.stim_res == 0:
                stimulus_mask = np.repeat(stimulus_mask,
                                          self.channel_density / self.stim_res)
            else:
                raise NotImplementedError("This code doesn't currently support"
                                          " stimuli which are not square "
                                          "functions in the feature domain, or"
                                          " stimulus widths that are not even"
                                          "divisors of the number of points in"
                                          " the feature domain.")

        C = stimulus_mask @ self.channels_.transpose()
        # Check that C is full rank
        if np.linalg.matrix_rank(C) < self.n_channels:
            warnings.warn("Stimulus matrix is {}, not full rank. May cause "
                          "issues with stimulus prediction/reconstruction.".
                          format(np.linalg.matrix_rank(C)), RuntimeWarning)
        return C

    def _predict_channel_responses(self, X):
        """Computes predicted channel responses from data
        (e.g. C2 in Brouwer & Heeger 2009)

        Parameters
        ----------
            X: numpy data matrix. [observations, voxels]

        Returns
        -------
            channel_response: numpy matrix of channel responses
        """
        channel_response = np.matmul(np.linalg.pinv(self.W_), X.transpose())
        return channel_response

    def _predict_feature_responses(self, X):
        """Takes channel weights and transforms them into continuous
        functions defined in the feature domain.

        Parameters
         ---------
            X: numpy matrix of data. [observations, voxels]

        Returns
        -------
            pred_response: predict response from all channels. Used
                to predict feature (e.g. direction).
        """
        pred_response = np.matmul(self.channels_.transpose(),
                                  self._predict_channel_responses(X))
        return pred_response

    def _predict_features(self, X):
        """Predicts feature value (e.g. direction) from data in X.
        Takes the maximum of the 'reconstructed' or predicted response
        function.

        Parameters
         ---------
            X: numpy matrix of data. [observations, voxels]

        Returns
        -------
            pred_features: predicted feature from response across all
                channels.
        """
        pred_response = self._predict_feature_responses(X)
        feature_ind = np.argmax(pred_response, 0)
        pred_features = self.channel_domain[feature_ind]

        return pred_features


class InvertedEncoding2D(BaseEstimator):
    """Basis function-based reconstruction method

    Inverted encoding models (alternatively known as forward models) are used
    to reconstruct a feature represented in a N-dimensional space, here 2D,
    (e.g. position on a projector screen) from patterns across voxels in
    functional data. The model uses some number of idealized basis functions
    that cover the 2D space, and assumes that the transformation from
    stimulus feature (e.g. 2D spatial position) to basis function is one-
    to-one and invertible. The response of a voxel is expressed as the
    weighted sum of basis functions. In this implementation, basis functions
    were half-wave rectified sinusoid functions raised to some power (set by
    the user).

    The documentation will refer to the 'stimulus space' or 'stimulus domain',
    which should be a 2D space in consistent units (e.g. screen pixels,
    or degrees visual angle). The stimulus space is the domain in which the
    stimulus is reconstructed. We will refer to the each point in this 2D
    stimulus domain as a 'pixel'.

    The model:
    Inverted encoding models reconstruct a stimulus feature from
    patterns of BOLD activity by relating the activity in each
    voxel, B, to the values of hypothetical channels (or basis
    functions), C, according to Equation 1 below.

    (1)     B = W*C

    where W is a weight matrix that represents the relationship
    between BOLD activity and Channels. W must be estimated from
    training data; this implementation (and most described in the
    literature) uses linear regression to estimate W as in Equation
    2 below [note: inv() represents matrix inverse or
    pseudo-inverse].

    (2)     W_est = B_train*inv(C_train)

    The weights in W_est (short for "estimated") represent the
    contributions of each channel to the response of each voxel.
    Estimated channel responses can be computed given W_est and
    new voxel activity represented in matrix B_exp (short for
    "experiment") through inversion of Equation 1:

    (3)     C_est = inv(W_est)*B_exp

    Given estimated channel responses, C_est, it is straightforward
    to obtain the reconstructed feature value by summing over
    channels multiplied by their channel responses and taking the
    argmax (i.e. the feature associated with the maximum value).

    Using this model:
    Use fit() to estimate the weights of the basis functions given
    input data (e.g. beta values from fMRI data). This function
    will execute equation 2 above.

    Use predict() to compute predicted stimulus values
    from new functional data. This function computes estimated
    channel responses, as in equation 3, then computes summed
    channel output and finds the argmax (within the stimulus
    feature space) associated with those responses.

    Use score() to compute a measure of the error of the prediction
    based on known stimuli.

    Parameters
    ----------
    stim_xlim: list of 2 floats Specifies the minimum and maximum x-values
        of the area to be reconstructed. In order to be estimated properly, a
        stimulus must appear at these limits. Specifying limits outside the
        range of the stimuli can lead to spurious estimates.

    stim_ylim: list of 2 floats Specifies the minimum and maximum y-values
        of the area to be reconstructed. In order to be estimated properly, a
        stimulus must appear at these limits. Specifying limits outside the
        range of the stimuli can lead to spurious estimates.

    stimulus_resolution: float or list of 2 floats. If a single float is
        given, it will be expanded to a list (i.e. we will assume that the
        reconstructed area is composed of square pixels).

    stim_radius: float, or sequence of floats [n_stim], default None. If the
        user does not define the design matrix of the encoding model (e.g. C
        in B = W*C), it will be defined automatically on the assumption that
        each observation was for a 2D circular stimulus of some radius.

    chan_xlim: list of 2 floats, default None. Specifies the minimum and
        maximum x-values of the channels, or basis functions.

    chan_ylim: list of 2 floats, default None. Specifies the minimum and
        maximum y-values of the channels, or basis functions.

    channels: [n_channels, n_pixels] NumPy 2D array, default None. If None at
        initialization, it can be defined with
        either define_basis_functions_sqgrid() or
        define_basis_functions_trigrid(), each of which tiles the given 2D
        space with some grid (square or triangular/hexagonal, respectively).
        Alternatively, the user can specify their own channels.

    channel_exp: int, default 7. Basis function exponent. The exponent of the
        sinuoidal basis functions, which helps control their width.

    Attributes
    ----------
    channels: [n_channels, n_pixels] NumPy 2D array defining channels

    W_: sklearn.linear_model containing weight matrix that relates estimated
        channel responses to response data

    See get_params() for the rest of the attributes.
    """

    def __init__(self, stim_xlim, stim_ylim, stimulus_resolution,
                 stim_radius=None, chan_xlim=None, chan_ylim=None,
                 channels=None, channel_exp=7):
        """Defines a 2D inverted encoding model object.

        While the parameters defining the domain in which to reconstruct
        the stimuli are required (e.g. all `stim_*` inputs), the parameters
        to define the channels (`chan*`) are optional, in case the user
        wishes to define their own channels (a.k.a basis functions).


        Parameters
        ----------
        stim_xlim: sequence of 2 float values, specifying the lower & upper
            limits on the horizontal axis, respectively.
        stim_ylim: sequence of 2 float values, specifying the lower & upper
            limits on the vertical axis, respectively.
        stimulus_resolution: a float or sequence of 2 floats, specifying the
            number of pixels that exist in the x- and y- directions.
        stim_radius: float, default None. The radius in pixels, assuming that
            the stimulus is circular. If None, the user must either define it
            before running fit(), or pass in a custom C in B = W*C.
        chan_xlim: sequence of 2 float values, default None. Specifies the
            lower & upper limits of the channels in the horizontal axis. If
            None, the user must define this before using the class functions
            to create basis functions, or pass in custom-defined channels.
        chan_ylim: sequence of 2 float values, default None. Specifies the
            lower & upper limits of the channels in the vertical axis. If
            None, the user must define this before using the class functions
            to create basis functions, or pass in custom-defined channels.
        channel_exp: float or int, default None. The exponent for a
            sinusoidal basis function. If None, it must be set before the
            channels or defined, or pass in custom-defined channels.

        """
        # Automatically expand stimulus_resolution if only one value is given.
        # This will create a square field  of view (FOV) for the
        # reconstruction.
        if not isinstance(stimulus_resolution, list):  # make FOV square
            stimulus_resolution = [stimulus_resolution, stimulus_resolution]
        if (len(stim_xlim) != 2) or (len(stim_ylim) != 2):
            raise ValueError("Stimulus limits should be a sequence, 2 values")
        self.stim_fov = [stim_xlim, stim_ylim]
        self.stim_pixels = [np.linspace(stim_xlim[0], stim_xlim[1],
                                        stimulus_resolution[0]),
                            np.linspace(stim_ylim[0], stim_ylim[1],
                                        stimulus_resolution[1])]
        self.xp, self.yp = np.meshgrid(self.stim_pixels[0],
                                       self.stim_pixels[1])
        self.stim_radius_px = stim_radius
        self.channels = channels
        if self.channels is None:
            self.n_channels = None
        else:
            self.n_channels = self.channels.shape[0]
        if chan_xlim is None:
            chan_xlim = stim_xlim
            logger.info("Set channel x-limits to stimulus x-limits", stim_xlim)
        if chan_ylim is None:
            chan_ylim = stim_ylim
            logger.info("Set channel y-limits to stimulus y-limits", stim_ylim)
        self.channel_limits = [chan_xlim, chan_ylim]
        self.channel_exp = channel_exp
        self._check_params()

    def _check_params(self):
        if len(self.stim_fov) != 2:
            raise ValueError("Stim FOV needs to have an x-list and a y-list")
        elif len(self.stim_fov[0]) != 2 or len(self.stim_fov[1]) != 2:
            raise ValueError("Stimulus limits should be a sequence, 2 values")
        else:
            if (self.stim_fov[0][0] >= self.stim_fov[0][1]) or \
                    (self.stim_fov[1][0] >= self.stim_fov[1][1]):
                raise ValueError("Stimulus x or y limits should be ascending "
                                 "values")
        if self.xp.size != self.yp.size:
            raise ValueError("xpixel grid and ypixel grid do not have same "
                             "number of elements")
        if self.n_channels and np.all(self.channels):
            if self.n_channels != self.channels.shape[0]:
                raise ValueError("Number of channels {} does not match the "
                                 "defined channels: {}".
                                 format(self.n_channels,
                                        self.channels.shape[0]))
            if self.channels.shape[1] != self.xp.size:
                raise ValueError("Defined {} channels over {} pixels, but "
                                 "stimuli are represented over {} pixels. "
                                 "Pixels should match.".
                                 format(self.n_channels,
                                        self.channels.shape[1],
                                        self.xp.size))

    def fit(self, X, y, C=None):
        """Use data and feature variable labels to fit an IEM

        Parameters
        ----------
        X: numpy matrix of voxel activation data. [observations, voxels]
            Should contain the beta values for each observation or
            trial and each voxel of training data.
        y: numpy array of response variable. [observations]
            Should contain the feature for each observation in X.
        C: numpy matrix of channel activations for every observation (e.g.
            the design matrix C in the linear equation B = W*C), matrix size
            [observations, pixels]. If None (default), this assumes that each
            observation contains a 2D circular stimulus and will define the
            activations with self._define_trial_activations(y).
        """
        # Check that data matrix is well conditioned:
        if np.linalg.cond(X) > MAX_CONDITION_CHECK:
            logger.error("Data is singular.")
            raise ValueError("Data matrix is nearly singular.")
        if self.channels is None:
            raise ValueError("Must define channels (set of basis functions).")
        if X.shape[0] < self.n_channels:
            logger.error("Not enough observations. Cannot calculate "
                         "pseudoinverse.")
            raise ValueError("Fewer observations (trials) than "
                             "channels. Cannot compute pseudoinverse.")
        # Check that the data matrix is the right size
        shape_data = np.shape(X)
        shape_labels = np.shape(y)
        if shape_data[0] != shape_labels[0]:
            raise ValueError("Mismatched data samples and label samples")
        if C is None:
            # Create a matrix of channel activations for every observation.
            # (i.e., C1 in Brouwer & Heeger 2009.)
            C = self._define_trial_activations(y)
        # Solve for W in B = WC
        self.W_ = X.transpose() @ np.linalg.pinv(C.transpose())
        if np.linalg.cond(self.W_) > MAX_CONDITION_CHECK:
            logger.error("Weight matrix is nearly singular.")
            raise ValueError("Weight matrix is nearly singular.")

        return self

    def predict(self, X):
        """Use test data to predict the feature

        Parameters
        ----------
        X: numpy matrix of voxel activation from test trials [observations,
            voxels]. Used to predict feature associated with the given
            observation.

        Returns
        -------
        model_prediction: numpy array of estimated feature values.
        """
        # Check that the data matrix is the right size
        shape_data = np.shape(X)
        if len(shape_data) != 2:
            raise ValueError("Data matrix has too many or too few "
                             "dimensions.")

        model_prediction = self._predict_features(X)

        return model_prediction

    def score(self, X, y):
        """Calculate error measure of prediction, assuming that the predicted
        feature is at the maximum of the reconstructed values.

        To score the reconstructions against expected features defined in the
        stimulus domain (i.e. in pixels), see score_against_reconstructed().

        Parameters
        ----------
        X: numpy matrix of voxel activation from new data
            [observations,voxels]
        y: numpy array of stimulus features. [observations, 2]

        Returns
        -------
        score_value: the error measurement between the actual
            feature and predicted features, [observations].
        """
        pred_features = self.predict(X)
        ssres = np.sum((pred_features - y) ** 2, axis=1)
        sstot = np.sum((y - np.mean(y)) ** 2, axis=1)
        score_value = 1 - (ssres / sstot)

        return score_value

    def score_against_reconstructed(self, X, y, metric="euclidean"):
        """Calculates a distance metric between reconstructed features in
        the 2D stimulus domain (i.e. reconstructions in pixels) given
        some observations X, and expected features y. Expected features must
        also be in the pixel stimulus domain.

        To score the reconstructions against the expected maxima, see score().

        Parameters
        ----------
        X: numpy matrix of voxel activation from new data
            [observations, voxels]
        y: numpy array of the expected stimulus reconstruction values [pixels,
            observations].
        metric: string specifying the distance metric, either "euclidean" or
            "cosine".

        Returns
        -------
        score_value: the error measurement between the reconstructed feature
            values as the expected values, [observations].
        """
        yhat = self.predict_feature_responses(X)
        if metric == "euclidean":
            score_value = euclidean_distances(y.T, yhat.T)
        elif metric == "cosine":
            score_value = cosine_distances(y.T, yhat.T)
        return score_value[0, :]

    def get_params(self, deep: bool = True):
        """Returns model parameters.

        Returns
        -------
        params: parameter of this object
        """
        return {"n_channels": self.n_channels, "channel_exp": self.channel_exp,
                "stim_fov": self.stim_fov, "stim_pixels": self.stim_pixels,
                "stim_radius_px": self.stim_radius_px, "xp": self.xp,
                "yp": self.yp, "channels": self.channels, "channel_limits":
                    self.channel_limits}

    def set_params(self, **parameters):
        """Sets model parameters after initialization.

        Parameters
        ----------
            parameters: structure with parameters and change values
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self._check_params()
        return self

    def _make_2d_cosine(self, x, y, x_center, y_center, s):
        """Defines a 2D exponentiated cosine (isometric, e.g. constant width
        in x & y) for use as a basis function. Function goes to zero at the
        given size constant s. That is, the function is given by if r <= s:
        f(r) = (0.5 + 0.5*cos(r*pi/s)))**channel_exp else:       0 where r is
        the Euclidean distance from the center of the function. This will
        yield a Gaussian-like function, centered at (x_center, y_center).

        Parameters
        ----------
        x: x-coordinates of the stimulus space, [npixels, 1] matrix
        y: y-coordinates of the stimulus space, [npixels, 1] matrix
        x_center: x-coordinate of basis function centers (sequence, nchannels)
        y_center: y-coordinate of basis function centers (sequence, nchannels)
        s: size constant of the 2D cosine function. This is the radius where
            the function is non-zero.

        Returns
        -------
        cos_functions: basis functions defined in the 2D stimulus space.
            returns a [nchannels, npixels] matrix.
        """
        cos_functions = np.zeros((len(x_center), len(x)))
        for i in range(len(x_center)):
            myr = np.sqrt((x - x_center[i]) ** 2 + (y - y_center[i]) ** 2). \
                squeeze()
            qq = (myr <= s) * 1
            zp = ((0.5 * (1 + np.cos(myr * np.pi / s))) ** self.channel_exp)
            cos_functions[i, :] = zp * qq
        return cos_functions

    def _2d_cosine_sz_to_fwhm(self, size_constant):
        fwhm = 2 * size_constant \
               * np.arccos((0.5 ** (1 / self.channel_exp) - 0.5) / 0.5) / np.pi
        return fwhm

    def _2d_cosine_fwhm_to_sz(self, fwhm):
        """For an exponentiated 2D cosine basis function, converts the
        full-width half-maximum (FWHM) of that function to the function's
        size constant. The size constant is the variable s in the function
        below:
            if r <= s:   f(r) = (0.5 + 0.5*cos(r*pi/s)))**channel_exp
            else:       0 where r is the Euclidean distance from the center of
        the function.

        Parameters
        ----------
        fwhm: a float indicating the full-width half-maximum in stimulus space

        Returns
        -------
        sz: the size constant of the exponentiated cosine
        """
        sz = (0.5 * np.pi * fwhm) / \
             (np.arccos((0.5 ** (1 / self.channel_exp) - 0.5) / 0.5))
        return sz

    def define_basis_functions_sqgrid(self, nchannels, channel_size=None):
        """Define basis functions (aka channels) arrange in a square grid.
        Sets the self.channels parameter.

        Parameters
        ----------
        nchannels: number of channels in the x (horizontal) direction
        channel_size: the desired full-width half-maximum (FWHM) of the
            channel, in stimulus space.

        Returns
        -------
        self.channels: defines channels, a [nchannels, npixels] matrix.
        channel_centers: numpy array of the centers of each channel, given as
            [nchannels x 2] matrix
        """
        # If given a single value for nchannels, expand to make a square
        if not isinstance(nchannels, list):
            nchannels = [nchannels, nchannels]
        chan_xcenters = np.linspace(self.channel_limits[0][0],
                                    self.channel_limits[0][1], nchannels[0])
        chan_ycenters = np.linspace(self.channel_limits[1][0],
                                    self.channel_limits[1][1], nchannels[1])
        cx, cy = np.meshgrid(chan_xcenters, chan_ycenters)
        cx = cx.reshape(-1, 1)
        cy = cy.reshape(-1, 1)
        if channel_size is None:
            # To get even coverage, setting the channel FWHM to ~1.1x-1.2x the
            # spacing between the channels might work. (See Sprague et al. 2013
            # Methods & Supplementary Figure 3 -- this is for cosine exp = 7,
            # your mileage may vary for other exponents!).
            channel_size = 1.2 * (chan_xcenters[1] - chan_xcenters[0])
        cos_width = self._2d_cosine_fwhm_to_sz(channel_size)
        # define exponentiated function
        self.channels = self._make_2d_cosine(self.xp.reshape(-1, 1),
                                             self.yp.reshape(-1, 1), cx, cy,
                                             cos_width)
        self.n_channels = self.channels.shape[0]

        return self.channels, np.hstack([cx, cy])

    def define_basis_functions_trigrid(self, grid_radius, channel_size=None):
        """Define basis functions (aka channels) arranged in a triangular grid.

        Returns
        -------
        self.channels: defines channels, [nchannels, npixels] matrix.
        channel_centers: numpy array of the centers of each channel
        """
        x_dist = np.diff(self.channel_limits[0]) / (grid_radius * 2)
        y_dist = x_dist * np.sqrt(3) * 0.5
        trigrid = np.zeros((0, 2))
        xbase = np.expand_dims(np.arange(self.channel_limits[0][0],
                                         self.channel_limits[0][1],
                                         x_dist.item()), 1)
        for yi, y in enumerate(np.arange(self.channel_limits[1][0],
                                         self.channel_limits[1][1],
                                         y_dist.item())):
            if (yi % 2) == 0:
                xx = xbase.copy()
                yy = np.ones((xx.size, 1)) * y
            else:
                xx = xbase.copy() + x_dist / 2
                yy = np.ones((xx.size, 1)) * y
            trigrid = np.vstack(
                (trigrid, np.hstack((xx, yy))))

        if channel_size is None:
            # To get even coverage, setting the channel FWHM to ~1.1x-1.2x the
            # spacing between the channels might work. (See Sprague et al. 2013
            # Methods & Supplementary Figure 3 -- this is for cosine exp = 7,
            # your mileage may vary for other exponents!).
            channel_size = 1.1 * x_dist
        cos_width = self._2d_cosine_fwhm_to_sz(channel_size)
        self.channels = self._make_2d_cosine(self.xp.reshape(-1, 1),
                                             self.yp.reshape(-1, 1),
                                             trigrid[:, 0],
                                             trigrid[:, 1], cos_width)
        self.n_channels = self.channels.shape[0]

        return self.channels, trigrid

    def _define_trial_activations(self, stim_centers, stim_radius=None):
        """Defines a numpy matrix of predicted channel responses for each
        trial/observation. Assumes that the presented stimulus is circular in
        the 2D stimulus space. This can effectively be a single circular
        pixel if stim_radius=0.5.

        Parameters
        -------
        stim_centers: numpy array of 2D stimulus features for each observation,
            expected dimensions are [observations, 2].
        stim_radius: scalar value or array-like specifying the radius of the
            circular stimulus for each observation, [observations]. While
            this can be read-out from the property self.stim_radius_px,
            here the user can specify it in case they are retraining the
            model with new observations.

        Returns
        -------
        C: numpy array of predicted channel responses [observations, pixels]
        """
        nstim = stim_centers.shape[0]
        if self.stim_radius_px is None:
            if stim_radius is None:
                raise ValueError("No defined stimulus radius. Please set.")
            else:
                self.stim_radius_px = stim_radius
        if not isinstance(self.stim_radius_px, np.ndarray) or not isinstance(
                self.stim_radius_px, list):
            self.stim_radius_px = np.ones(nstim) * self.stim_radius_px
        # Create a mask for every stimulus observation in the stimulus domain
        stimulus_mask = np.zeros((self.xp.size, nstim))
        for i in range(nstim):
            rad_vals = ((self.xp.reshape(-1, 1) - stim_centers[i, 0]) ** 2 +
                        (self.yp.reshape(-1, 1) - stim_centers[i, 1]) ** 2)
            inds = np.where(rad_vals < self.stim_radius_px[i])[0]
            stimulus_mask[inds, i] = 1
        # Go from the stimulus domain to the channel domain
        C = self.channels.squeeze() @ stimulus_mask
        C = C.transpose()
        # Check that C is full rank
        if np.linalg.matrix_rank(C) < self.n_channels:
            warnings.warn("Stimulus matrix is {}, not full rank. May cause "
                          "issues with stimulus prediction/reconstruction.".
                          format(np.linalg.matrix_rank(C)), RuntimeWarning)
        return C

    def _predict_channel_responses(self, X):
        """Computes predicted channel responses from data
        (e.g. C2 in Brouwer & Heeger 2009)

        Parameters
        ----------
        X: numpy data matrix. [observations, voxels]

        Returns
        -------
        channel_response: numpy matrix of channel responses. [channels,
            observations]
        """
        channel_response = np.matmul(np.linalg.pinv(self.W_), X.transpose())
        return channel_response

    def predict_feature_responses(self, X):
        """Takes channel weights and transforms them into continuous
        functions defined in the feature domain.

        Parameters
        ---------
        X: numpy matrix of data. [observations, voxels]

        Returns
        -------
        pred_response: predict response from all channels. This is the stimulus
            reconstruction in the channel domain. [pixels, observations]
        """
        pred_response = np.matmul(self.channels.transpose(),
                                  self._predict_channel_responses(X))
        return pred_response

    def _predict_features(self, X):
        """Predicts feature value from data in X.
        Takes the maximum of the reconstructed, i.e. predicted response
        function.

        Parameters
        ---------
        X: numpy matrix of data. [observations, voxels]

        Returns
        -------
        pred_features: numpy matrix of predicted stimulus features.
            [observations, 2]
        """
        pred_response = self.predict_feature_responses(X)
        feature_ind = np.argmax(pred_response, 0)
        pred_features = np.hstack((self.xp.reshape(-1, 1)[feature_ind],
                                   self.yp.reshape(-1, 1)[feature_ind]))

        return pred_features
