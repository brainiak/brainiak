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
"""

# Authors: David Huberdeau (Yale University) &
# Peter Kok (Yale University), 2018

import logging
import numpy as np
from sklearn import linear_model
from sklearn.base import BaseEstimator

__all__ = [
    "InvertedEncoding"
]

logger = logging.getLogger(__name__)


class InvertedEncoding(BaseEstimator):
    """Basis function-based reconstruction method

    Inverted encoding models (alternatively known as forward
    models) are used to reconstruct a feature, e.g. color of
    a stimulus, from patterns across voxels in functional
    data. The model uses n_channels number of idealized
    basis functions and assumes that the transformation from
    stimulus feature (e.g. color) to basis function is one-
    to-one and invertible. The response of a voxel is
    expressed as the weighted sum of basis functions.
    In this implementation, basis functions were half-wave
    rectified sinusoid functions raised to a power set by
    the user (e.g. 6).

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

    Given estimated channel responses, C_est, it is straighforward
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
        The number of channels, or basis functions, to be used
        in the inverted encoding model.

    channel_exp: int, default 6. Basis function exponent.
        The exponent of the sinuoidal basis functions, which
        establishes the width of the functions.

    range_start: double, default 0. Lowest value of domain.
        Beginning value of range of independent variable
        (usually degrees).

    range_stop: double, default 180. Highest value of domain.
        Ending value of range of independent variable
        (usually degrees).

    Attributes
    ----------
    C_: [n_channels, channel density] NumPy 2D array
        matrix defining channel values

    C_D_: [n_channels, channel density] NumPy 2D array
        matrix defining channel independent variable (usually
        in degrees)

    W_: sklearn.linear_model model containing weight matrix that
        relates estimated channel responses to response amplitude
        data
    """
    def __init__(self,
                 n_channels=5,
                 channel_exp=6,
                 range_start=0,
                 range_stop=180):

        # Check that range_start is less than range_stop
        if range_start >= range_stop:
            raise ValueError("range_start must be less than range_stop.")
        self.n_channels = n_channels  # default = 5
        self.channel_exp = channel_exp  # default = 6
        self.range_start = range_start  # in degrees, def=0
        self.range_stop = range_stop  # in degrees, def=180

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
        # Check that there are channels specified
        if self.n_channels < 2:
            raise ValueError("Insufficient channels.")

        # Check that data matrix is well conditioned:
        if np.linalg.cond(X) > 9000:
            logger.error("Data is singular.")
            raise ValueError("Data matrix is nearly singular.")

        # Check that the data matrix is the right size
        shape_data = np.shape(X)
        shape_labels = np.shape(y)
        if len(shape_data) != 2:
            raise ValueError("Data matrix has too many or too few "
                             "dimensions.")
        else:
            if shape_data[0] != shape_labels[0]:
                raise ValueError(
                    "Mismatched data samples and label samples")

        self.C_, self.C_D_ = self._define_channels()
        n_train = len(y)
        # Create a matrix of channel activations for every observation.
        # This is the C1 matrix in Brouwer, 2009.
        F = np.empty((n_train, self.n_channels))
        for i_tr in range(n_train):
            # Find channel activation for this feature value
            k_min = np.argmin((y[i_tr] - self.C_D_)**2)
            F[i_tr, :] = self.C_[:, k_min]

        # clf = linear_model.LinearRegression(fit_intercept=False,
        #                                     normalize=False)


        # Do regression
        # clf.fit(F, X)
        # self.W_ = clf

        self.W_ = np.matmul(X.transpose(), np.linalg.pinv(F.transpose()))
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
            pred_dir: numpy array of estimated feature values.
        """
        # Check that data matrix is well conditioned:
        if np.linalg.cond(X) > 9000:
            logger.error("Data is singular.")
            raise ValueError("Data matrix is nearly singular.")

        # Check that the data matrix is the right size
        shape_data = np.shape(X)
        if len(shape_data) != 2:
            raise ValueError("Data matrix has too many or too few "
                             "dimensions.")

        pred_dir = self._predict_directions(X)

        return pred_dir

    def score(self, X, y):
        """Calculate error measure of prediction.

        Parameters
        ----------
            X: numpy matrix of voxel activation from new data
                [observations,voxels]
            y: numpy array of responses. [observations]

        Returns
        -------
            rss: residual sum of squares of predicted
                features compared to to actual features.
        """
        pred_dir = self.predict(X)
        u = ((y - pred_dir)**2).sum()
        v = ((y - np.mean(y))**2).sum()
        rss = (1 - u/v)
        return rss

    def get_params(self, deep=True):
        """Returns model parameters.

        Parameters
        ----------
        deep: boolean. default true.
            if true, returns params of sub-objects

        Returns
        -------
        params: parameter of this object
        """
        return{"n_channels": self.n_channels,
               "channel_exp": self.channel_exp,
               "range_start": self.range_start,
               "range_stop": self.range_stop}

    def set_params(self, **parameters):
        """Sets model parameters after initialization.

        Parameters
        ----------
            params: structure with parameters and change values
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _define_channels(self):
        """Define basis functions (aka channels).

        Returns
        -------
            channels: numpy matrix of basis functions.
                    [n_channels, function resolution].
            channel_domain: numpy array of domain values.
        """
        channel_density = 180
        shifts_ = np.linspace(0,
                              channel_density -
                              channel_density/self.n_channels,
                              self.n_channels)
        shifts = [int(i) for i in shifts_]
        channel_domain = np.linspace(self.range_start,
                                     self.range_stop,
                                     channel_density)
        channel_0 = np.sin(np.linspace(0, np.pi, channel_density)
                           ) ** self.channel_exp
        channels = np.zeros((self.n_channels, channel_density))
        for i in range(self.n_channels):
            this_ch = np.roll(channel_0, shifts[i])
            channels[i, :] = np.maximum(np.zeros(channel_density), this_ch)

        # Check that channels provide sufficient coverage
        ch_sum_range = np.max(np.sum(channels, 0)) - \
            np.min(np.sum(channels, 0))
        if ch_sum_range > \
                np.deg2rad(self.range_stop - self.range_start)*0.1:
            # if range of channel sum > 10% channel domain size
            raise ValueError("Insufficient channel coverage.")
        return channels, channel_domain

    def _predict_channel_responses(self, X):
        """Computes predicted basis function values from data

        Parameters
        ----------
            X: numpy data matrix. [observations, voxels]

        Returns
        -------
            channel_response: numpy matrix of channel responses
        """
        # clf = linear_model.LinearRegression(fit_intercept=False,
        #                                    normalize=False)
        # clf.fit(self.W_.coef_, X.transpose())

        channel_response = \
            np.matmul(np.linalg.pinv(self.W_), X.transpose())

        return channel_response

    def _predict_direction_responses(self, X):
        """Predicts basis function response across channels from data in X

        Parameters
         ---------
            X: numpy matrix of data. [observations, voxels]

        Returns
        -------
            pred_response: predict response from all channels. Used
                        to predict feature (direction).
        """

        pred_response = np.matmul(self.C_.transpose(),
                                  self._predict_channel_responses(X))
        return pred_response

    def _predict_directions(self, X):
        """Predicts feature value (direction) from data in X

        Parameters
         ---------
            X: numpy matrix of data. [observations, voxels]

        Returns
        -------
            pred_direction: predicted direction from collective response across all
                channels. Used to predict feature (direction).
        """

        pred_response = self._predict_direction_responses(X)
        dir_ind = np.argmax(pred_response, 0)
        pred_direction = self.C_D_[dir_ind]

        return pred_direction