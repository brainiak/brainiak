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
"""

    Inverted Encoding Model (IEM)

    The implementation is roughly based on the following publications:

    .. [Kok2013] "1.Kok, P., Brouwer, G. J., Gerven, M. A. J. van &
    Lange, F. P. de. Prior Expectations Bias Sensory Representations
    in Visual Cortex. J. Neurosci. 33, 16275–16284 (2013).

    .. [Bouwer2009] "1.Brouwer, G. J. & Heeger, D. J.
    Decoding and Reconstructing Color from Responses in Human Visual
    Cortex. J. Neurosci. 29, 13992–14003 (2009).

"""

# Authors: David Huberdeau (Yale University) &
# Peter Kok (Yale University), 2018

import logging
import numpy as np
from sklearn import linear_model
from sklearn.base import BaseEstimator
import math

__all__ = [
    "InvertedEncoding"
]

logger = logging.getLogger(__name__)


class InvertedEncoding(BaseEstimator):

    def __init__(self, n_channels=5, range_start=0, range_stop=180):
        self.n_channels = n_channels  # default = 5
        self.range_start = range_start  # in degrees, 0 - 360, def=0
        self.range_stop = range_stop  # in degrees, 0 - 360, def=180

    def fit(self, X, y):
        # Check that there are channels specified
        if self.n_channels < 2:
            raise ValueError("Insufficient channels.")
        # Check that there is enough data.. should be more
        # samples than voxels (i.e. X should be tall)
        shape_data = np.shape(X)
        shape_labels = np.shape(y)
        if len(shape_data) < 2:
            raise ValueError("Not enough data")
        else:
            if np.size(X, 0) <= np.size(X, 1):
                raise ValueError("Data Matrix ill-conditioned")
            if shape_data[0] != shape_labels[0]:
                raise ValueError(
                    "Mismatched data samples and label samples")

        self.C_, self.C_D_ = self._define_channels()
        n_train = len(y)
        F = np.empty((n_train, self.n_channels))
        for i_tr in range(n_train):
            # Find channel activation for this orientation
            k_min = np.argmin((y[i_tr] - self.C_D_)**2)
            F[i_tr, :] = self.C_[:, k_min]
        clf = linear_model.LinearRegression(fit_intercept=False,
                                            normalize=False)
        clf.fit(F, X)
        self.W_ = clf
        return self

    def predict(self, X):
        # Check that there is enough data.. should be more
        # samples than voxels (i.e. X should be tall)
        shape_data = np.shape(X)
        if len(shape_data) < 2:
            raise ValueError("Not enough data")
        else:
            if np.size(X, 0) <= np.size(X, 1):
                raise ValueError("Data Matrix ill-conditioned")
        pred_response = self._predict_directions(X)
        pred_indx = np.argmax(pred_response, axis=1)
        pred_dir = self.C_D_[pred_indx]
        return pred_dir

    def score(self, X, y):
        pred_dir = self.predict(X)
        u = ((y - pred_dir)**2).sum()
        v = ((y - np.mean(y))**2).sum()
        rss = (1 - u/v)
        return rss

    def get_params(self, deep=True):
        return{"n_channels": self.n_channels,
               "range_start": self.range_start,
               "range_stop": self.range_stop}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _define_channels(self):
        channel_exp = 6
        channel_density = 180
        shifts = np.linspace(0,
                             math.pi - math.pi/self.n_channels,
                             self.n_channels)

        channel_domain = np.linspace(self.range_start,
                                     self.range_stop,
                                     channel_density)

        channels = np.zeros((self.n_channels, channel_density))
        for i in range(self.n_channels):
            channels[i, :] = np.cos(np.linspace(0, math.pi, channel_density)
                                    - shifts[i]) ** channel_exp
        # Check that channels provide sufficient coverage
        ch_sum_range = np.max(np.sum(channels, 0)) - min(np.sum(channels, 0))
        if ch_sum_range > np.deg2rad(self.range_stop - self.range_start)*0.1:
            # if range of channel sum > 10% channel domain size
            raise ValueError("Insufficient channel coverage.")
        return channels, channel_domain

    def _predict_channel_responses(self, X):
        clf = linear_model.LinearRegression(fit_intercept=False,
                                            normalize=False)
        clf.fit(self.W_.coef_, X.transpose())
        channel_response = clf.coef_
        return channel_response

    def _predict_directions(self, X):
        pred_response = self._predict_channel_responses(X).dot(self.C_)
        return pred_response
