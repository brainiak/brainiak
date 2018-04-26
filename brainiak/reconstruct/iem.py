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

The implementations are based on the following publications:

.. [Kok2013] "1.Kok, P., Brouwer, G. J., Gerven, M. A. J. van &
Lange, F. P. de. Prior Expectations Bias Sensory Representations
in Visual Cortex. J. Neurosci. 33, 16275–16284 (2013).

.. [Bouwer2009] "1.Brouwer, G. J. & Heeger, D. J.
Decoding and Reconstructing Color from Responses in Human Visual
Cortex. J. Neurosci. 29, 13992–14003 (2009).

"""

# Authors: David Huberdeau (Yale University), 2018;
# Peter Kok (Yale University)

import logging
import numpy as np
from sklearn import linear_model
from sklearn.base import BaseEstimator
import math
from scipy.ndimage.interpolation import shift

__all__ = [
    "InvertedEncoding"
]

logger = logging.getLogger(__name__)


class InvertedEncoding(BaseEstimator):

    def __init__(self, n_channels=5, range_start=0, range_stop=180):
        self.n_channels = n_channels
        self.range_start = range_start
        self.range_stop = range_stop

    def fit(self, X, y):
        # Check that there are channels specified
        if self.n_channels < 1:
            raise ValueError("Insufficient channels.")

        self.range_ = np.linspace(self.range_start,
                                  self.range_stop,
                                  self.n_channels)
        self.C_ = self._define_channels()

        n_train = len(y)
        train_labels = np.round(y).astype(int)
        F = np.empty((n_train, self.n_channels))
        for i_tr in range(n_train):
            # Find channel activation for this orientation
            F[i_tr, :] = self.C_[:, train_labels[i_tr]]

        clf = linear_model.LinearRegression(fit_intercept=False,
                                            normalize=False)
        clf.fit(F, X)
        self.W_ = clf.coef_
        return self

    def predict(self, X):
        clf = linear_model.LinearRegression(fit_intercept=False,
                                            normalize=False)
        clf.fit(self.W_, X)
        channel_response = clf.coef_
        return channel_response

    def score(self, X, y):
        pred_channel_resp = self.predict(X)

        # from the channel responses, we can reconstruct the
        # represented orientation
        orientation = pred_channel_resp.dot(self.C)
        u = ((y - orientation)**2).sum()
        v = ((y - np.mean(y))**2).sum()
        rss = (1 - u/v)
        return rss

    def _define_channels(self):
        # avoid non-integers
        channel_peaks = np.round(self.range_[0]).astype(int)
        channel_shifts = channel_peaks - int(self.range_stop / 2)

        idealized_channel = np.sin(np.linspace(self.range_[0],
                                               math.pi,
                                               self.range_[-1]))
        idealized_channel = idealized_channel ** (self.n_channels - 1)

        channels = np.zeros(self.n_channels, self.range_)
        for i in range(self.n_channels):
            channels[i, :] = shift(idealized_channel,
                                   channel_shifts[i],
                                   cval=0)
            channels[i, :] = np.roll(idealized_channel,
                                     channel_shifts[i])
        return channels
