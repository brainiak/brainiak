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

    Inverted Encoding Model (IEM) Test with fabricated data

    author: David Huberdeau

"""

import numpy as np
import brainiak.reconstruct.iem
import logging

logger = logging.getLogger(__name__)

# Generate synthetic data with dimension 9 and linearly separable

n, dim = 300, 9
n_ = int(n/3)
np.random.seed(0)
C = -.25 + .5*np.random.rand(dim, dim)  # covariance matrix
centers_0 = np.linspace(-1, 1, dim)
centers_60 = np.roll(centers_0,5)
centers_120 = centers_0[::-1]
X_ = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
          np.dot(np.random.randn(n_, dim), C) + centers_60,
          np.dot(np.random.randn(n_, dim), C) + centers_120))

X = X_/np.max((np.max(X_), np.min(X_)))

y = np.hstack((np.zeros(n_), 60*np.ones(n_), 120*np.ones(n_)))

# Create iem object

Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6, 6, -30, 210)
Invt_model.fit(X, y)

X2_0 = np.dot(np.random.randn(n_, dim), C) + centers_0
X2_60 = np.dot(np.random.randn(n_, dim), C) + centers_60
X2_120 = np.dot(np.random.randn(n_, dim), C) + centers_120

y2_0 = np.zeros(n_)
y2_60 = 60*np.ones(n_)
y2_120 = 120*np.ones(n_)

r_hat_0 = Invt_model.predict(X2_0)
r_hat_60 = Invt_model.predict(X2_60)
r_hat_120 = Invt_model.predict(X2_120)

y_hat_0 = Invt_model._predict_directions(X2_0)
y_hat_60 = Invt_model._predict_directions(X2_60)
y_hat_120 = Invt_model._predict_directions(X2_120)

m_reconstruct = [np.mean(r_hat_0), np.mean(r_hat_60), np.mean(r_hat_120)]
logger.info('Reconstructed angles: ' + str(m_reconstruct))

m0 = np.mean(y_hat_0, axis=0)
m60 = np.mean(y_hat_60, axis=0)
m120 = np.mean(y_hat_120, axis=0)

d0 = np.argmax(m0)
d60= np.argmax(m60)
d120 = np.argmax(m120)

X2_ = np.vstack((X2_0, X2_60, X2_120))
y2_ = np.hstack((y2_0, y2_60, y2_120))

score_ = Invt_model.score(X2_, y2_)

logger.info('Scores: ' + str(score_))