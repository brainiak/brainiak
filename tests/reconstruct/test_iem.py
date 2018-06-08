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

import pytest
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Simple test: can an instance be instantiated?
def test_can_instantiate():
    import brainiak.reconstruct.iem
    s = brainiak.reconstruct.iem.InvertedEncoding()
    assert s, "Invalid InvertedEncoding instance"


# Simple test for checking range values.
def test_instantiate_improper_range():
    import brainiak.reconstruct.iem
    with pytest.raises(ValueError):
        s = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                      6,
                                                      180,
                                                      90)
        assert s, "Invalid InvertedEncoding instance"


# Provide invalid data so that channels cannot be created.
def test_cannot_instantiate_channels():
    import brainiak.reconstruct.iem
    s = brainiak.reconstruct.iem.InvertedEncoding(n_channels=0)
    with pytest.raises(ValueError):
        X = 0
        y = 0
        s.fit(X, y)


# Test is valid (though fabricated) data can be fit.
def test_can_fit_data():
    import brainiak.reconstruct.iem

    n, dim = 300, 9
    n_ = int(n / 3)
    np.random.seed(0)
    C = -.25 + .5 * np.random.rand(dim, dim)  # covariance matrix
    centers_0 = np.linspace(-1, 1, dim)
    centers_60 = np.roll(centers_0, 5)
    centers_120 = centers_0[::-1]
    # create fabricated data that is valid for the method
    X = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                   np.dot(np.random.randn(n_, dim), C) + centers_60,
                   np.dot(np.random.randn(n_, dim), C) + centers_120))

    y = np.hstack((np.zeros(n_), 60 * np.ones(n_), 120 * np.ones(n_)))

    # Create iem object and fit data to it.
    Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                           6,
                                                           -30,
                                                           210)
    Invt_model.fit(X, y)


# Show that a data matrix with improper format (dimensions) breaks the
# algorithm.
def test_cannot_fit_data():
    import brainiak.reconstruct.iem
    with pytest.raises(ValueError):
        n, dim = 300, 9
        n_ = int(n / 3)
        np.random.seed(0)
        C = -.25 + .5 * np.random.rand(dim, dim)  # covariance matrix
        centers_0 = np.linspace(-1, 1, dim)
        centers_60 = np.roll(centers_0, 5)
        centers_120 = centers_0[::-1]
        # create fabricated data that is NOT valid for the method
        X = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                       np.dot(np.random.randn(n_, dim), C) + centers_60,
                       np.dot(np.random.randn(n_, dim), C) + centers_120))
        X = X.transpose()  # the offending line - data not correct dim.

        y = np.hstack((np.zeros(n_), 60 * np.ones(n_), 120 * np.ones(n_)))

        # Create iem object
        Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                               6,
                                                               -30,
                                                               210)
        Invt_model.fit(X, y)


def test_ill_conditioned_train_data():
    import brainiak.reconstruct.iem
    with pytest.raises(ValueError):
        n, dim = 9, 9
        n_ = int(n / 3)
        np.random.seed(0)
        C_0 = -.25 + .5 * np.random.rand(dim, 5)  # covariance matrix, initial
        C = np.hstack((C_0, C_0[:, 0:4]))
        centers_0 = np.linspace(-1, 1, dim)
        centers_60 = np.roll(centers_0, 5)
        centers_120 = centers_0[::-1]
        X = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                       np.dot(np.random.randn(n_, dim), C) + centers_60,
                       np.dot(np.random.randn(n_, dim), C) + centers_120,
                       np.dot(np.random.randn(n_, dim), C) + centers_120))

        y = np.hstack((np.zeros(3), 60 * np.ones(3), 120 * np.ones(6)))

        # Create iem object
        Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                               6,
                                                               -30,
                                                               210)
        Invt_model.fit(X, y)


# Test case if data dimensions are wrong
def test_extra_data_dimensions():
    import brainiak.reconstruct.iem
    with pytest.raises(ValueError):
        n, dim1, dim2 = 300, 3, 3
        n_ = int(n / 3)
        X = np.random.rand(n_, dim1, dim2)
        y = np.hstack((np.zeros(n_), 60 * np.ones(n_), 120 * np.ones(n_)))

        # Create iem object
        Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                               6,
                                                               -30,
                                                               120)
        Invt_model.fit(X, y)


# Test case when number of observations are not matched btwn data & labels
def test_mismatched_observations():
    import brainiak.reconstruct.iem
    with pytest.raises(ValueError):
        n, dim1 = 300, 9
        n_ = int(n / 3)
        X = np.random.rand(n_, dim1)
        y = np.hstack((np.zeros(n_ - 1),
                       60 * np.ones(n_ - 1),
                       120 * np.ones(n_ - 1)))

        # Create iem object
        Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                               6,
                                                               -30,
                                                               120)
        Invt_model.fit(X, y)


# Test prediction capability from valid (fabricated) data
def test_can_predict_from_data():
    import brainiak.reconstruct.iem
    n, dim = 300, 9
    n_ = int(n / 3)
    np.random.seed(0)
    C = -.25 + .5 * np.random.rand(dim, dim)  # covariance matrix
    centers_0 = np.linspace(-1, 1, dim)
    centers_60 = np.roll(centers_0, 5)
    centers_120 = centers_0[::-1]
    # create fabricated data that is valid for the method
    X = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                   np.dot(np.random.randn(n_, dim), C) + centers_60,
                   np.dot(np.random.randn(n_, dim), C) + centers_120))

    y = np.hstack((np.zeros(n_), 60 * np.ones(n_), 120 * np.ones(n_)))

    # Create iem object
    Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                           6,
                                                           -30,
                                                           210)
    Invt_model.fit(X, y)

    X2_0 = np.dot(np.random.randn(n_, dim), C) + centers_0
    X2_60 = np.dot(np.random.randn(n_, dim), C) + centers_60
    X2_120 = np.dot(np.random.randn(n_, dim), C) + centers_120

    r_hat_0 = Invt_model.predict(X2_0)
    r_hat_60 = Invt_model.predict(X2_60)
    r_hat_120 = Invt_model.predict(X2_120)
    m_reconstruct = [np.mean(r_hat_0), np.mean(r_hat_60), np.mean(r_hat_120)]
    logger.info('Reconstructed angles: ' + str(m_reconstruct))


# Show that prediction is invalid when input data is improperly formated
def test_cannot_predict_from_data():
    import brainiak.reconstruct.iem
    with pytest.raises(ValueError):
        n, dim = 300, 9
        n_ = int(n / 3)
        np.random.seed(0)
        C = -.25 + .5 * np.random.rand(dim, dim)  # covariance matrix
        centers_0 = np.linspace(-1, 1, dim)
        centers_60 = np.roll(centers_0, 5)
        centers_120 = centers_0[::-1]
        X = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                       np.dot(np.random.randn(n_, dim), C) + centers_60,
                       np.dot(np.random.randn(n_, dim), C) + centers_120))

        y = np.hstack((np.zeros(n_), 60 * np.ones(n_), 120 * np.ones(n_)))

        # Create iem object
        Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                               6,
                                                               -30,
                                                               210)
        Invt_model.fit(X, y)

        X2_0 = np.dot(np.random.randn(n_, dim), C) + centers_0
        X2_60 = np.dot(np.random.randn(n_, dim), C) + centers_60
        X2_120 = np.dot(np.random.randn(n_, dim), C) + centers_120

        X2_0 = X2_0.transpose()  # offending lines - data wrong dimensions.
        X2_60 = X2_60.transpose()
        X2_120 = X2_120.transpose()

        r_hat_0 = Invt_model.predict(X2_0)
        r_hat_60 = Invt_model.predict(X2_60)
        r_hat_120 = Invt_model.predict(X2_120)
        m_reconstruct = [np.mean(r_hat_0),
                         np.mean(r_hat_60),
                         np.mean(r_hat_120)]
        logger.info('Reconstructed angles: ' + str(m_reconstruct))


def test_ill_conditioned_test_data():
    import brainiak.reconstruct.iem
    with pytest.raises(ValueError):
        n, dim = 300, 9
        n_ = int(n / 3)
        np.random.seed(0)
        C = -.25 + .5 * np.random.rand(dim, dim)  # covariance matrix
        centers_0 = np.linspace(-1, 1, dim)
        centers_60 = np.roll(centers_0, 5)
        centers_120 = centers_0[::-1]
        X = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                       np.dot(np.random.randn(n_, dim), C) + centers_60,
                       np.dot(np.random.randn(n_, dim), C) + centers_120))

        y = np.hstack((np.zeros(n_), 60 * np.ones(n_), 120 * np.ones(n_)))

        # Create iem object
        Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                               6,
                                                               -30,
                                                               210)
        Invt_model.fit(X, y)

        # offending lines - data ill conditioned.
        n_ = 3
        C_0 = -.25 + .5 * np.random.rand(dim, 5)  # covariance matrix, initial
        C = np.hstack((C_0, C_0[:, 0:4]))  # cov. matrix is non-invertible
        X_120 = np.dot(np.random.randn(n_, dim), C) + centers_120
        X_t = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                         np.dot(np.random.randn(n_, dim), C) + centers_60,
                         X_120,
                         X_120))
        r_hat = Invt_model.predict(X_t)
        m_reconstruct = np.mean(r_hat)
        logger.info('Reconstructed angles: ' + str(m_reconstruct))


# Show proper scoring function with valid (fabricated) test data
def test_can_score():
    import brainiak.reconstruct.iem

    n, dim = 300, 9
    n_ = int(n / 3)
    np.random.seed(0)
    C = -.25 + .5 * np.random.rand(dim, dim)  # covariance matrix
    centers_0 = np.linspace(-1, 1, dim)
    centers_60 = np.roll(centers_0, 5)
    centers_120 = centers_0[::-1]
    X = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                   np.dot(np.random.randn(n_, dim), C) + centers_60,
                   np.dot(np.random.randn(n_, dim), C) + centers_120))

    y = np.hstack((np.zeros(n_), 60 * np.ones(n_), 120 * np.ones(n_)))

    # Create iem object
    Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                           6,
                                                           -30,
                                                           210)
    Invt_model.fit(X, y)

    X2_0 = np.dot(np.random.randn(n_, dim), C) + centers_0
    X2_60 = np.dot(np.random.randn(n_, dim), C) + centers_60
    X2_120 = np.dot(np.random.randn(n_, dim), C) + centers_120

    y2_0 = np.zeros(n_)
    y2_60 = 60 * np.ones(n_)
    y2_120 = 120 * np.ones(n_)

    X2_ = np.vstack((X2_0, X2_60, X2_120))
    y2_ = np.hstack((y2_0, y2_60, y2_120))

    score = Invt_model.score(X2_, y2_)
    logger.info('Scores: ' + str(score))


# Test scoring with invalid data formatting
def test_cannot_score():
    import brainiak.reconstruct.iem
    with pytest.raises(ValueError):
        n, dim = 300, 9
        n_ = int(n / 3)
        np.random.seed(0)
        C = -.25 + .5 * np.random.rand(dim, dim)  # covariance matrix
        centers_0 = np.linspace(-1, 1, dim)
        centers_60 = np.roll(centers_0, 5)
        centers_120 = centers_0[::-1]
        X = np.vstack((np.dot(np.random.randn(n_, dim), C) + centers_0,
                       np.dot(np.random.randn(n_, dim), C) + centers_60,
                       np.dot(np.random.randn(n_, dim), C) + centers_120))

        y = np.hstack((np.zeros(n_), 60 * np.ones(n_), 120 * np.ones(n_)))

        # Create iem object
        Invt_model = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                               6,
                                                               -30,
                                                               210)
        Invt_model.fit(X, y)

        # create data intentionally insufficient
        X2_0 = np.dot(np.random.randn(2, dim), C) + centers_0
        X2_60 = np.dot(np.random.randn(2, dim), C) + centers_60
        X2_120 = np.dot(np.random.randn(2, dim), C) + centers_120

        y2_0 = np.zeros(n_)
        y2_60 = 60 * np.ones(n_)
        y2_120 = 120 * np.ones(n_)

        X2_ = np.vstack((X2_0, X2_60, X2_120))
        y2_ = np.hstack((y2_0, y2_60, y2_120))

        score = Invt_model.score(X2_, y2_)
        logger.info('Scores: ' + str(score))


# Test ability to get model parameters from object
def test_can_get_params():
    import brainiak.reconstruct.iem
    s = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                  6,
                                                  0,
                                                  180)
    param_out = s.get_params()
    logger.info('Returned Parameters: ' +
                str(param_out.get('n_channels')) +
                ', ' + str(param_out.get('range_start')) +
                ', ' + str(param_out.get('range_stop')))


# Test ability to set model parameters of an object instance
def test_can_set_params():
    import brainiak.reconstruct.iem
    s = brainiak.reconstruct.iem.InvertedEncoding(6,
                                                  6,
                                                  0,
                                                  180)
    s.set_params(n_channels=10,
                 range_start=-90,
                 range_stop=270)
