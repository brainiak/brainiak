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
from brainiak.reconstruct.iem import InvertedEncoding1D, InvertedEncoding2D
from brainiak.utils.fmrisim import generate_1d_gaussian_rfs, \
    generate_1d_rf_responses
from scipy.stats import circmean

logger = logging.getLogger(__name__)


# Simple test: can an instance be instantiated?
def test_can_instantiate():
    s = InvertedEncoding1D()
    assert s, "Invalid InvertedEncoding1D instance"
    s2 = InvertedEncoding2D(stim_xlim=[0, 0], stim_ylim=[0, 0],
                            stimulus_resolution=[0, 0])
    assert s2, "Invalid InvertedEncoding2D instance"


# Simple test for checking range values.
def test_instantiate_improper_range():
    with pytest.raises(ValueError):
        s = InvertedEncoding1D(6, 5, 'halfcircular', range_start=20,
                               range_stop=0)
        assert s, "Invalid InvertedEncoding1D instance"
    with pytest.raises(ValueError):
        s2 = InvertedEncoding2D(stim_xlim=[0, -1], stim_ylim=[0, -1],
                                stimulus_resolution=[10, 10])
        assert s2, "Invalid InvertedEncoding2D instance"


# Test to check stimulus resolution input
def test_1d_stimulus_resolution():
    s = InvertedEncoding1D(6, 5, stimulus_resolution=360)
    assert s.stim_res == 360


# Test to check that stimulus resolution is used properly
def test_2d_stimulus_resolution():
    s2 = InvertedEncoding2D(stim_xlim=[-1, 1], stim_ylim=[-1, 1],
                            stimulus_resolution=10) 
    assert len(s2.stim_pixels[0] == 10)
    assert len(s2.stim_pixels[1] == 10)
    s2 = InvertedEncoding2D(stim_xlim=[-1, 1], stim_ylim=[-2, 2],
                            stimulus_resolution=[10, 20]) 
    assert len(s2.stim_pixels[0] == 10)
    assert len(s2.stim_pixels[1] == 20)


# Test that 2D channels can be set by the user
def test_2d_custom_channels():
    nchan = 8
    res = 10
    npix = res*res
    channels = np.random.rand(nchan, npix)*2 - 1
    bds = [-1, 1]
    s = InvertedEncoding2D(stim_xlim=bds, stim_ylim=bds,
                           stimulus_resolution=res, chan_xlim=bds,
                           chan_ylim=bds, channels=channels)
    assert s, "Unable to define custom InvertedEncoding2D channels"


# Provide invalid data so that channels cannot be created.
def test_cannot_instantiate_1d_channels():
    with pytest.raises(ValueError):
        s = InvertedEncoding1D(n_channels=0)
        assert s, "Invalid InvertedEncoding1D instance"


# Test that modifying channel properties is not allowed.
def test_cannot_instantiate_2d_channels():
    nchan = 8
    res = 10
    npix = res*res
    channels = np.random.rand(nchan, npix)*2 - 1
    bds = [-1, 1]
    s = InvertedEncoding2D(stim_xlim=bds, stim_ylim=bds,
                           stimulus_resolution=res, chan_xlim=bds,
                           chan_ylim=bds, channels=channels)
    with pytest.raises(ValueError):
        s.set_params({'n_channels': nchan - 1})


# Provide invalid stimulus mode
def test_stimulus_mode():
    with pytest.raises(ValueError):
        s = InvertedEncoding1D(6, 5, 'random')
        assert s, "Invalid InvertedEncoding1D instance"


# Provide mismatching range and stimulus_mode input
def test_range_stimulus_mode_circ():
    with pytest.raises(ValueError):
        s = InvertedEncoding1D(6, 5, 'circular', 0, 180)
        assert s, "Invalid InvertedEncoding1D instance"


# Provide mismatching range & stimulus mode, with half circular
def test_range_stimulus_mode_halfcirc():
    with pytest.raises(ValueError):
        s = InvertedEncoding1D(6, 5, 'halfcircular', -10, 350)
        assert s, "Invalid InvertedEncoding1D instance"


# Test for n_observations < n_channels
def test_data_amount():
    x = np.random.rand(5, 1000)
    s = InvertedEncoding1D()
    with pytest.raises(ValueError):
        s.fit(x, np.random.rand(5))
        assert s, "Invalid data"


# Test number of data dimensions
def test_data_dimensions():
    x = np.random.rand(5, 10, 2)
    s = InvertedEncoding1D()
    with pytest.raises(ValueError):
        s.fit(x, np.random.rand(5))


# Define some data to use in the following tests.
n, dim = 297, 9
n_ = n // dim
y = np.repeat(np.linspace(0, 180-(180/dim), dim), n_)
voxel_rfs, _ = generate_1d_gaussian_rfs(dim, 180, (0, 179),
                                        random_tuning=False)
X = generate_1d_rf_responses(voxel_rfs, y, 180, (0, 179),
                             trial_noise=0.25).transpose()
X2 = generate_1d_rf_responses(voxel_rfs, y, 180, (0, 179),
                              trial_noise=0.25).transpose()


# Test if valid data can be fit.
def test_can_fit_data():
    Invt_model = InvertedEncoding1D()
    Invt_model.fit(X, y)


# Test if valid data can be fit in circular space.
def test_can_fit_circular_space():
    s = InvertedEncoding1D(6, 5, 'circular', range_stop=360)
    s.fit(X, y)


# Show that a data matrix with improper format (dimensions) breaks the
# algorithm.
def test_cannot_fit_data():
    with pytest.raises(ValueError):
        Invt_model = InvertedEncoding1D()
        Invt_model.fit(X.transpose(), y)


def test_ill_conditioned_train_data():
    Invt_model = InvertedEncoding1D()
    with pytest.raises(ValueError):
        X = np.array([[0, 0, 0], [1, 1, 1]])
        Invt_model.fit(X, np.array([0, 0, 0]))


# Test case if data dimensions are wrong
def test_extra_data_dimensions():
    with pytest.raises(ValueError):
        n, dim1, dim2 = 300, 3, 3
        X = np.random.rand(n//3, dim1, dim2)
        Invt_model = InvertedEncoding1D()
        Invt_model.fit(X, y)


# Test case when # of observations are not matched btwn data & labels
def test_mismatched_observations():
    with pytest.raises(ValueError):
        Invt_model = InvertedEncoding1D()
        Invt_model.fit(X, y[:-50])


# Test prediction capability from valid (fabricated) data
def test_can_predict_from_data():
    Invt_model = InvertedEncoding1D()
    Invt_model.fit(X, y)
    m_reconstruct = []
    for j in np.arange(dim):
        preds = Invt_model.predict(X2[n_*j:n_*(j+1), :])
        tmp = circmean(np.deg2rad(preds))
        m_reconstruct.append(np.rad2deg(tmp))
    logger.info('Reconstructed angles: ' + str(m_reconstruct))


# Show that prediction is invalid when input data is wrong size
def test_cannot_predict_from_data():
    Invt_model = InvertedEncoding1D()
    Invt_model.fit(X, y)
    with pytest.raises(ValueError):
        _ = Invt_model.predict(X2[0:n_, :].transpose())


# Show proper scoring function with valid (fabricated) test data
def test_can_score():
    Invt_model = InvertedEncoding1D()
    Invt_model.fit(X, y)
    score = Invt_model.score(X2, y)
    logger.info('Scores: ' + str(score))


# Test scoring with invalid data formatting
def test_cannot_score():
    with pytest.raises(ValueError):
        Invt_model = InvertedEncoding1D()
        Invt_model.fit(X, y)
        score = Invt_model.score(X2.transpose(), y)
        logger.info('Scores: ' + str(score))


# Test stimulus resolution that is not even multiple
def test_stimulus_resolution_odd():
    Invt_model = InvertedEncoding1D(stimulus_resolution=59)
    with pytest.raises(NotImplementedError):
        Invt_model.fit(X, y)


# Test stimulus masking
def test_stimulus_mask():
    Invt_model = InvertedEncoding1D(6, 5, range_start=-10,
                                    range_stop=170,
                                    stimulus_resolution=60)
    chans, _ = Invt_model._define_channels()
    Invt_model.set_params(channels_=chans)
    with pytest.warns(RuntimeWarning):
        C = Invt_model._define_trial_activations(np.array([50]))
        tmp_C = np.repeat([0, 1, 0], 60) @ chans.transpose()
        assert np.all((C - tmp_C) < 1e-7)


# Test stimulus masking with different range
def test_stimulus_mask_shift_positive():
    Invt_model = InvertedEncoding1D(6, 5, range_start=10,
                                    range_stop=190,
                                    stimulus_resolution=60)
    chans, _ = Invt_model._define_channels()
    Invt_model.set_params(channels_=chans)
    with pytest.warns(RuntimeWarning):
        C = Invt_model._define_trial_activations(np.array([70]))
        tmp_C = np.repeat([0, 1, 0], 60) @ chans.transpose()
        assert np.all((C - tmp_C) < 1e-7)


# Test ability to get model parameters from object
def test_can_get_params():
    s = InvertedEncoding1D()
    param_out = s.get_params()
    logger.info('Returned Parameters: ' +
                str(param_out.get('n_channels')) +
                ', ' + str(param_out.get('range_start')) +
                ', ' + str(param_out.get('range_stop')))


# Test ability to set model parameters of an object instance
def test_can_set_params():
    s = InvertedEncoding1D()
    s.set_params(n_channels=10,
                 stimulus_mode='circular',
                 range_start=-90,
                 range_stop=270,
                 channel_exp=4,
                 verbose=False)
