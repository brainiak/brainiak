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

# Authors: David Huberdeau (Yale University) &
# Peter Kok (Yale University), 2018 &
# Vy Vo (Intel Corp., UCSD), 2019

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
    s2 = InvertedEncoding2D(stim_xlim=[0, 1], stim_ylim=[0, 1],
                            stimulus_resolution=[1, 1])
    assert s2, "Invalid InvertedEncoding2D instance"


# Test for checking range values.
def test_instantiate_improper_range():
    with pytest.raises(ValueError):
        s = InvertedEncoding1D(6, 5, 'halfcircular', range_start=20,
                               range_stop=0)
        assert s, "Invalid InvertedEncoding1D instance"
    with pytest.raises(ValueError):
        s2 = InvertedEncoding2D(stim_xlim=[0, -1], stim_ylim=[0, -1],
                                stimulus_resolution=[10, 10])
        assert s2, "Invalid InvertedEncoding2D instance"
    with pytest.raises(ValueError):
        s2 = InvertedEncoding2D(stim_xlim=[0], stim_ylim=[-1, 0],
                                stimulus_resolution=10)
        assert s2, "Invalid InvertedEncoding2D instance"


# Test for n_observations < n_channels
def test_data_amount():
    x = np.random.rand(5, 1000)
    s = InvertedEncoding1D()
    with pytest.raises(ValueError):
        s.fit(x, np.random.rand(5))
        assert s, "Invalid data"
    s2 = InvertedEncoding2D(stim_xlim=[-1, 1], stim_ylim=[-1, 1],
                            stimulus_resolution=10)
    with pytest.raises(ValueError):
        s2.fit(x, np.random.rand(5))


# Test number of data dimensions
def test_data_dimensions():
    x = np.random.rand(5, 10, 2)
    s = InvertedEncoding1D()
    with pytest.raises(ValueError):
        s.fit(x, np.random.rand(5))
    s2 = InvertedEncoding2D(stim_xlim=[-1, 1], stim_ylim=[-1, 1],
                            stimulus_resolution=10)
    with pytest.raises(ValueError):
        s2.fit(x, np.random.rand(5))


# TESTS FOR 2D MODEL #
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


# Test that channel definition should be consistent.
def test_cannot_instantiate_2d_channels():
    # Channel definition over wrong number of pixels (5 instead of 100)
    with pytest.raises(ValueError):
        s = InvertedEncoding2D(stim_xlim=[-1, 1], stim_ylim=[-1, 1],
                               stimulus_resolution=10,
                               channels=np.random.rand(5, 5))
        assert s, "Invalid InvertedEncoding2D instance"


# Test that you cannot modify properties in an inconsistent way.
def test_modify_2d_properties():
    nchan = 8
    res = 10
    npix = res*res
    channels = np.random.rand(nchan, npix)*2 - 1
    bds = [-1, 1]
    s = InvertedEncoding2D(stim_xlim=bds, stim_ylim=bds,
                           stimulus_resolution=res, chan_xlim=bds,
                           chan_ylim=bds, channels=channels)
    with pytest.raises(ValueError):
        s = s.set_params(n_channels=nchan - 1)
        assert s, "Invalid InvertedEncoding2D instance"
    with pytest.raises(ValueError):
        s = s.set_params(xp=np.random.rand(npix - 10))
        assert s, "Invalid InvertedEncoding2D instance"
    with pytest.raises(ValueError):
        s = s.set_params(stim_fov=[[0, 1], [0, -1]])
        assert s, "Invalid InvertedEncoding2D instance"
    with pytest.raises(ValueError):
        s = s.set_params(stim_fov=[[0, 1]])
        assert s, "Invalid InvertedEncoding2D instance"
    with pytest.raises(ValueError):
        s = s.set_params(stim_fov=[[0], [0, 1]])
        assert s, "Invalid InvertedEncoding2D instance"


# Test that you can get object properties
def test_get_2d_params():
    bds = [-1, 1]
    res = 10
    s = InvertedEncoding2D(stim_xlim=bds, stim_ylim=bds,
                           stimulus_resolution=res)
    param_out = s.get_params()
    assert np.all(param_out.get('stim_fov')[0] == bds)
    assert param_out.get('xp').size == res*res


# Test helper function to create 2D cosine
def test_2d_cos():
    nchan = 8
    res = 10
    npix = res*res
    bds = [-1, 1]
    sz = 2
    s = InvertedEncoding2D(stim_xlim=bds, stim_ylim=bds,
                           stimulus_resolution=res,
                           channels=np.random.rand(nchan, npix))
    sz = s._2d_cosine_fwhm_to_sz(1)
    fcn = s._make_2d_cosine(s.xp.reshape(-1, 1), s.yp.reshape(-1, 1),
                            np.linspace(bds[0], bds[1], nchan),
                            np.linspace(bds[0], bds[1], nchan), sz)
    assert fcn.shape == (nchan, npix)
    # Test that masking works -- basis function should have fewer non-zero
    # elements than specified by the size constant
    xd = np.diff(s.xp)[0][0]
    nval = (np.nonzero(fcn[0, :])[0]).size
    assert nval*(xd**2) <= sz**2


# Test size conversion functions
def test_2d_cos_size_fcns():
    bds = [-1, 1]
    s = np.random.rand()
    imodel = InvertedEncoding2D(stim_xlim=bds, stim_ylim=bds,
                                stimulus_resolution=10)
    fwhm = imodel._2d_cosine_sz_to_fwhm(s)
    s2 = imodel._2d_cosine_fwhm_to_sz(fwhm)
    assert np.isclose(s, s2)
    fwhm2 = imodel._2d_cosine_sz_to_fwhm(s2)
    assert np.isclose(fwhm, fwhm2)


def test_square_basis_grid():
    nchan = 8
    bds = [-1, 1]
    s = InvertedEncoding2D(stim_xlim=bds, stim_ylim=bds,
                           stimulus_resolution=10)
    _, centers = s.define_basis_functions_sqgrid(nchannels=nchan)
    assert centers.shape[0] == nchan*nchan
    xspacing = np.round(np.diff(centers[:, 0]), 5)
    yspacing = np.round(np.diff(centers[:, 1]), 5)
    assert xspacing[0] == xspacing[28] == xspacing[-1]
    assert yspacing[0] == yspacing[25] == yspacing[-1]


def test_triangular_basis_grid():
    grid_rad = 3
    n_channels = (grid_rad*2 + 1) * (grid_rad*2)
    bds = [-1, 1]
    s = InvertedEncoding2D(stim_xlim=bds, stim_ylim=bds,
                           stimulus_resolution=10)
    _, centers = s.define_basis_functions_trigrid(grid_rad)
    assert centers.shape[0] == n_channels
    xspacing = np.round(np.diff(centers[:, 0]), 4)
    assert xspacing[0] == xspacing[np.random.randint(n_channels)] == \
        xspacing[-1]
    ysp = xspacing[0] * np.sqrt(3) * 0.5
    yspacing = np.diff(centers[:, 1])
    yspace = yspacing[yspacing > 0.0]
    assert np.all((ysp - yspace) < 1e-5)


# Define some data to use in the following tests.
nobs, nvox, ntest = 100, 1000, 5
xlim, ylim = [[-6, 6], [-3, 3]]
res = [100, 100]
sxx, syy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                       np.linspace(ylim[0], ylim[1], 10))
yd = np.hstack((sxx.reshape(-1, 1), syy.reshape(-1, 1)))
Xd = np.zeros((nobs, nvox))
for i, l in enumerate(np.linspace(-1, 1, 10)):
    Xd[i*10:i*10+10, :] = np.random.normal(loc=l, scale=1.5,
                                           size=(10, nvox))
X2d = np.zeros((ntest, nvox))
for i, l in enumerate(np.linspace(-1, 1, 5)):
    X2d[i, :] = np.random.normal(loc=l, scale=1.5,
                                 size=(1, nvox))


# Test that 2D model raises error if design matrix C cannot be defined
def test_fit_invalid_2d():
    # C=None and stim_radius=None here, cannot define C
    i2 = InvertedEncoding2D(stim_xlim=xlim, stim_ylim=ylim,
                            stimulus_resolution=res, stim_radius=None)
    i2.define_basis_functions_sqgrid(nchannels=[12, 6])
    with pytest.raises(ValueError):
        i2.fit(Xd, yd)


# Test attempt to fit with list of varying stimulus radii
def test_fit_2d_radius_list():
    i2 = InvertedEncoding2D(stim_xlim=xlim, stim_ylim=ylim,
                            stimulus_resolution=res,
                            stim_radius=np.random.rand(nobs))
    i2.define_basis_functions_sqgrid(nchannels=[12, 6])
    i2.fit(Xd, yd)


# Test with custom C input
def test_fit_custom_channel_activations():
    i2 = InvertedEncoding2D(stim_xlim=xlim, stim_ylim=ylim,
                            stimulus_resolution=res,
                            stim_radius=12)
    i2.define_basis_functions_sqgrid(nchannels=[12, 6])
    # Define C by expanding y & adding noise to avoid singular W matrix error
    C0 = np.repeat(np.expand_dims(yd[:, 0], 1), 12*3, axis=1) + \
        np.random.rand(nobs, 12*3)
    C1 = np.repeat(np.expand_dims(yd[:, 1], 1), 12*3, axis=1) + \
        np.random.rand(nobs, 12*3)
    i2.fit(Xd, yd, np.hstack((C0, C1)))
    assert np.all(i2.W_)


iem_2d = InvertedEncoding2D(stim_xlim=xlim, stim_ylim=ylim,
                            stimulus_resolution=res, stim_radius=12)
iem_2d.define_basis_functions_sqgrid(nchannels=[12, 6])


# Test if valid data can be fit.
def test_can_fit_2d_data():
    iem_2d.fit(Xd, yd)


# Show that a data matrix with improper format (dimensions) breaks the
# algorithm.
def test_cannot_fit_2d_data():
    with pytest.raises(ValueError):
        iem_2d.fit(Xd.transpose(), yd)


# Ill conditioned data matrix will raise error
def test_ill_conditioned_2d_train_data():
    with pytest.raises(ValueError):
        Xt = np.ones((nobs, nvox))
        y = np.random.rand(nobs, 2)
        iem_2d.fit(Xt, y)


# Ill conditioned channel activations C will raise warning
def test_ill_conditioned_2d_channel_activations():
    with pytest.warns(RuntimeWarning):
        C = iem_2d._define_trial_activations(np.ones((nobs, 2)))
        assert np.linalg.matrix_rank(C) == 1


# Ill conditioned weight matrix will raise error
def test_ill_conditioned_2d_weights():
    with pytest.raises(ValueError):
        Xt = np.random.rand(nobs, nvox)
        y = np.random.rand(nobs, 2)
        iem_2d.fit(Xt, y)


# Not enough observations will trigger error
def test_insufficient_2d_data():
    with pytest.raises(ValueError):
        Xt = np.random.rand(10, nvox)
        y = np.random.rand(10, 2)
        iem_2d.fit(Xt, y)


# Test case when # of observations are not matched btwn data & labels
def test_mismatched_2d_observations():
    with pytest.raises(ValueError):
        iem_2d.fit(Xd, yd[:-50, :])


# Test prediction capability from valid (fabricated) data
def test_can_predict_from_2d_data():
    iem_2d.fit(Xd, yd)
    preds = iem_2d.predict(X2d)
    assert preds.shape == (ntest, 2)


# Show that prediction is invalid when input data is wrong size
def test_cannot_predict_from_2d_data():
    iem_2d.fit(Xd, yd)
    with pytest.raises(ValueError):
        _ = iem_2d.predict(X2d.T)


# Show proper scoring function with valid (fabricated) test data
def test_can_score_2d():
    iem_2d.fit(Xd, yd)
    score = iem_2d.score(X2d, yd[:ntest, :])
    assert score.shape == (ntest,)
    score = iem_2d.score_against_reconstructed(X2d,
                                               np.random.rand(res[0]*res[1],
                                                              ntest))
    assert score.shape == (ntest,)
    score = iem_2d.score_against_reconstructed(X2d,
                                               np.random.rand(res[0]*res[1],
                                                              ntest),
                                               metric="cosine")
    assert score.shape == (ntest,)


# Test scoring with invalid data formatting
def test_cannot_score_2d():
    iem_2d.fit(Xd, yd)
    with pytest.raises(ValueError):
        score = iem_2d.score(X2d.transpose(), yd[ntest, :])
        assert score


# TESTS FOR 1D MODEL #
# Test to check stimulus resolution input
def test_1d_stimulus_resolution():
    s = InvertedEncoding1D(6, 5, stimulus_resolution=360)
    assert s.stim_res == 360


# Provide invalid data so that channels cannot be created.
def test_cannot_instantiate_1d_channels():
    with pytest.raises(ValueError):
        s = InvertedEncoding1D(n_channels=0)
        assert s, "Invalid InvertedEncoding1D instance"


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
    assert param_out.get('channel_exp') == 5
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
