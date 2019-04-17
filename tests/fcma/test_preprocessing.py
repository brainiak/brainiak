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

from pathlib import Path

import numpy as np

from brainiak.fcma.preprocessing import (prepare_fcma_data, prepare_mvpa_data,
                                         prepare_searchlight_mvpa_data)
from brainiak import io

data_dir = Path(__file__).parents[1] / 'io' / 'data'
expected_dir = Path(__file__).parent / 'data'
suffix = 'bet.nii.gz'
mask_file = data_dir / 'mask.nii.gz'
epoch_file = data_dir / 'epoch_labels.npy'
expected_labels = np.array([0, 1, 0, 1])


def test_prepare_fcma_data():
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    mask = io.load_boolean_mask(mask_file)
    conditions = io.load_labels(epoch_file)
    raw_data, _, labels = prepare_fcma_data(images, conditions, mask)
    expected_raw_data = np.load(expected_dir / 'expected_raw_data.npy')
    assert len(raw_data) == len(expected_raw_data), \
        'numbers of epochs do not match in test_prepare_fcma_data'
    for idx in range(len(raw_data)):
        assert np.allclose(raw_data[idx], expected_raw_data[idx]), \
            'raw data do not match in test_prepare_fcma_data'
    assert np.array_equal(labels, expected_labels), \
        'the labels do not match in test_prepare_fcma_data'
    from brainiak.fcma.preprocessing import RandomType
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    random_raw_data, _, _ = prepare_fcma_data(images, conditions, mask,
                                              random=RandomType.REPRODUCIBLE)
    assert len(random_raw_data) == len(expected_raw_data), \
        'numbers of epochs do not match in test_prepare_fcma_data'
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    random_raw_data, _, _ = prepare_fcma_data(images, conditions, mask,
                                              random=RandomType.UNREPRODUCIBLE)
    assert len(random_raw_data) == len(expected_raw_data), \
        'numbers of epochs do not match in test_prepare_fcma_data'


def test_prepare_mvpa_data():
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    mask = io.load_boolean_mask(mask_file)
    conditions = io.load_labels(epoch_file)
    processed_data, labels = prepare_mvpa_data(images, conditions, mask)
    expected_processed_data = np.load(expected_dir
                                      / 'expected_processed_data.npy')
    assert len(processed_data) == len(expected_processed_data), \
        'numbers of epochs do not match in test_prepare_mvpa_data'
    for idx in range(len(processed_data)):
        assert np.allclose(processed_data[idx],
                           expected_processed_data[idx]), (
            'raw data do not match in test_prepare_mvpa_data')
    assert np.array_equal(labels, expected_labels), \
        'the labels do not match in test_prepare_mvpa_data'


def test_prepare_searchlight_mvpa_data():
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    conditions = io.load_labels(epoch_file)
    processed_data, labels = prepare_searchlight_mvpa_data(images,
                                                           conditions)
    expected_searchlight_processed_data = np.load(
        expected_dir / 'expected_searchlight_processed_data.npy')
    for idx in range(len(processed_data)):
        assert np.allclose(processed_data[idx],
                           expected_searchlight_processed_data[idx]), (
            'raw data do not match in test_prepare_searchlight_mvpa_data')
    assert np.array_equal(labels, expected_labels), \
        'the labels do not match in test_prepare_searchlight_mvpa_data'
    from brainiak.fcma.preprocessing import RandomType
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    random_processed_data, _ = prepare_searchlight_mvpa_data(
        images,
        conditions,
        random=RandomType.REPRODUCIBLE)
    assert (len(random_processed_data)
            == len(expected_searchlight_processed_data)), (
        'numbers of epochs do not match in test_prepare_searchlight_mvpa_data')
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    random_processed_data, _ = prepare_searchlight_mvpa_data(
        images,
        conditions,
        random=RandomType.UNREPRODUCIBLE)
    assert (len(random_processed_data)
            == len(expected_searchlight_processed_data)), (
        'numbers of epochs do not match in test_prepare_searchlight_mvpa_data')


if __name__ == '__main__':
    test_prepare_fcma_data()
    test_prepare_mvpa_data()
    test_prepare_searchlight_mvpa_data()
