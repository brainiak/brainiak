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

import numpy as np
import brainiak.fcma.io as io
import nibabel as nib
import os

dir = os.path.join(os.path.dirname(__file__), 'data')
extension = 'bet.nii.gz'
mask_file = os.path.join(os.path.dirname(__file__), 'data/mask.nii.gz')
epoch_file = os.path.join(os.path.dirname(__file__), 'data/epoch_labels.npy')
expected_labels = np.array([0, 1, 0, 1])

def test_read_activity_data():
    masked_data1 = io.read_activity_data(dir, extension, mask_file)
    raw_data = io.read_activity_data(dir, extension)
    mask_img = nib.load(mask_file)
    mask = mask_img.get_data().astype(np.bool)
    masked_data2 = []
    for data in raw_data:
        masked_data2.append(data[mask])
    assert len(masked_data1) == len(masked_data2), \
        'numbers of subjects do not match in test_read_activity_data'
    for idx in range(len(masked_data1)):
        assert np.allclose(masked_data1[idx], masked_data2[idx]), \
            'masked data do not match in test_read_activity_data'

def test_prepare_fcma_data():
    raw_data, labels = io.prepare_fcma_data(dir, extension, mask_file, epoch_file)
    expected_raw_data = np.load(os.path.join(os.path.dirname(__file__),
                                             'data/expected_raw_data.npy'))
    assert len(raw_data) == len(expected_raw_data), \
        'numbers of epochs do not match in test_prepare_fcma_data'
    for idx in range(len(raw_data)):
        assert np.allclose(raw_data[idx], expected_raw_data[idx]), \
            'raw data do not match in test_prepare_fcma_data'
    assert np.array_equal(labels, expected_labels), \
        'the labels do not match in test_prepare_fcma_data'

def test_prepare_mvpa_data():
    processed_data, labels = io.prepare_mvpa_data(dir, extension, mask_file, epoch_file)
    expected_processed_data = np.load(os.path.join(os.path.dirname(__file__),
                                                   'data/expected_processed_data.npy'))
    assert len(processed_data) == len(expected_processed_data), \
        'numbers of epochs do not match in test_prepare_mvpa_data'
    for idx in range(len(processed_data)):
        assert np.allclose(processed_data[idx], expected_processed_data[idx]), \
            'raw data do not match'
    assert np.array_equal(labels, expected_labels), \
        'the labels do not match in test_prepare_mvpa_data'

if __name__ == '__main__':
    test_read_activity_data()
    test_prepare_fcma_data()
    test_prepare_mvpa_data()
