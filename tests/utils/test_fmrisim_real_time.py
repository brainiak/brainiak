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

"""fmrisim real-time generator test script

 Authors: Cameron Ellis (Princeton) 2020
"""
import numpy as np
from brainiak.utils import fmrisim_real_time_generator as gen
from importlib.resources import files
import pytest
import os
import time
import glob

from typing import Dict
from nibabel.nifti1 import Nifti1Image
import gzip

# Test that it crashes without inputs
with pytest.raises(TypeError):
    gen.generate_data()  # type: ignore

data_dict: Dict = {}

rf = files('brainiak').joinpath('utils/sim_parameters/ROI_A.nii.gz')
with rf.open("rb") as f:
    vol = f.read()
data_dict["ROI_A_file"] = np.asanyarray(
    Nifti1Image.from_bytes(gzip.decompress(vol)).dataobj
)

rf = files('brainiak').joinpath('utils/sim_parameters/ROI_B.nii.gz')
with rf.open("rb") as f:
    vol = f.read()
data_dict["ROI_B_file"] = np.asanyarray(
    Nifti1Image.from_bytes(gzip.decompress(vol)).dataobj
)
rf = files('brainiak').joinpath('utils/sim_parameters/sub_template.nii.gz')
with rf.open("rb") as f:
    vol = f.read()
data_dict["template_path"] = np.asanyarray(
    Nifti1Image.from_bytes(gzip.decompress(vol)).dataobj
)

rf = files('brainiak').joinpath('utils/sim_parameters/sub_noise_dict.txt')
with rf.open("rb") as f:
    noise_dict_file = f.read()

data_dict['noise_dict_file'] = noise_dict_file
data_dict['numTRs'] = 30
data_dict['event_duration'] = 2
data_dict['scale_percentage'] = 1
data_dict['different_ROIs'] = True
data_dict['multivariate_pattern'] = False
data_dict['save_dicom'] = False
data_dict['save_realtime'] = False
data_dict['trDuration'] = 2
data_dict['isi'] = 4
data_dict['burn_in'] = 6


# Run default test
def test_default(tmp_path, dd=data_dict):

    # copy data_dict so values aren't changed
    dd = dd.copy()

    # Run the simulation
    gen.generate_data(str(tmp_path),
                      dd)

    # Check that there are 32 files where there should be (30 plus label and
    # mask)
    assert len(os.listdir(str(tmp_path))) == 32, "Incorrect file number"

    # Check that the data is the right shape
    input_template = dd['template_path']
    input_shape = input_template.shape
    output_vol = np.load(tmp_path / 'rt_000.npy')
    output_shape = output_vol.shape
    assert input_shape == output_shape, 'Output shape is incorrect'

    # Check the labels have the correct count
    labels = np.load(tmp_path / 'labels.npy')

    assert np.sum(labels > 0) == 9, 'Incorrect number of events'


def test_signal_size(tmp_path, dd=data_dict):

    dd = dd.copy()

    # Change it to only use ROI A
    dd['different_ROIs'] = False

    # Make the signal large
    dd['scale_percentage'] = 100

    # Run the simulation
    gen.generate_data(str(tmp_path),
                      dd)

    # Load in the ROI masks
    ROI_A = dd['ROI_A_file']
    ROI_B = dd['ROI_B_file']

    # Load in the data just simulated
    ROI_A_mean = []
    ROI_B_mean = []
    for TR_counter in range(dd['numTRs']):

        # Load the data
        vol_name = 'rt_%03d.npy' % TR_counter
        vol = np.load(tmp_path / vol_name)

        # Mask the data
        ROI_A_mean += [np.mean(vol[ROI_A == 1])]
        ROI_B_mean += [np.mean(vol[ROI_B == 1])]

    assert np.std(ROI_A_mean) > np.std(ROI_B_mean), 'Signal not scaling'


def test_multivariate(tmp_path, dd=data_dict):

    dd = dd.copy()

    dd['multivariate_pattern'] = True
    dd['different_ROIs'] = False

    # Make the signal large
    dd['scale_percentage'] = 100

    # Run the simulation
    gen.generate_data(str(tmp_path),
                      dd)

    # Load in the ROI masks
    ROI_A = dd['ROI_A_file']
    ROI_B = dd['ROI_B_file']

    # Test this volume
    vol = np.load(str(tmp_path / 'rt_007.npy'))

    ROI_A_std = np.std(vol[ROI_A == 1])
    ROI_B_std = np.std(vol[ROI_B == 1])

    assert ROI_A_std > ROI_B_std, 'Multivariate not making variable signal'


def test_save_dicoms_realtime(tmp_path, dd=data_dict):

    dd = dd.copy()
    start_time = time.time()

    dd['save_dicom'] = True
    dd['save_realtime'] = True

    # test when ROI files are not set
    dd['ROI_A_file'] = None
    dd['ROI_B_file'] = None
    dd['template_path'] = None
    dd['noise_dict_file'] = None

    # Run the simulation
    gen.generate_data(str(tmp_path),
                      dd)

    end_time = time.time()

    # Check it took 2s per TR
    assert (end_time - start_time) > 60, 'Realtime ran fast'

    # Check correct file number
    file_path = str(tmp_path / '*.dcm')
    assert len(glob.glob(file_path)) == 30, "Wrong dicom file num"
