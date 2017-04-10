#  Copyright 2017 Intel Corporation
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
from typing import Sequence

import nibabel as nib
import numpy as np
import pytest

from brainiak import io


@pytest.fixture
def in_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def expected_image_data_shape() -> Sequence[int]:
    return (64, 64, 26, 10)


@pytest.fixture
def mask_file_name() -> str:
    return "mask.nii.gz"


@pytest.fixture
def labels_file_name() -> str:
    return "epoch_labels.npy"


@pytest.fixture
def expected_condition_spec_shape() -> Sequence[int]:
    return (2, 2, 10)


@pytest.fixture
def expected_n_subjects() -> int:
    return 2


def test_load_images_from_dir_data_shape(
        in_dir: str,
        expected_image_data_shape: Sequence[int],
        expected_n_subjects: int
        ) -> None:
    i = 0
    for image in io.load_images_from_dir(in_dir, "bet.nii.gz"):
        assert image.get_data().shape == (64, 64, 26, 10)
        i += 1
    assert i == expected_n_subjects


def test_load_boolean_mask(in_dir: str, mask_file_name: str) -> None:
    mask = io.load_boolean_mask(in_dir / mask_file_name)
    assert mask.dtype == np.bool


def test_load_labels(in_dir: str, labels_file_name: str,
                     expected_condition_spec_shape: Sequence[int],
                     expected_n_subjects: int) -> None:
    condition_specs = io.load_labels(in_dir / labels_file_name)
    i = 0
    for condition_spec in condition_specs:
        assert condition_spec.shape == expected_condition_spec_shape
        i += 1
    assert i == expected_n_subjects


def test_save_as_nifti_file(tmpdir) -> None:
    out_file = str(tmpdir / "nifti.nii")
    shape = (4, 4, 4)
    io.save_as_nifti_file(np.ones(shape), np.eye(4), out_file)
    assert nib.load(out_file).get_data().shape == shape
