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

from typing import Iterable, Sequence

import numpy as np
import pytest

from nibabel.nifti1 import Nifti1Pair
from nibabel.spatialimages import SpatialImage

from brainiak.image import (mask_image, multimask_images,
                            SingleConditionSpec)


@pytest.fixture
def condition_spec() -> SingleConditionSpec:
    return np.array([[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]]],
                    dtype=np.int8).view(SingleConditionSpec)


class TestUniqueLabelConditionSpec:
    def test_extract_labels(self, condition_spec: SingleConditionSpec
                            ) -> None:
        assert np.array_equal(condition_spec.extract_labels(),
                              np.array([0, 1]))


@pytest.fixture
def spatial_image() -> SpatialImage:
    return Nifti1Pair(np.array([[[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]],
                                [[0, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 0]],
                                [[0, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 0]],
                                [[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]]]),
                      np.eye(4))


@pytest.fixture
def mask() -> np.ndarray:
    return np.array([[[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]],
                     [[0, 0, 0, 0],
                      [0, 1, 1, 0],
                      [0, 1, 1, 0],
                      [0, 0, 0, 0]],
                     [[0, 0, 0, 0],
                      [0, 1, 1, 0],
                      [0, 1, 1, 0],
                      [0, 0, 0, 0]],
                     [[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]], dtype=np.bool)


@pytest.fixture
def masked_data() -> np.ndarray:
    return np.array([1, 0, 0, 1, 0, 1, 1, 0])


@pytest.fixture
def images(spatial_image: SpatialImage) -> Iterable[SpatialImage]:
    images = [spatial_image]
    image_data = spatial_image.get_data().copy()
    image_data[1, 1, 1] = 2
    images.append(Nifti1Pair(image_data, np.eye(4)))
    return images


@pytest.fixture
def masks(mask: np.ndarray) -> Sequence[np.ndarray]:
    masks = [mask]
    mask2 = mask.copy()
    mask2[0, 0, 0] = 1
    masks.append(mask2)
    mask3 = mask.copy()
    mask3[2, 2, 2] = 0
    masks.append(mask3)
    return masks


@pytest.fixture
def multimasked_data(masked_data) -> Iterable[Iterable[np.ndarray]]:
    masked_data_2 = np.hstack((2, masked_data[1:]))
    return [[masked_data, masked_data_2],
            [np.hstack((0, masked_data)), np.hstack((0, masked_data_2))],
            [masked_data[:-1], masked_data_2[:-1]]]


def test_mask_image(spatial_image: SpatialImage, mask: np.ndarray,
                    masked_data: np.ndarray) -> None:
    result = mask_image(spatial_image, mask)
    assert np.array_equal(result, masked_data)


def test_mask_image_with_type(spatial_image: SpatialImage, mask: np.ndarray,
                              masked_data: np.ndarray) -> None:
    masked_data_type = np.float32
    result = mask_image(spatial_image, mask, masked_data_type)
    assert result.dtype == masked_data_type
    assert np.allclose(result, masked_data)


def test_multimask_images(images: Iterable[SpatialImage],
                          masks: Sequence[np.ndarray],
                          multimasked_data: Iterable[Iterable[np.ndarray]]
                          ) -> None:
    result = multimask_images(images, masks)
    for mask_data in zip(result, multimasked_data):
        for result_data, precomputed_data in zip(mask_data[0], mask_data[1]):
            assert np.array_equal(result_data, precomputed_data)
