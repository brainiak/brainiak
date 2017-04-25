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

from brainiak.image import (mask_image, mask_images, MaskedMultiSubjectData,
                            multimask_images, SingleConditionSpec)


@pytest.fixture
def masked_multi_subject_data(masked_images):
    return np.stack(masked_images, axis=-1)


class TestMaskedMultiSubjectData:
    def test_from_masked_images(self, masked_images,
                                masked_multi_subject_data):
        result = MaskedMultiSubjectData.from_masked_images(masked_images,
                                                           len(masked_images))
        assert np.array_equal(result, masked_multi_subject_data)


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
    return Nifti1Pair(np.array([[[[0, 0, 0, 0],
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
                                  [0, 0, 0, 0]]]]).reshape(4, 4, 4, 1),
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
    return np.array([[1, 0, 0, 1, 0, 1, 1, 0]]).reshape(8, 1)


@pytest.fixture
def images(spatial_image: SpatialImage) -> Iterable[SpatialImage]:
    images = [spatial_image]
    image_data = spatial_image.get_data().copy()
    image_data[1, 1, 1, 0] = 2
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
def multimasked_images(masked_data) -> Iterable[Iterable[np.ndarray]]:
    masked_data_2 = np.concatenate(([[2]], masked_data[1:, :]))
    return [[masked_data, np.concatenate(([[0]], masked_data)),
             masked_data[:-1, :]],
            [masked_data_2, np.concatenate(([[0]], masked_data_2)),
             masked_data_2[:-1, :]]]


@pytest.fixture
def masked_images(multimasked_images) -> Iterable[np.ndarray]:
    return [multimasked_image[0] for multimasked_image in multimasked_images]


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


def test_multimask_images(
        images: Iterable[SpatialImage],
        masks: Sequence[np.ndarray],
        multimasked_images: Iterable[Iterable[np.ndarray]]
        ) -> None:
    result = multimask_images(images, masks)
    for result_images, expected_images in zip(result,
                                              multimasked_images):
        for result_image, expected_image in zip(result_images,
                                                expected_images):
            assert np.array_equal(result_image, expected_image)


def test_mask_images(
        images: Iterable[SpatialImage],
        mask: np.ndarray,
        masked_images: Iterable[np.ndarray]
        ) -> None:
    result = mask_images(images, mask)
    for result_image, expected_image in zip(result, masked_images):
        assert np.array_equal(result_image, expected_image)
