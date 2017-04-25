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
"""Generic image functionality."""

__all__ = [
    "ConditionSpec",
    "SingleConditionSpec",
    "mask_image",
    "multimask_images",
]

import itertools

from typing import Iterable, Sequence, Type, TypeVar

import numpy as np

from nibabel.spatialimages import SpatialImage


T = TypeVar("T", bound="MaskedMultiSubjectData")


class MaskedMultiSubjectData(np.ndarray):
    """Array in shape n_voxels, n_trs, n_subjects."""
    @classmethod
    def from_masked_images(cls: Type[T], masked_images: Iterable[np.ndarray],
                           n_sub: int) -> T:
        """Create a new instance from masked images.

        Parameters
        ----------
        masked_images
            Images to concatenate.
        n_sub
            Number of subjects. Must match the number of images.

        Returns
        -------
        T
            A new instance.

        Raises
        ------
        ValueError
            Images have different shapes.

            The number of images differs from n_sub.
        """
        images_iterator = iter(masked_images)
        first_image = next(images_iterator)
        result = np.empty((first_image.shape[0], first_image.shape[1], n_sub))
        for n_images, image in enumerate(itertools.chain([first_image],
                                                         images_iterator)):
            if image.shape != first_image.shape:
                raise ValueError("Image {} has different shape from first "
                                 "image: {} != {}".format(n_images,
                                                          image.shape,
                                                          first_image.shape))
            result[:, :, n_images] = image
        n_images += 1
        if n_images != n_sub:
            raise ValueError("n_sub != number of images: {} != {}"
                             .format(n_sub, n_images))
        return result.view(cls)


class ConditionSpec(np.ndarray):
    """One-hot representation of conditions across epochs and TRs.

    The shape is (n_conditions, n_epochs, n_trs).
    """


class SingleConditionSpec(ConditionSpec):
    """ConditionSpec with a single condition applicable to an epoch."""

    def extract_labels(self) -> np.ndarray:
        """Extract condition labels.

        Returns
        -------
        np.ndarray
            The condition label of each epoch.
        """
        condition_idxs, epoch_idxs, _ = np.where(self)
        _, unique_epoch_idxs = np.unique(epoch_idxs, return_index=True)
        return condition_idxs[unique_epoch_idxs]


def mask_image(image: SpatialImage, mask: np.ndarray, data_type: type = None
               ) -> np.ndarray:
    """Mask image after optionally casting its type.

    Parameters
    ----------
    image
        Image to mask. Can include time as the last dimension.
    mask
        Mask to apply. Must have the same shape as the image data.
    data_type
        Type to cast image to.

    Returns
    -------
    np.ndarray
        Masked image.

    Raises
    ------
    ValueError
        Image data and masks have different shapes.
    """
    image_data = image.get_data()
    if image_data.shape[:3] != mask.shape:
        raise ValueError("Image data and mask have different shapes.")
    if data_type is not None:
        cast_data = image_data.astype(data_type)
    else:
        cast_data = image_data
    return cast_data[mask]


def multimask_images(images: Iterable[SpatialImage],
                     masks: Sequence[np.ndarray], image_type: type = None
                     ) -> Iterable[Sequence[np.ndarray]]:
    """Mask images with multiple masks.

    Parameters
    ----------
    images:
        Images to mask.
    masks:
        Masks to apply.
    image_type:
        Type to cast images to.

    Yields
    ------
    Sequence[np.ndarray]
        For each mask, a masked image.
    """
    for image in images:
        yield [mask_image(image, mask, image_type) for mask in masks]


def mask_images(images: Iterable[SpatialImage], mask: np.ndarray,
                image_type: type = None) -> Iterable[np.ndarray]:
    """Mask images.

    Parameters
    ----------
    images:
        Images to mask.
    mask:
        Mask to apply.
    image_type:
        Type to cast images to.

    Yields
    ------
    np.ndarray
        Masked image.
    """
    for images in multimask_images(images, (mask,), image_type):
        yield images[0]
