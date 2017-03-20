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

from typing import Iterable, List

import numpy as np

from nibabel.spatialimages import SpatialImage


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
        Image to mask.
    mask
        Mask to apply.
    data_type
        Type to cast image to.

    Returns
    -------
    np.ndarray
        Masked image.
    """
    image_data = image.get_data()
    if data_type is not None:
        cast_data = image_data.astype(data_type)
    else:
        cast_data = image_data
    return cast_data[mask]


def multimask_images(images: Iterable[SpatialImage],
                     masks: Iterable[np.ndarray], image_type: type = None
                     ) -> List[List[np.ndarray]]:
    """Mask images with multiple masks.

    Parameters
    ----------
    images:
        Images to mask.
    masks:
        Masks to apply.
    image_type:
        Type to cast images to.

    Returns
    -------
    List[List[np.ndarray]]
        For each mask, a list of masked images.
    """
    masked_images = [[] for _ in range(len(masks))]
    for image in images:
        for i, mask in enumerate(masks):
            masked_images[i].append(mask_image(image, mask, image_type))
    return masked_images
