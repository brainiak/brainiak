#  Copyright 2018 Intel Corporation
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
"""Example of intersubject correlation (ISC) analysis

Computes ISC for all voxels within a brain mask, and computes
ISFC for voxels with high ISC.

First download the example dataset by running the download_data.sh
script locally (e.g., ./download_data.sh). This download includes
functional data for 5 subjects and a gray-matter anatomical mask.
"""

# Authors: Christopher Baldassano, Sam Nastase, and Mor Regev
# Princeton University, 2018

from os.path import abspath, dirname, join
from brainiak.isc import isc, isfc
import numpy as np
import nibabel as nib
from brainiak import image, io
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage


curr_dir = dirname(abspath("__file__"))

mask_fn = join(curr_dir,'avg152T1_gray_3mm.nii.gz')
func_fns = [join(curr_dir,
                 'sub-{0:03d}-task-intact1.nii.gz'.format(sub))
            for sub in np.arange(1, 6)]

print('Loading data from {0} subjects...'.format(len(func_fns)))

mask_image = io.load_boolean_mask(mask_fn, lambda x: x > 50)
masked_images = image.mask_images(io.load_images(func_fns),
                                  mask_image)
coords = np.where(mask_image)
data = image.MaskedMultiSubjectData.from_masked_images(masked_images,
                                                       len(func_fns))

print('Calculating mean ISC on {0} voxels'.format(data.shape[1]))
iscs = isc(data, pairwise=False, summary_statistic='mean')
iscs = np.nan_to_num(iscs)

print('Writing ISC map to file...')
nii_template = nib.load(mask_fn)
isc_vol = np.zeros(nii_template.shape)
isc_vol[coords] = iscs
isc_image = nib.Nifti1Image(isc_vol, nii_template.affine,
                            nii_template.header)
nib.save(isc_image, 'example_isc.nii.gz')

isc_mask = (iscs > 0.2)[0, :]
print('Calculating mean ISFC on {0} voxels...'.format(np.sum(isc_mask)))
data_masked = data[:, isc_mask, :]
isfcs = isfc(data_masked, pairwise=False, summary_statistic='mean')

print('Clustering ISFC...')
Z = linkage(isfcs, 'ward')
z = fcluster(Z, 2, criterion='maxclust')
clust_inds = np.argsort(z)

# Show the ISFC matrix, sorted to show the two main clusters
plt.imshow(isfcs[np.ix_(clust_inds, clust_inds)])
plt.show()
