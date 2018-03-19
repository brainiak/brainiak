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
"""Example of intersubject analyses (ISC/ISFC)

Computes ISC for all voxels within a brain mask, and
computes ISFC for voxels with high ISC
"""

# Authors: Christopher Baldassano and Mor Regev
# Princeton University, 2017

import brainiak.isfc
from brainiak import image, io
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
import sys
import os

curr_dir = os.path.dirname(__file__)

brain_fname = os.path.join(curr_dir,'avg152T1_gray_3mm.nii.gz')
fnames = [os.path.join(curr_dir,
          'sub-0' + format(subj, '02') + '-task-intact1.nii.gz') for
          subj in np.arange(1, 5)]

print('Loading data from ', len(fnames), ' subjects...')

brain_mask = io.load_boolean_mask(brain_fname, lambda x: x > 50)
masked_images = image.mask_images(io.load_images(fnames), brain_mask)
coords = np.where(brain_mask)
D = image.MaskedMultiSubjectData.from_masked_images(masked_images, len(fnames))

print('Calculating ISC on ', D.shape[0], ' voxels')
ISC = brainiak.isfc.isc(D)
ISC[np.isnan(ISC)] = 0

print('Writing ISC map to file...')
brain_nii = nib.load(brain_fname)
ISC_vol = np.zeros(brain_nii.shape)
ISC_vol[coords] = ISC
ISC_nifti = nib.Nifti1Image(ISC_vol, brain_nii.affine, brain_nii.header)
nib.save(ISC_nifti, 'ISC.nii.gz')

ISC_mask = ISC > 0.2
print('Calculating ISFC on ', np.sum(ISC_mask), ' voxels...')
D_masked = D[ISC_mask, :, :]
ISFC = brainiak.isfc.isfc(D_masked)

print('Clustering ISFC...')
Z = linkage(ISFC, 'ward')
z = fcluster(Z, 2, criterion='maxclust')
clust_inds = np.argsort(z)

# Show the ISFC matrix, sorted to show the two main clusters
plt.imshow(ISFC[np.ix_(clust_inds,clust_inds)])
plt.show()
