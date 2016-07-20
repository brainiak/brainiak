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
import scipy.io
from scipy.stats import stats
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker
import os
import logging
import sys
import click

@click.command()
@click.argument('nifti-file', type=click.File('rb'))
@click.argument('out-file', type=click.File('wb'))
@click.option('--mask-file', default=None, type=click.File('rb'), help='The mask file to get ROI')
@click.option('--zscore', is_flag=True, help='to zscore fMRI data')
@click.option('--detrend', is_flag=True, help='to detrend fMRI data')
@click.option('--smoothing-fwmw', default=False, type=float, help='the spatial smooth window size')



def extract_data(nifti_file, mask_file, out_file, zscore, detrend, smoothing_fwmw):
    if mask_file is None:
        #whole brain, get coordinate info from nifti_file itself
        mask = nib.load(nifti_file.name)
    else:
        mask = nib.load(mask_file.name)
    affine = mask.get_affine()
    if mask_file is None:
        mask_data = mask.get_data()
        if mask_data.ndim == 4:
            #get mask in 3D
            img_data_type = mask.header.get_data_dtype()
            n_tr = mask_data.shape[3]
            mask_data = mask_data[:,:,:,n_tr//2].astype(bool)
            mask = nib.Nifti1Image(mask_data.astype(img_data_type), affine)
        else:
            mask_data = mask_data.astype(bool)
    else:
        mask_data = mask.get_data().astype(bool)

    #get voxel coordinates
    R = np.float64(np.argwhere(mask_data))

    #get scanner RAS coordinates based on voxel coordinates
    if affine is not []:
        R = (np.dot(affine[:3,:3], R.T) + affine[:3,3:4]).T

    #get ROI data, and run preprocessing
    nifti_masker = NiftiMasker(mask_img=mask, standardize=zscore, detrend=detrend, smoothing_fwhm=smoothing_fwmw)
    img = nib.load(nifti_file.name)
    all_images = np.float64(nifti_masker.fit_transform(img))
    data = all_images.T.copy()

    #save data
    subj_data = {'data': data, 'R': R}
    scipy.io.savemat(out_file.name, subj_data)


if __name__ == '__main__':
    extract_data()

