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

from brainiak.fcma.mvpa_voxelselector import MVPAVoxelSelector
from sklearn import svm
import sys
import os
from mpi4py import MPI
import logging
from file_io import generate_epochs_info
from file_io import write_nifti_file
import nibabel as nib
import numpy as np
from brainiak.searchlight.searchlight import Searchlight

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

"""
example running command in run_mvpa_voxel_selection.sh
"""
if __name__ == '__main__':
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'programming starts in %d process(es)' %
            MPI.COMM_WORLD.Get_size()
        )
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]

    raw_data = []
    # all MPI processes read the mask; the mask file is small
    mask_img = nib.load(mask_file)
    mask = mask_img.get_data()
    epoch_info = None
    if MPI.COMM_WORLD.Get_rank()==0:
        files = [f for f in sorted(os.listdir(data_dir))
                if os.path.isfile(os.path.join(data_dir, f))
                and f.endswith(extension)]
        for f in files:
            img = nib.load(os.path.join(data_dir, f))
            data = img.get_data()
            raw_data.append(data)
            logger.info(
            'file %s is loaded, with data shape %s' % (f, data.shape)
            )
        epoch_list = np.load(epoch_file)
        epoch_info = generate_epochs_info(epoch_list)
    num_subjs = int(sys.argv[5])
    # create a Searchlight object
    sl = Searchlight(sl_rad=2)
    mvs = MVPAVoxelSelector(raw_data, mask, epoch_info, num_subjs, sl)
    # for cross validation, use SVM with precomputed kernel
    # no shrinking, set C=10
    clf = svm.SVC(kernel='rbf', shrinking=False, C=10)
    # only rank 0 has meaningful return values
    result_volume, results = mvs.run(clf)
    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        result_volume = np.nan_to_num(result_volume.astype(np.float))
        write_nifti_file(result_volume, mask_img.affine, 'result.nii.gz')
