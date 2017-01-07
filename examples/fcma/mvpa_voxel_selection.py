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
import nibabel as nib
import numpy as np

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

"""
example running command:
mpirun -np 2 python mvpa_voxel_selection.py /Users/yidawang/data/face_scene/raw nii.gz /Users/yidawang/data/face_scene/mask.nii.gz
                        data/fs_epoch_labels.npy 18
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
    mask = None
    epoch_info = None
    if MPI.COMM_WORLD.Get_rank()==0:
        mask_img = nib.load(mask_file)
        mask = mask_img.get_data()
        count = 0
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
    mvs = MVPAVoxelSelector(raw_data, mask, epoch_info, num_subjs)
    # for cross validation, use SVM with precomputed kernel
    # no shrinking, set C=10
    clf = svm.SVC(kernel='rbf', shrinking=False, C=10)
    results = mvs.run(clf)
    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        print(results[0:100])
