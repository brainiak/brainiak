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
import brainiak.fcma.io as io
from sklearn import svm
import sys
from mpi4py import MPI
import logging
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
    mask = mask_img.get_data().astype(np.bool)
    epoch_info = None
    if MPI.COMM_WORLD.Get_rank()==0:
        raw_data = io.read_activity_data(data_dir, extension)
        epoch_list = np.load(epoch_file)
        epoch_info = io.generate_epochs_info(epoch_list)
        # the following line is an example to leaving a subject out
        #epoch_info = [x for x in epoch_info if x[1] != 0]
    num_subjs = int(sys.argv[5])
    # create a Searchlight object
    sl = Searchlight(sl_rad=1)
    mvs = MVPAVoxelSelector(raw_data, mask, epoch_info, num_subjs, sl)
    clf = svm.SVC(kernel='linear', shrinking=False, C=1)
    # only rank 0 has meaningful return values
    score_volume, results = mvs.run(clf)
    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        score_volume = np.nan_to_num(score_volume.astype(np.float))
        io.write_nifti_file(score_volume, mask_img.affine, 'result_score.nii.gz')
        seq_volume = np.zeros(mask.shape, dtype=np.int)
        seq = np.zeros(len(results), dtype=np.int)
        with open('result_list.txt', 'w') as fp:
            for idx, tuple in enumerate(results):
                fp.write(str(tuple[0]) + ' ' + str(tuple[1]) + '\n')
                seq[tuple[0]] = idx
        seq_volume[mask] = seq
        io.write_nifti_file(seq_volume, mask_img.affine, 'result_seq.nii.gz')
