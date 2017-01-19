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

from brainiak.fcma.voxelselector import VoxelSelector
from brainiak.fcma.io import prepare_fcma_data
from brainiak.fcma.io import write_nifti_file
from sklearn import svm
import sys
from mpi4py import MPI
import logging
import numpy as np
import nibabel as nib

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

"""
example running command in run_voxel_selection.sh
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
    raw_data, labels = prepare_fcma_data(data_dir, extension, mask_file, epoch_file)
    epochs_per_subj = int(sys.argv[5])
    num_subjs = int(sys.argv[6])
    # the following line is an example to leaving a subject out
    #vs = VoxelSelector(raw_data[0:204], epochs_per_subj, labels[0:204], num_subjs-1)
    # if using all subjects
    vs = VoxelSelector(raw_data, epochs_per_subj, labels, num_subjs)
    # for cross validation, use SVM with precomputed kernel
    clf = svm.SVC(kernel='precomputed', shrinking=False, C=10)
    results = vs.run(clf)
    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        #print(results[0:100])
        mask_img = nib.load(mask_file)
        mask = mask_img.get_data().astype(np.bool)
        score_volume = np.zeros(mask.shape, dtype=np.float32)
        score = np.zeros(len(results), dtype=np.float32)
        seq_volume = np.zeros(mask.shape, dtype=np.int)
        seq = np.zeros(len(results), dtype=np.int)
        with open('result_list.txt', 'w') as fp:
            for idx, tuple in enumerate(results):
                fp.write(str(tuple[0]) + ' ' + str(tuple[1]) + '\n')
                score[tuple[0]] = tuple[1]
                seq[tuple[0]] = idx
        score_volume[mask] = score
        seq_volume[mask] = seq
        write_nifti_file(score_volume, mask_img.affine, 'result_score.nii.gz')
        write_nifti_file(seq_volume, mask_img.affine, 'result_seq.nii.gz')


