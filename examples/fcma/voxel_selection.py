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
from brainiak.fcma.preprocessing import prepare_fcma_data
from brainiak import io
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
    if len(sys.argv) != 7:
        logger.error('the number of input argument is not correct')
        sys.exit(1)
    data_dir = sys.argv[1]
    suffix = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    mask = io.load_boolean_mask(mask_file)
    conditions = io.load_labels(epoch_file)
    raw_data, _, labels = prepare_fcma_data(images, conditions, mask)

    # setting the random argument produces random voxel selection results
    # for non-parametric statistical analysis.
    # There are three random options:
    # RandomType.NORANDOM is the default
    # RandomType.REPRODUCIBLE permutes the voxels in the same way every run
    # RandomType.UNREPRODUCIBLE permutes the voxels differently across runs
    # example:
    # from brainiak.fcma.preprocessing import RandomType
    # raw_data, _, labels = prepare_fcma_data(images, conditions, mask,
    #                                         random=RandomType.REPRODUCIBLE)

    # if providing two masks, just append the second mask as the last input argument
    # and specify raw_data2
    # example:
    # images = io.load_images_from_dir(data_dir, extension)
    # mask2 = io.load_boolean_mask('face_scene/mask.nii.gz')
    # raw_data, raw_data2, labels = prepare_fcma_data(images, conditions, mask,
    #                                                 mask2)

    epochs_per_subj = int(sys.argv[5])
    num_subjs = int(sys.argv[6])
    # the following line is an example to leaving a subject out
    #vs = VoxelSelector(labels[0:204], epochs_per_subj, num_subjs-1, raw_data[0:204])
    # if using all subjects
    vs = VoxelSelector(labels, epochs_per_subj, num_subjs, raw_data)
    # if providing two masks, just append raw_data2 as the last input argument
    #vs = VoxelSelector(labels, epochs_per_subj, num_subjs, raw_data, raw_data2=raw_data2)
    # for cross validation, use SVM with precomputed kernel
    clf = svm.SVC(kernel='precomputed', shrinking=False, C=10)
    results = vs.run(clf)
    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'correlation-based voxel selection is done'
        )
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
        io.save_as_nifti_file(score_volume, mask_img.affine,
                              'result_score.nii.gz')
        io.save_as_nifti_file(seq_volume, mask_img.affine,
                                   'result_seq.nii.gz')
