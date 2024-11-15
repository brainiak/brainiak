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
from brainiak.fcma.preprocessing import RandomType
from brainiak import io
from sklearn.svm import SVC
import sys
from mpi4py import MPI
import logging
import numpy as np
import nibabel as nib
import os

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

"""
Perform leave one participant out voxel selection with FCMA
"""

data_dir = sys.argv[1]  # What is the directory containing data?
suffix = sys.argv[2]  # What is the extension of the data you're loading
mask_file = sys.argv[3]  # What is the path to the whole brain mask
epoch_file = sys.argv[4]  # What is the path to the epoch file
left_out_subj = sys.argv[5]  # Which participant (as an integer) are you leaving out for this cv?
output_dir = sys.argv[6]  # What is the path to the folder you want to save this data in

# Only run the following from the controller core
if __name__ == '__main__':
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'Testing for participant %d.\nProgramming starts in %d process(es)' %
            (int(left_out_subj), MPI.COMM_WORLD.Get_size())
        )
        # create output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Load in the volumes, mask and labels
    images = io.load_images_from_dir(data_dir, suffix=suffix)
    mask = io.load_boolean_mask(mask_file)
    epoch_list = io.load_labels(epoch_file)

    # Parse the epoch data for useful dimensions
    epochs_per_subj = epoch_list[0].shape[1]
    num_subjs = len(epoch_list)

    # Preprocess the data and prepare for FCMA
    raw_data, _, labels = prepare_fcma_data(images, epoch_list, mask)

    # enforce left one out
    file_str = output_dir + '/fc_no' + str(left_out_subj) + '_'
    start_idx = int(int(left_out_subj) * epochs_per_subj)
    end_idx = int(start_idx + epochs_per_subj)

    # Take out the idxs corresponding to all participants but this one
    subsampled_idx = list(set(range(len(labels))) - set(range(start_idx, end_idx)))
    labels_subsampled = [labels[i] for i in subsampled_idx]
    raw_data_subsampled = [raw_data[i] for i in subsampled_idx]

    # Set up the voxel selection object for fcma
    vs = VoxelSelector(labels_subsampled, epochs_per_subj, num_subjs - 1, raw_data_subsampled)

    # for cross validation, use SVM with precomputed kernel
    clf = SVC(kernel='precomputed', shrinking=False, C=1)
    results = vs.run(clf)

    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'correlation-based voxel selection is done'
        )

        # Load in the mask with nibabel
        mask_img = nib.load(mask_file)
        mask = mask_img.get_fdata().astype(bool)

        # Preset the volumes
        score_volume = np.zeros(mask.shape, dtype=np.float32)
        score = np.zeros(len(results), dtype=np.float32)
        seq_volume = np.zeros(mask.shape, dtype=np.int)
        seq = np.zeros(len(results), dtype=np.int)

        # Write a text document of the voxel selection results
        with open(file_str + 'result_list.txt', 'w') as fp:
            for idx, tuple in enumerate(results):
                fp.write(str(tuple[0]) + ' ' + str(tuple[1]) + '\n')

                # Store the score for each voxel
                score[tuple[0]] = tuple[1]
                seq[tuple[0]] = idx

        # Convert the list into a volume
        score_volume[mask] = score
        seq_volume[mask] = seq

        # Save volume
        io.save_as_nifti_file(score_volume, mask_img.affine,
                                file_str + 'result_score.nii.gz')
        io.save_as_nifti_file(seq_volume, mask_img.affine,
                                file_str + 'result_seq.nii.gz')
