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
from brainiak.fcma.preprocessing import prepare_searchlight_mvpa_data
from brainiak import io
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
    if len(sys.argv) != 6:
        logger.error('the number of input argument is not correct')
        sys.exit(1)

    data_dir = sys.argv[1]
    suffix = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]

    # all MPI processes read the mask; the mask file is small
    mask_image = nib.load(mask_file)
    mask = io.load_boolean_mask(mask_file)
    data = None
    labels = None
    if MPI.COMM_WORLD.Get_rank()==0:
        logger.info(
            'mask size: %d' %
            np.sum(mask)
        )
        images = io.load_images_from_dir(data_dir, suffix=suffix)
        conditions = io.load_labels(epoch_file)
        data, labels = prepare_searchlight_mvpa_data(images, conditions)

        # setting the random argument produces random voxel selection results
        # for non-parametric statistical analysis.
        # There are three random options:
        # RandomType.NORANDOM is the default
        # RandomType.REPRODUCIBLE permutes the voxels in the same way every run
        # RandomType.UNREPRODUCIBLE permutes the voxels differently across runs
        # example
        #from brainiak.fcma.preprocessing import RandomType
        #data, labels = prepare_searchlight_mvpa_data(images, conditions,
        #                                                    random=RandomType.UNREPRODUCIBLE)

        # the following line is an example to leaving a subject out
        #epoch_info = [x for x in epoch_info if x[1] != 0]

    num_subjs = int(sys.argv[5])
    # create a Searchlight object
    sl = Searchlight(sl_rad=1)
    mvs = MVPAVoxelSelector(data, mask, labels, num_subjs, sl)
    clf = svm.SVC(kernel='linear', shrinking=False, C=1)
    # only rank 0 has meaningful return values
    score_volume, results = mvs.run(clf)
    # this output is just for result checking
    if MPI.COMM_WORLD.Get_rank()==0:
        score_volume = np.nan_to_num(score_volume.astype(np.float))
        io.save_as_nifti_file(score_volume, mask_image.affine,
                                   'result_score.nii.gz')
        seq_volume = np.zeros(mask.shape, dtype=np.int)
        seq = np.zeros(len(results), dtype=np.int)
        with open('result_list.txt', 'w') as fp:
            for idx, tuple in enumerate(results):
                fp.write(str(tuple[0]) + ' ' + str(tuple[1]) + '\n')
                seq[tuple[0]] = idx
        seq_volume[mask] = seq
        io.save_as_nifti_file(seq_volume, mask_image.affine,
                                   'result_seq.nii.gz')
