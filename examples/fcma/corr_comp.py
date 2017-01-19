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

import sys
import logging
import numpy as np
from brainiak.fcma.util import compute_correlation
import brainiak.fcma.io as io
import scipy.io

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

# python corr_comp.py face_scene bet.nii.gz face_scene/prefrontal_top_mask.nii.gz face_scene/fs_epoch_labels.npy
if __name__ == '__main__':
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]

    raw_data = io.read_activity_data(data_dir, extension, mask_file)
    epoch_list = np.load(epoch_file)
    epoch_info = io.generate_epochs_info(epoch_list)

    for idx, epoch in enumerate(epoch_info):
        label = epoch[0]
        sid = epoch[1]
        start = epoch[2]
        end = epoch[3]
        mat = raw_data[sid][:, start:end]
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        logger.info(
            'start to compute correlation for subject %d epoch %d with label %d' %
            (sid, idx, label)
        )
        corr = compute_correlation(mat, mat)
        mdict = {}
        mdict['corr'] = corr
        filename = str(label) + '_' + str(sid) + '_' + str(idx)
        logger.info(
            'start to write the correlation matrix to disk as %s' %
            filename
        )
        scipy.io.savemat(filename, mdict)
