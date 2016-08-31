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

from brainiak.fcma.fcma import *
from sklearn import svm
import sys
from mpi4py import MPI
"""
example running command:
mpirun -np 2 python run.py /Users/yidawang/data/face_scene/raw nii.gz /Users/yidawang/data/face_scene/mask.nii.gz
                        data/fs_epoch_labels.npy 12 18
"""
if __name__ == '__main__':
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]
    raw_data, labels = prepareData(data_dir, extension, mask_file, epoch_file)
    epochs_per_subj = int(sys.argv[5])
    num_subjs = int(sys.argv[6])
    vs = VoxelSelector(raw_data, epochs_per_subj, labels, num_subjs)
    # for cross validation, use SVM with precomputed kernel
    # no shrinking, set C=10
    clf = svm.SVC(kernel='precomputed', shrinking=False, C=10)
    results = vs.run(clf)
    if MPI.COMM_WORLD.Get_rank()==0:
        print(results[0:100])
