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

from brainiak.fcma.classifier import Classifier
from sklearn import svm
import sys
import logging
from file_io import prepareData

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)


# python classification.py /Users/yidawang/data/face_scene/raw nii.gz
#     /Users/yidawang/data/face_scene/prefrontal_top_mask.nii.gz data/fs_epoch_labels.npy 12
if __name__ == '__main__':
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]
    raw_data, labels = prepareData(data_dir, extension, mask_file, epoch_file)
    epochs_per_subj = int(sys.argv[5])
    # no shrinking, set C=1
    use_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    clf = Classifier(epochs_per_subj, use_clf)
    clf.fit(raw_data, labels)
