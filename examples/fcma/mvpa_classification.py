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

from sklearn import svm
#from sklearn.linear_model import LogisticRegression
import sys
import os
import logging
from file_io import generate_epochs_info
import numpy as np
from scipy.stats.mstats import zscore
import nibabel as nib
#from sklearn.externals import joblib

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)


def prepare_data(raw_data, epoch_info):
    """ prepare the data for model training and prediction

    Average the activity within epochs and z-scoring within subject.

    Parameters
    ----------
    raw\_data: list of 2D array
        each element of the list is the raw data of a subject,
        in the shape of [num_voxels, num_TRs]
        Assuming all data have the same num_voxels

    epoch\_info: list of tuple (label, sid, start, end).
        label is the condition labels of the epochs;
        sid is the subject id, corresponding to the index of raw_data;
        start is the start TR of an epoch (inclusive);
        end is the end TR of an epoch(exclusive).
        Assuming len(labels) labels equals the number of epochs and
        the epochs of the same sid are adjacent in epoch_info

    Returns
    -------
    processed\_data: 2D array in shape [num_voxels, num_epochs]
        averaged epoch by epoch processed data

    labels: 1D array
        contains labels of the data
    """
    num_epochs = len(epoch_info)
    (d1, _) = raw_data[0].shape
    processed_data = np.empty([d1, num_epochs])
    labels = np.empty(num_epochs)
    subject_count = [0]  # counting the epochs per subject for z-scoring
    cur_sid = -1
    # averaging
    for idx, epoch in enumerate(epoch_info):
        labels[idx] = epoch[0]
        if cur_sid != epoch[1]:
            subject_count.append(0)
            cur_sid = epoch[1]
        subject_count[-1] += 1
        processed_data[:, idx] = \
            np.mean(raw_data[cur_sid][:, epoch[2]:epoch[3]],
                    axis=1)
    # z-scoring
    cur_epoch = 0
    for i in subject_count:
        if i > 1:
            processed_data[:, cur_epoch:cur_epoch + i] = \
                zscore(processed_data[:, cur_epoch:cur_epoch + i],
                       axis=1, ddof=0)
        cur_epoch += i
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    processed_data = np.nan_to_num(processed_data)

    return processed_data, labels

# python mvpa_classification.py face_scene bet.nii.gz face_scene/visual_top_mask.nii.gz face_scene/fs_epoch_labels.npy 18
if __name__ == '__main__':
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]

    raw_data = []
    mask_img = nib.load(mask_file)
    mask = mask_img.get_data()
    mask = mask.astype(np.bool)
    logger.info(
        'mask size: %d' %
        np.sum(mask)
    )
    files = [f for f in sorted(os.listdir(data_dir))
             if os.path.isfile(os.path.join(data_dir, f))
             and f.endswith(extension)]
    for f in files:
        img = nib.load(os.path.join(data_dir, f))
        data = img.get_data()
        # apply mask
        data = data[mask, :]
        raw_data.append(data)
        logger.info(
            'file %s is loaded, with data shape %s' % (f, data.shape)
        )
    epoch_list = np.load(epoch_file)
    epoch_info = generate_epochs_info(epoch_list)

    num_subjs = int(sys.argv[5])

    # preparing data
    processed_data, labels = prepare_data(raw_data, epoch_info)

    # transpose data to facilitate training and prediction
    processed_data = processed_data.T

    # no shrinking, set C=10
    clf = svm.SVC(kernel='rbf', shrinking=False, C=10)
    training_data = processed_data[0:204]
    test_data = processed_data[204:]
    clf.fit(training_data, labels[0:204])
    # joblib can be used for saving and loading models
    #joblib.dump(clf, 'model/logistic.pkl')
    #clf = joblib.load('model/svm.pkl')
    print(clf.predict(test_data))
    print(clf.decision_function(test_data))
    print(np.asanyarray(labels[204:]))
