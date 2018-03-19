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
import logging
from brainiak.fcma.preprocessing import prepare_mvpa_data
from brainiak import io
import numpy as np
from scipy.spatial.distance import hamming
from sklearn import model_selection
#from sklearn.externals import joblib

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

# python3 mvpa_classification.py face_scene bet.nii.gz face_scene/visual_top_mask.nii.gz face_scene/fs_epoch_labels.npy
if __name__ == '__main__':
    if len(sys.argv) != 5:
        logger.error('the number of input argument is not correct')
        sys.exit(1)

    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]

    epoch_list = np.load(epoch_file)
    num_subjects = len(epoch_list)
    num_epochs_per_subj = epoch_list[0].shape[1]

    logger.info(
        'doing MVPA training and classification on %d subjects, each of which has %d epochs' %
        (num_subjects, num_epochs_per_subj)
    )

    images = io.load_images_from_dir(data_dir, extension)
    mask = io.load_boolean_mask(mask_file)
    conditions = io.load_labels(epoch_file)
    processed_data, labels = prepare_mvpa_data(images, conditions, mask)

    # transpose data to facilitate training and prediction
    processed_data = processed_data.T

    clf = svm.SVC(kernel='linear', shrinking=False, C=1)
    # doing leave-one-subject-out cross validation
    for i in range(num_subjects):
        leave_start = i * num_epochs_per_subj
        leave_end = (i+1) * num_epochs_per_subj
        training_data = np.concatenate((processed_data[0:leave_start], processed_data[leave_end:]), axis=0)
        test_data = processed_data[leave_start:leave_end]
        training_labels = np.concatenate((labels[0:leave_start], labels[leave_end:]), axis=0)
        test_labels = labels[leave_start:leave_end]
        clf.fit(training_data, training_labels)
        # joblib can be used for saving and loading models
        #joblib.dump(clf, 'model/logistic.pkl')
        #clf = joblib.load('model/svm.pkl')
        predict = clf.predict(test_data)
        print(predict)
        print(clf.decision_function(test_data))
        print(np.asanyarray(test_labels))
        incorrect_predict = hamming(predict, np.asanyarray(test_labels)) * num_epochs_per_subj
        logger.info(
            'when leaving subject %d out for testing, the accuracy is %d / %d = %.2f' %
            (i, num_epochs_per_subj-incorrect_predict, num_epochs_per_subj,
             (num_epochs_per_subj-incorrect_predict) * 1.0 / num_epochs_per_subj)
        )

    # use model selection
    # no shuffling in cv
    skf = model_selection.StratifiedKFold(n_splits=num_subjects,
                                          shuffle=False)
    scores = model_selection.cross_val_score(clf, processed_data,
                                             y=labels,
                                             cv=skf)
    print(scores)
    logger.info(
        'the overall cross validation accuracy is %.2f' %
        np.mean(scores)
    )
    logger.info('MVPA training and classification done')
