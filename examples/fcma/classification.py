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
from brainiak.fcma.preprocessing import prepare_fcma_data
from brainiak.io import dataset
from sklearn import svm
#from sklearn.linear_model import LogisticRegression
import sys
import logging
import numpy as np
from scipy.spatial.distance import hamming
from sklearn import model_selection
#from sklearn.externals import joblib

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)

def example_of_aggregating_sim_matrix(raw_data, labels, num_subjects, num_epochs_per_subj):
    # aggregate the kernel matrix to save memory
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    clf = Classifier(svm_clf, num_processed_voxels=1000, epochs_per_subj=num_epochs_per_subj)
    rearranged_data = raw_data[num_epochs_per_subj:] + raw_data[0:num_epochs_per_subj]
    rearranged_labels = labels[num_epochs_per_subj:] + labels[0:num_epochs_per_subj]
    clf.fit(list(zip(rearranged_data, rearranged_data)), rearranged_labels,
            num_training_samples=num_epochs_per_subj*(num_subjects-1))
    predict = clf.predict()
    print(predict)
    print(clf.decision_function())
    test_labels = labels[0:num_epochs_per_subj]
    incorrect_predict = hamming(predict, np.asanyarray(test_labels)) * num_epochs_per_subj
    logger.info(
        'when aggregating the similarity matrix to save memory, '
        'the accuracy is %d / %d = %.2f' %
        (num_epochs_per_subj-incorrect_predict, num_epochs_per_subj,
         (num_epochs_per_subj-incorrect_predict) * 1.0 / num_epochs_per_subj)
    )
    # when the kernel matrix is computed in portion, the test data is already in
    print(clf.score(None, test_labels))

def example_of_cross_validation_with_detailed_info(raw_data, labels, num_subjects, num_epochs_per_subj):
    # no shrinking, set C=1
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    #logit_clf = LogisticRegression()
    clf = Classifier(svm_clf, epochs_per_subj=num_epochs_per_subj)
    # doing leave-one-subject-out cross validation
    for i in range(num_subjects):
        leave_start = i * num_epochs_per_subj
        leave_end = (i+1) * num_epochs_per_subj
        training_data = raw_data[0:leave_start] + raw_data[leave_end:]
        test_data = raw_data[leave_start:leave_end]
        training_labels = labels[0:leave_start] + labels[leave_end:]
        test_labels = labels[leave_start:leave_end]
        clf.fit(list(zip(training_data, training_data)), training_labels)
        # joblib can be used for saving and loading models
        #joblib.dump(clf, 'model/logistic.pkl')
        #clf = joblib.load('model/svm.pkl')
        predict = clf.predict(list(zip(test_data, test_data)))
        print(predict)
        print(clf.decision_function(list(zip(test_data, test_data))))
        incorrect_predict = hamming(predict, np.asanyarray(test_labels)) * num_epochs_per_subj
        logger.info(
            'when leaving subject %d out for testing, the accuracy is %d / %d = %.2f' %
            (i, num_epochs_per_subj-incorrect_predict, num_epochs_per_subj,
             (num_epochs_per_subj-incorrect_predict) * 1.0 / num_epochs_per_subj)
        )
        print(clf.score(list(zip(test_data, test_data)), test_labels))

def example_of_cross_validation_using_model_selection(raw_data, labels, num_subjects, num_epochs_per_subj):
    # NOTE: this method does not work for sklearn.svm.SVC with precomputed kernel
    # when the kernel matrix is computed in portions; also, this method only works
    # for self-correlation, i.e. correlation between the same data matrix.

    # no shrinking, set C=1
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    #logit_clf = LogisticRegression()
    clf = Classifier(svm_clf, epochs_per_subj=num_epochs_per_subj)
    # doing leave-one-subject-out cross validation
    # no shuffling in cv
    skf = model_selection.StratifiedKFold(n_splits=num_subjects,
                                          shuffle=False)
    scores = model_selection.cross_val_score(clf, list(zip(raw_data, raw_data)),
                                             y=labels,
                                             cv=skf)
    print(scores)
    logger.info(
        'the overall cross validation accuracy is %.2f' %
        np.mean(scores)
    )

def example_of_correlating_two_components(raw_data, raw_data2, labels, num_subjects, num_epochs_per_subj):
    # aggregate the kernel matrix to save memory
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    clf = Classifier(svm_clf, epochs_per_subj=num_epochs_per_subj)
    num_training_samples=num_epochs_per_subj*(num_subjects-1)
    clf.fit(list(zip(raw_data[0:num_training_samples], raw_data2[0:num_training_samples])),
            labels[0:num_training_samples])
    X = list(zip(raw_data[num_training_samples:], raw_data2[num_training_samples:]))
    predict = clf.predict(X)
    print(predict)
    print(clf.decision_function(X))
    test_labels = labels[num_training_samples:]
    incorrect_predict = hamming(predict, np.asanyarray(test_labels)) * num_epochs_per_subj
    logger.info(
        'when aggregating the similarity matrix to save memory, '
        'the accuracy is %d / %d = %.2f' %
        (num_epochs_per_subj-incorrect_predict, num_epochs_per_subj,
         (num_epochs_per_subj-incorrect_predict) * 1.0 / num_epochs_per_subj)
    )
    # when the kernel matrix is computed in portion, the test data is already in
    print(clf.score(X, test_labels))

def example_of_correlating_two_components_aggregating_sim_matrix(raw_data, raw_data2, labels,
                                                                 num_subjects, num_epochs_per_subj):
    # aggregate the kernel matrix to save memory
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    clf = Classifier(svm_clf, num_processed_voxels=1000, epochs_per_subj=num_epochs_per_subj)
    num_training_samples=num_epochs_per_subj*(num_subjects-1)
    clf.fit(list(zip(raw_data, raw_data2)), labels,
            num_training_samples=num_training_samples)
    predict = clf.predict()
    print(predict)
    print(clf.decision_function())
    test_labels = labels[num_training_samples:]
    incorrect_predict = hamming(predict, np.asanyarray(test_labels)) * num_epochs_per_subj
    logger.info(
        'when aggregating the similarity matrix to save memory, '
        'the accuracy is %d / %d = %.2f' %
        (num_epochs_per_subj-incorrect_predict, num_epochs_per_subj,
         (num_epochs_per_subj-incorrect_predict) * 1.0 / num_epochs_per_subj)
    )
    # when the kernel matrix is computed in portion, the test data is already in
    print(clf.score(None, test_labels))

# python3 classification.py face_scene bet.nii.gz face_scene/prefrontal_top_mask.nii.gz face_scene/fs_epoch_labels.npy
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

    images = dataset.load_images_from_dir(data_dir, extension)
    mask = dataset.load_boolean_mask(mask_file)
    conditions = dataset.load_labels(epoch_file)
    raw_data, _, labels = prepare_fcma_data(images, conditions, mask)

    example_of_aggregating_sim_matrix(raw_data, labels, num_subjects, num_epochs_per_subj)

    example_of_cross_validation_with_detailed_info(raw_data, labels, num_subjects, num_epochs_per_subj)

    example_of_cross_validation_using_model_selection(raw_data, labels, num_subjects, num_epochs_per_subj)

    # test of two different components for correlation computation
    # images = dataset.load_images_from_dir(data_dir, extension)
    # mask2 = dataset.load_boolean_mask('face_scene/visual_top_mask.nii.gz')
    # raw_data, raw_data2, labels = prepare_fcma_data(images, conditions, mask,
    #                                                 mask2)
    #example_of_correlating_two_components(raw_data, raw_data2, labels, num_subjects, num_epochs_per_subj)
    #example_of_correlating_two_components_aggregating_sim_matrix(raw_data, raw_data2, labels, num_subjects, num_epochs_per_subj)
