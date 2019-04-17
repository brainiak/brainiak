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
from scipy.stats.mstats import zscore
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from numpy.random import RandomState
from scipy.spatial.distance import hamming

# specify the random state to fix the random numbers
prng = RandomState(1234567890)


def create_epoch(idx, num_voxels):
    row = 12
    col = num_voxels
    mat = prng.rand(row, col).astype(np.float32)
    # impose a pattern to even epochs
    if idx % 2 == 0:
        mat = np.sort(mat, axis=0)
    mat = zscore(mat, axis=0, ddof=0)
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    mat = np.nan_to_num(mat)
    mat = mat / math.sqrt(mat.shape[0])
    return mat


def test_classification():
    fake_raw_data = [create_epoch(i, 5) for i in range(20)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # 5 subjects, 4 epochs per subject
    epochs_per_subj = 4
    # svm
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    training_data = fake_raw_data[0:12]
    clf = Classifier(svm_clf, epochs_per_subj=epochs_per_subj)
    clf.fit(list(zip(training_data, training_data)), labels[0:12])
    expected_confidence = np.array([-1.18234421, 0.97403604, -1.04005679,
                                    0.92403019, -0.95567738, 1.11746593,
                                    -0.83275891, 0.9486868])
    recomputed_confidence = clf.decision_function(list(zip(
        fake_raw_data[12:], fake_raw_data[12:])))
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(recomputed_confidence)
                               ) * expected_confidence.size
    assert hamming_distance <= 1, \
        'decision function of SVM with recomputation ' \
        'does not provide correct results'
    y_pred = clf.predict(list(zip(fake_raw_data[12:], fake_raw_data[12:])))
    expected_output = [0, 0, 0, 1, 0, 1, 0, 1]
    hamming_distance = hamming(y_pred, expected_output) * len(y_pred)
    assert hamming_distance <= 1, \
        'classification via SVM does not provide correct results'
    confidence = clf.decision_function(list(zip(fake_raw_data[12:],
                                                fake_raw_data[12:])))
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(confidence)
                               ) * confidence.size
    assert hamming_distance <= 1, \
        'decision function of SVM without recomputation ' \
        'does not provide correct results'
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    score = clf.score(list(zip(fake_raw_data[12:], fake_raw_data[12:])), y)
    assert np.isclose([hamming(y_pred, y)], [1-score])[0], \
        'the prediction score is incorrect'
    # svm with partial similarity matrix computation
    clf = Classifier(svm_clf, num_processed_voxels=2,
                     epochs_per_subj=epochs_per_subj)
    clf.fit(list(zip(fake_raw_data, fake_raw_data)),
            labels,
            num_training_samples=12)
    y_pred = clf.predict()
    expected_output = [0, 0, 0, 1, 0, 1, 0, 1]
    hamming_distance = hamming(y_pred, expected_output) * len(y_pred)
    assert hamming_distance <= 1, \
        'classification via SVM (partial sim) does not ' \
        'provide correct results'
    confidence = clf.decision_function()
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(confidence)) * confidence.size
    assert hamming_distance <= 1, \
        'decision function of SVM (partial sim) without recomputation ' \
        'does not provide correct results'
    # logistic regression
    lr_clf = LogisticRegression()
    clf = Classifier(lr_clf, epochs_per_subj=epochs_per_subj)
    clf.fit(list(zip(training_data, training_data)), labels[0:12])
    expected_confidence = np.array([-4.49666484, 3.73025553, -4.04181695,
                                    3.73027436, -3.77043872, 4.42613412,
                                    -3.35616616, 3.77716609])
    recomputed_confidence = clf.decision_function(list(zip(
        fake_raw_data[12:], fake_raw_data[12:])))
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(recomputed_confidence)
                               ) * expected_confidence.size
    assert hamming_distance <= 1, \
        'decision function of logistic regression with recomputation ' \
        'does not provide correct results'
    y_pred = clf.predict(list(zip(fake_raw_data[12:], fake_raw_data[12:])))
    expected_output = [0, 0, 0, 1, 0, 1, 0, 1]
    hamming_distance = hamming(y_pred, expected_output) * len(y_pred)
    assert hamming_distance <= 1, \
        'classification via logistic regression ' \
        'does not provide correct results'
    confidence = clf.decision_function(list(zip(
        fake_raw_data[12:], fake_raw_data[12:])))
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(confidence)
                               ) * confidence.size
    assert hamming_distance <= 1, \
        'decision function of logistic regression without precomputation ' \
        'does not provide correct results'


def test_classification_with_two_components():
    fake_raw_data = [create_epoch(i, 5) for i in range(20)]
    fake_raw_data2 = [create_epoch(i, 6) for i in range(20)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # 5 subjects, 4 epochs per subject
    epochs_per_subj = 4
    # svm
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    training_data = fake_raw_data[0: 12]
    training_data2 = fake_raw_data2[0: 12]
    clf = Classifier(svm_clf, epochs_per_subj=epochs_per_subj)
    clf.fit(list(zip(training_data, training_data2)), labels[0:12])
    expected_confidence = np.array([-1.23311606, 1.02440964, -0.93898336,
                                    1.07028798, -1.04420007, 0.97647772,
                                    -1.0498268, 1.04970111])
    recomputed_confidence = clf.decision_function(list(zip(
        fake_raw_data[12:], fake_raw_data2[12:])))
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(recomputed_confidence)
                               ) * expected_confidence.size
    assert hamming_distance <= 1, \
        'decision function of SVM with recomputation ' \
        'does not provide correct results'
    y_pred = clf.predict(list(zip(fake_raw_data[12:], fake_raw_data2[12:])))
    expected_output = [0, 1, 0, 1, 0, 1, 0, 1]
    hamming_distance = hamming(y_pred, expected_output) * len(y_pred)
    assert hamming_distance <= 1, \
        'classification via SVM does not provide correct results'
    confidence = clf.decision_function(list(zip(
        fake_raw_data[12:], fake_raw_data2[12:])))
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(confidence)) * confidence.size
    assert hamming_distance <= 1, \
        'decision function of SVM without recomputation ' \
        'does not provide correct results'
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    score = clf.score(list(zip(fake_raw_data[12:], fake_raw_data2[12:])), y)
    assert np.isclose([hamming(y_pred, y)], [1-score])[0], \
        'the prediction score is incorrect'
    # svm with partial similarity matrix computation
    clf = Classifier(svm_clf, num_processed_voxels=2,
                     epochs_per_subj=epochs_per_subj)
    clf.fit(list(zip(fake_raw_data, fake_raw_data2)),
            labels,
            num_training_samples=12)
    y_pred = clf.predict()
    expected_output = [0, 1, 0, 1, 0, 1, 0, 1]
    hamming_distance = hamming(y_pred, expected_output) * len(y_pred)
    assert hamming_distance <= 1, \
        'classification via SVM (partial sim) does not ' \
        'provide correct results'
    confidence = clf.decision_function()
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(confidence)) * confidence.size
    assert hamming_distance <= 1, \
        'decision function of SVM (partial sim) without recomputation ' \
        'does not provide correct results'
    # logistic regression
    lr_clf = LogisticRegression()
    clf = Classifier(lr_clf, epochs_per_subj=epochs_per_subj)
    # specifying num_training_samples is for coverage
    clf.fit(list(zip(training_data, training_data2)),
            labels[0:12],
            num_training_samples=12)
    expected_confidence = np.array([-4.90819848, 4.22548132, -3.76255726,
                                    4.46505975, -4.19933099, 4.08313584,
                                    -4.23070437, 4.31779758])
    recomputed_confidence = clf.decision_function(list(zip(
        fake_raw_data[12:], fake_raw_data2[12:])))
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(recomputed_confidence)
                               ) * expected_confidence.size
    assert hamming_distance <= 1, \
        'decision function of logistic regression with recomputation ' \
        'does not provide correct results'
    y_pred = clf.predict(list(zip(fake_raw_data[12:], fake_raw_data2[12:])))
    expected_output = [0, 1, 0, 1, 0, 1, 0, 1]
    hamming_distance = hamming(y_pred, expected_output) * len(y_pred)
    assert hamming_distance <= 1, \
        'classification via logistic regression ' \
        'does not provide correct results'
    confidence = clf.decision_function(list(zip(fake_raw_data[12:],
                                                fake_raw_data2[12:])))
    hamming_distance = hamming(np.sign(expected_confidence),
                               np.sign(confidence)) * confidence.size
    assert hamming_distance <= 1, \
        'decision function of logistic regression without precomputation ' \
        'does not provide correct results'


if __name__ == '__main__':
    test_classification()
    test_classification_with_two_components()
