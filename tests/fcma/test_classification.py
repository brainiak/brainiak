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

def create_epoch(idx):
    row = 12
    col = 5
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
    fake_raw_data = [create_epoch(i) for i in range(20)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # 4 subjects, 4 epochs per subject
    epochs_per_subj = 4
    # svm
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    training_data = fake_raw_data[0: 12]
    clf = Classifier(svm_clf, epochs_per_subj=epochs_per_subj)
    clf.fit(training_data, labels)
    expected_confidence = np.array([-1.18234421, 0.97403604, -1.04005679, 
                                    0.92403019, -0.95567738, 1.11746593,
                                    -0.83275891, 0.9486868])
    recomputed_confidence = clf.decision_function(fake_raw_data[12:])
    hamming_distance = hamming(np.sign(expected_confidence), 
			       np.sign(recomputed_confidence))
    assert hamming_distance <= 1, \
        'decision function of SVM with recomputation ' \
        'does not provide correct results'
    y_pred = clf.predict(fake_raw_data[12:])
    expected_output = [0, 0, 0, 1, 0, 1, 0, 1]
    hamming_distance = hamming(y_pred, expected_output) * len(y_pred)
    assert hamming_distance <= 1, \
        'classification via SVM does not provide correct results'
    confidence = clf.decision_function(fake_raw_data[12:])
    hamming_distance = hamming(np.sign(expected_confidence),
			       np.sign(confidence))
    assert hamming_distance <= 1, \
        'decision function of SVM without recomputation ' \
        'does not provide correct results'
    # logistic regression
    lr_clf = LogisticRegression()
    clf = Classifier(lr_clf, epochs_per_subj=epochs_per_subj)
    clf.fit(training_data, labels[0:12])
    expected_confidence = np.array([-4.49666484, 3.73025553, -4.04181695, 
                                    3.73027436, -3.77043872, 4.42613412,
                                    -3.35616616, 3.77716609])
    recomputed_confidence = clf.decision_function(fake_raw_data[12:])
    hamming_distance = hamming(np.sign(expected_confidence), 
			       np.sign(recomputed_confidence))
    assert hamming_distance <= 1, \
        'decision function of logistic regression with recomputation ' \
        'does not provide correct results'
    y_pred = clf.predict(fake_raw_data[12:])
    expected_output = [0, 0, 0, 1, 0, 1, 0, 1]
    hamming_distance = hamming(y_pred, expected_output) * len(y_pred)
    assert hamming_distance <= 1, \
        'classification via logistic regression ' \
        'does not provide correct results'
    confidence = clf.decision_function(fake_raw_data[12:])
    hamming_distance = hamming(np.sign(expected_confidence), 
			       np.sign(confidence))
    assert hamming_distance <= 1, \
        'decision function of logistic regression without precomputation ' \
        'does not provide correct results'

if __name__ == '__main__':
    test_classification()
