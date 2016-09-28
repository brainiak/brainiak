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

# specify the random state to fix the random numbers
prng = RandomState(1234567890)

def create_epoch():
    row = 12
    col = 5
    mat = prng.rand(row, col).astype(np.float32)
    mat = zscore(mat, axis=0, ddof=0)
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    mat = np.nan_to_num(mat)
    mat = mat / math.sqrt(mat.shape[0])
    return mat

def test_classification():
    fake_raw_data = [create_epoch(), create_epoch(),
                     create_epoch(), create_epoch(),
                     create_epoch(), create_epoch(),
                     create_epoch(), create_epoch(),
                     create_epoch(), create_epoch(),
                     create_epoch(), create_epoch()]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # 3 subjects, 4 epochs per subject
    epochs_per_subj = 4
    # svm
    svm_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    training_data = fake_raw_data[0: 8]
    clf = Classifier(epochs_per_subj, svm_clf)
    clf.fit(training_data, labels[0:8])
    y_pred = clf.predict(fake_raw_data[8:])
    expected_output = [0, 0, 1, 0]
    assert np.array_equal(y_pred, expected_output), \
       'classification via SVM does not provide correct results'
    # logistic regression
    lr_clf = LogisticRegression()
    clf = Classifier(epochs_per_subj, lr_clf)
    clf.fit(training_data, labels[0:8])
    y_pred = clf.predict(fake_raw_data[8:])
    assert np.array_equal(y_pred, expected_output), \
        'classification via logistic regression does not provide correct results'

if __name__ == '__main__':
    test_classification()
