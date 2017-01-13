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
from brainiak.fcma.io import prepare_fcma_data
from sklearn import svm
#from sklearn.linear_model import LogisticRegression
import sys
import logging
import numpy as np
#from sklearn.externals import joblib

format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# if want to output log to a file instead of outputting log to the console,
# replace "stream=sys.stdout" with "filename='fcma.log'"
logging.basicConfig(level=logging.INFO, format=format, stream=sys.stdout)
logger = logging.getLogger(__name__)


# python classification.py face_scene bet.nii.gz face_scene/prefrontal_top_mask.nii.gz face_scene/fs_epoch_labels.npy 12
if __name__ == '__main__':
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]
    raw_data, labels = prepare_fcma_data(data_dir, extension, mask_file, epoch_file)
    epochs_per_subj = int(sys.argv[5])
    # no shrinking, set C=1
    use_clf = svm.SVC(kernel='precomputed', shrinking=False, C=1)
    #use_clf = LogisticRegression()
    clf = Classifier(use_clf, epochs_per_subj=epochs_per_subj)
    training_data = raw_data[0:204]
    test_data = raw_data[204:]
    clf.fit(training_data, labels[0:204])
    # joblib can be used for saving and loading models
    #joblib.dump(clf, 'model/logistic.pkl')
    #clf = joblib.load('model/svm.pkl')
    print(clf.predict(test_data))
    print(clf.decision_function(test_data))
    print(np.asanyarray(labels[204:]))
