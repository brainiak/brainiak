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
import scipy.io
from scipy.stats import stats
import numpy as np

# Define the Theano flags to use cpu and float64 before theano is imported in brainiak
import os
os.environ['THEANO_FLAGS'] = 'device=cpu, floatX=float64'

import brainiak.funcalign.sssrm



# Load the input data that contains the movie stimuli for unsupervised training with SS-SRM
movie_file = scipy.io.loadmat('data/movie_data.mat')
movie_data_left = movie_file['movie_data_lh']
movie_data_right = movie_file['movie_data_rh']
subjects = movie_data_left.shape[2]

# Load the input data that contains the image stimuli and its labels for training a classifier
image_file = scipy.io.loadmat('data/image_data.mat')
image_data_left = image_file['image_data_lh']
image_data_right = image_file['image_data_rh']

# Merge the two hemispheres into one piece of data and
# convert data to a list of arrays matching SS-SRM input.
# Each element is a matrix of voxels by TRs_i.
image_data = []
movie_data = []
for s in range(subjects):
    image_data.append(np.concatenate([image_data_left[:, :, s], image_data_right[:, :, s]], axis=0))
    movie_data.append(np.concatenate([movie_data_left[:, :, s], movie_data_right[:, :, s]], axis=0))

# Read the labels of the image data for training the classifier.
labels = scipy.io.loadmat('data/label.mat')
labels = np.squeeze(labels['label'])
image_samples = labels.size

# Z-score the data
for subject in range(subjects):
    image_data[subject] = stats.zscore(image_data[subject], axis=1, ddof=1)
    movie_data[subject] = stats.zscore(movie_data[subject], axis=1, ddof=1)


# Run cross validation on the blocks of image stimuli (leave one block out)
# Note: There are 8 blocks of 7 samples (TRs) each
print("Running cross-validation with SS-SRM... (this may take a while)")
accuracy = np.zeros((8,))
for block in range(8):
    print("Block ", block)

    # Create masks with the train and validation samples
    idx_validation = np.zeros((image_samples,), dtype=bool)
    idx_validation[block*7:(block+1)*7] = True
    idx_train = np.ones((image_samples,), dtype=bool)
    idx_train[block*7:(block+1)*7] = False

    # Divide the samples and labels in train and validation sets
    image_data_train = [None] * subjects
    labels_train = [None] * subjects
    image_data_validation = [None] * subjects
    labels_validation = [None] * subjects
    for s in range(subjects):
        image_data_train[s] = image_data[s][:, idx_train]
        labels_train[s] = labels[idx_train]
        image_data_validation[s] = image_data[s][:, idx_validation]
        labels_validation[s] = labels[idx_validation]

    # Run SS-SRM with the movie data and training image data
    model = brainiak.funcalign.sssrm.SSSRM(n_iter=10, features=50, gamma=1.0, alpha=0.2)
    model.fit(movie_data, labels_train, image_data_train)

    # Predict on the validation samples and check results
    prediction = model.predict(image_data_validation)
    predicted = 0
    total_predicted = 0
    for s in range(subjects):
        predicted += sum(prediction[s] == labels_validation[s])
        total_predicted += prediction[s].size
    accuracy[block] = predicted/total_predicted
    print("Accuracy for this block: ",accuracy[block])

print("SS-SRM: The average accuracy among all subjects is {0:f} +/- {1:f}".format(np.mean(accuracy), np.std(accuracy)))
