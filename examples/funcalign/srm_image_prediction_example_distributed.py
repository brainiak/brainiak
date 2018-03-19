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
from sklearn.metrics import confusion_matrix
from sklearn.svm import NuSVC
import numpy as np
import brainiak.funcalign.srm
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load the input data that contains the movie stimuli for unsupervised training with SRM
movie_file = scipy.io.loadmat('data/movie_data.mat')
movie_data_left = movie_file['movie_data_lh']
movie_data_right = movie_file['movie_data_rh']
subjects = movie_data_left.shape[2]

# Convert data to a list of arrays matching SRM input.
# Each element is a matrix of voxels by TRs.
# Also, concatenate data from both hemispheres in the brain.
movie_data = []
for s in range(subjects):
    if s % size == rank:
        movie_data.append(np.concatenate([movie_data_left[:, :, s], movie_data_right[:, :, s]], axis=0))
    else:
        movie_data.append(None)

# Z-score the data
for subject in range(subjects):
    if movie_data[subject] is not None:
        movie_data[subject] = stats.zscore(movie_data[subject], axis=1, ddof=1)

# Run SRM with the movie data
srm = brainiak.funcalign.srm.SRM(n_iter=10, features=50, comm=comm)
srm.fit(movie_data)

# We define a function to present the output of the experiment.
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plots a confusion matrix for each subject
    """
    import matplotlib.pyplot as plt
    import math
    plt.figure()
    subjects = len(cm)
    root_subjects = math.sqrt(subjects)
    cols = math.ceil(root_subjects)
    rows = math.ceil(subjects/cols)
    classes = cm[0].shape[0]
    for subject in range(subjects):
        plt.subplot(rows, cols, subject+1)
        plt.imshow(cm[subject], interpolation='nearest', cmap=plt.cm.bone)
        plt.xticks(np.arange(classes), range(1, classes+1))
        plt.yticks(np.arange(classes), range(1, classes+1))
        cbar = plt.colorbar(ticks=[0.0, 1.0], shrink=0.6)
        cbar.set_clim(0.0, 1.0)
        plt.xlabel("Predicted")
        plt.ylabel("True label")
        plt.title("{0:d}".format(subject + 1))
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Load the input data that contains the image stimuli and its labels for training a classifier
image_file = scipy.io.loadmat('data/image_data.mat')
image_data_left = image_file['image_data_lh']
image_data_right = image_file['image_data_rh']

# Convert data to a list of arrays matching SRM input.
# Each element is a matrix of voxels by TRs.
# Also, concatenate data from both hemispheres in the brain.
image_data = []
for s in range(subjects):
    if s % size == rank:
        image_data.append(np.concatenate([image_data_left[:, :, s], image_data_right[:, :, s]], axis=0))
    else:
        image_data.append(None)

# Z-score the image data
for subject in range(subjects):
    if subject % size == rank:
        image_data[subject] = stats.zscore(image_data[subject], axis=1, ddof=1)
    else:
        image_data[subject] = None

# Z-score the shared response data
image_data_shared = srm.transform(image_data)

for subject in range(subjects):
    image_data_shared[subject] = comm.bcast(image_data_shared[subject],
            root=(subject % size))

if rank == 0:
    for subject in range(subjects):
        image_data_shared[subject] = stats.zscore(image_data_shared[subject], axis=1, ddof=1)
    
    # Read the labels of the image data for training the classifier.
    labels = scipy.io.loadmat('data/label.mat')
    labels = np.squeeze(labels['label'])
    
    # Run a leave-one-out cross validation with the subjects
    train_labels = np.tile(labels, subjects-1)
    test_labels = labels
    accuracy = np.zeros((subjects,))
    cm = [None] * subjects
    for subject in range(subjects):
        # Concatenate the subjects' data for training into one matrix
        train_subjects = list(range(subjects))
        train_subjects.remove(subject)
        TRs = image_data_shared[0].shape[1]
        train_data = np.zeros((image_data_shared[0].shape[0], len(train_labels)))
        for train_subject in range(len(train_subjects)):
            start_index = train_subject*TRs
            end_index = start_index+TRs
            train_data[:, start_index:end_index] = image_data_shared[train_subjects[train_subject]]
    
        # Train a Nu-SVM classifier using scikit learn
        classifier = NuSVC(nu=0.5, kernel='linear')
        classifier = classifier.fit(train_data.T, train_labels)
    
        # Predict on the test data
        predicted_labels = classifier.predict(image_data_shared[subject].T)
        accuracy[subject] = sum(predicted_labels == test_labels)/float(len(predicted_labels))
    
        # Create a confusion matrix to see the accuracy of each class
        cm[subject] = confusion_matrix(test_labels, predicted_labels)
    
        # Normalize the confusion matrix
        cm[subject] = cm[subject].astype('float') / cm[subject].sum(axis=1)[:, np.newaxis]
    
    
    # Plot and print the results
    plot_confusion_matrix(cm, title="Confusion matrices for different test subjects with Probabilistic SRM")
    print("SRM: The average accuracy among all subjects is {0:f} +/- {1:f}".format(np.mean(accuracy), np.std(accuracy)))
