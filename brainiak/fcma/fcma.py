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
"""Full Correlation Matrix Analysis (FCMA)

This implementation is based on the following publications:

.. [Wang2015-1] Full correlation matrix analysis (FCMA): An unbiased method for
   task-related functional connectivity",
   Yida Wang, Jonathan D Cohen, Kai Li, Nicholas B Turk-Browne.
   Journal of Neuroscience Methods, 2015.

.. [Wang2015-2] "Full correlation matrix analysis of fMRI data on Intel® Xeon
   Phi™ coprocessors",
   Yida Wang, Michael J. Anderson, Jonathan D. Cohen, Alexander Heinecke, Kai Li,
   Nadathur Satish, Narayanan Sundaram, Nicholas B. Turk-Browne, Theodore L. Willke.
   In Proceedings of the International Conference for High Performance Computing,
   Networking, Storage and Analysis. 2015.
"""

# Authors: Yida Wang
# (Intel Labs), 2016

import numpy as np
import nibabel as nib
import os, math, sys, time
from mpi4py import MPI
from scipy.stats.mstats import zscore
import ctypes
from ctypes.util import find_library
from ctypes import byref, c_char, c_float, c_int
from sklearn import cross_validation
from sklearn import svm


# __all__ = ['']

WORKTAG = 0
TERMINATETAG = 1

def readActivityData(dir, file_extension, mask_file):
    """ read in data in NIfTI format and apply the spacial mask to them
    :param dir: the path to all input file
    :param file_extension: the file extension, usually nii.gz or nii
    :param mask_file: the absolute path of the mask file, we apply the mask right after reading a file for saving memory
    :return: activity_data: array of matrices in (nTRs, nVoxels) shape (numpy array),
                            len(activity_data) equals the number of subjects
    """
    mask_img = nib.load(mask_file)
    mask = mask_img.get_data()
    count = 0
    for index in np.ndindex(mask.shape):
        if mask[index]!=0:
            count+=1
    files = [f for f in sorted(os.listdir(dir)) if os.path.isfile(os.path.join(dir, f)) and f.endswith(file_extension)]
    activity_data = []
    for f in files:
        img = nib.load(os.path.join(dir, f))
        data = img.get_data()
        (d1, d2, d3, d4) = data.shape
        masked_data = np.zeros([d4, count], np.float32, order='C')
        count1=0
        for index in np.ndindex(mask.shape):
            if mask[index]!=0:
                masked_data[:,count1] = np.copy(data[index])
                count1+=1
        activity_data.append(masked_data)
        print(f, masked_data.shape)
        sys.stdout.flush()
    return activity_data

def separateEpochs(activity_data, epoch_map):
    """ keep data in epochs of interest specified in epoch_map and z-score them for computing correlation
    :param activity_data: array of matrices in (nTRs, nVoxels) shape,
                          consisting of the masked activity data of all subjects
    :param epoch_map: array of (subject id, starting TR, ending TR) tuple
    :return: raw_data: array of matrices in (epoch length, nVoxels) shape,
                       len(raw_data) equals the number of epochs
    """
    raw_data = []
    for (sid, s, e) in epoch_map:
        mat = activity_data[sid][s:e+1,:]
        (r, c) = mat.shape
        mat = zscore(mat, axis=0, ddof=0)
        mat = np.nan_to_num(mat) # if zscore fails (standard deviation is zero), set all values to be zero
        mat = mat / math.sqrt(r)
        raw_data.append(mat)
    return raw_data


def preprocessActivityData():
    """Read, filter (applying masks and eliminating TRs not in epochs of interest) the activity data

    """

class VoxelSelector:
    """Correlation-based voxel selection component of FCMA
    Parameters
    ----------

    raw_data : numpy array of array of float
        Each array of float contains the activity data of an epoch of a subject,
            which essentially is a nVoxels x nTRs matrix but is serialized as one dimensional vector
        Assumption: 1. all activity data contains the same number of voxels
                    2. the activity data has been z-scored, ready to compute correlation as matrix multiplication
                    3. the epochs belonging to the same subject are continuous in the array
                    4. all subjects have the same number of epochs
                    5. voxel selection is always done in the self-correlation, i.e. raw_data correlate with themselves
                    6. the classifier used is SVM with linear kernel

    epochs_per_subj : int
        The number of epochs of each subject

    num_voxels: int
        The number of voxels participating in the voxel selection

    labels: 1D array of int
        Describing the condition labels of the epochs, the length of labels equals the number of epochs,
            i.e. the length of raw_data

    num_folds: int
        The number of folds to be conducted in the cross validation

    voxel_unit: int, default 100
        The number of voxel assigned to a worker each time
    """
    def __init__(self, raw_data, epochs_per_subj, labels, num_folds, voxel_unit=100):
        self.raw_data = raw_data
        self.epochs_per_subj = epochs_per_subj
        self.num_voxels = raw_data[0].shape[1]
        self.labels = labels
        self.num_folds = num_folds
        self.voxel_unit = voxel_unit
        libblas_so_name = find_library('blas') if find_library('blas') else find_library('mkl_rt')
        if libblas_so_name is None:
            raise RuntimeError("No Blas library is found in the system")
        self.blas_library = ctypes.cdll.LoadLibrary(libblas_so_name)
        if self.num_voxels == 0:
            raise RuntimeError("Zero processed voxels")

    def run(self):
        """run voxel selection in master-worker model
        Sort the voxels based on the cross-validation accuracy of their correlation vectors
        """
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            results = self.master()
            # TODO: sort results
            results.sort(key=lambda tup: tup[1], reverse=True)
        else:
            self.worker()
            results = []
        return results

    def master(self):
        """
        return value
            results: an array of tuple (voxel id, accuracy), the length of array equals the number of voxels
        """
        results = []
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        sending_voxels = self.voxel_unit if self.voxel_unit<self.num_voxels else self.num_voxels
        sys.stdout.flush()
        current_task = (0, sending_voxels)
        status = MPI.Status()
        using_size = size   # using_size is used when the number of tasks is smaller than the number of workers
        for i in range(1, size):
            if current_task[1]==0:
                using_size = i
                break
            comm.send(current_task, dest=i, tag=WORKTAG)
            next_start = current_task[0]+current_task[1]
            sending_voxels = self.voxel_unit if self.voxel_unit<self.num_voxels-next_start \
                else self.num_voxels-next_start
            current_task = (next_start, sending_voxels)

        while using_size==size:
            if current_task[1]==0:
                break
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            results += result
            comm.send(current_task, dest=status.Get_source(), tag=WORKTAG)
            next_start = current_task[0]+current_task[1]
            sending_voxels = self.voxel_unit if self.voxel_unit<self.num_voxels-next_start \
                else self.num_voxels-next_start
            current_task = (next_start, sending_voxels)

        for i in range(1, using_size):
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            results += result

        for i in range(1, size):
            comm.send(None, dest=i, tag=TERMINATETAG)

        return results

    def worker(self):
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        while 1:
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag(): break
            comm.send(self.voxelScoring(task), dest=0)

    def correlationComputation(self, task):
        s = task[0]
        e = s+task[1]
        time1 = time.time()
        nEpochs = len(self.raw_data)
        corr = np.zeros((task[1], nEpochs, self.num_voxels), np.float32, order='C')
        count=0
        for mat in self.raw_data:
            row_order = c_int(101)
            no_trans = c_int(111)
            trans = c_int(112)
            n1 = c_int(task[1])
            n2 = c_int(self.num_voxels)
            n3 = c_int(mat.shape[0])
            n4 = c_int(self.num_voxels*nEpochs)
            one = c_float(1.0)
            zero = c_float(0.0)
            self.blas_library.cblas_sgemm(row_order, trans, no_trans, n1, n2, n3, one,
                                 mat[:,s:e].ctypes.data_as(ctypes.c_void_p), n2,
                                 mat.ctypes.data_as(ctypes.c_void_p), n2,
                                 zero, corr[0,count,:].ctypes.data_as(ctypes.c_void_p), n4)
            count += 1
        time2 = time.time()
        print('correlation computation time:', (time2-time1)*1000.0)
        sys.stdout.flush()
        return corr

    def correlationNormalization(self, corr):
        time1 = time.time()
        (sv, e, av) = corr.shape
        for i in range(sv):
            start = 0
            while start<e:
                corr[i,start:start+self.epochs_per_subj,:] = \
                    .5 * np.log((corr[i,start:start+self.epochs_per_subj,:]+1)/
                                   (1-corr[i,start:start+self.epochs_per_subj,:]))
                corr[i,start:start+self.epochs_per_subj,:] = \
                    zscore(corr[i, start:start+self.epochs_per_subj, :], axis = 0, ddof = 0)
                start += self.epochs_per_subj
        corr = np.nan_to_num(corr) # if zscore fails (standard deviation is zero), set all values to be zero
        time2 = time.time()
        print('normalization time:', (time2-time1)*1000.0)
        sys.stdout.flush()
        return corr

    def crossValidation(self, task, corr):
        time1 = time.time()
        (sv, e, av) = corr.shape
        kernel_matrix = np.zeros((e, e), np.float32, order='C')
        results = []
        for i in range(sv):
            row_order = c_int(101)
            no_trans = c_int(111)
            lower = c_int(122)
            n1 = c_int(e)
            n2 = c_int(self.num_voxels)
            one = c_float(1.0)
            zero = c_float(0.0)
            self.blas_library.cblas_ssyrk(row_order, lower, no_trans, n1, n2, one, corr[i,:,:].ctypes.data_as(ctypes.c_void_p),
                                          n2, zero, kernel_matrix.ctypes.data_as(ctypes.c_void_p), n1)
            kernel_matrix *= .001
            for j in range(kernel_matrix.shape[0]):
                for k in range(j):
                    kernel_matrix[k,j] = kernel_matrix[j,k]
            clf = svm.SVC(kernel='precomputed', shrinking=False)
            scores = cross_validation.cross_val_score(clf, kernel_matrix, self.labels, cv=18)   # by default, no shuffling
            results.append((i+task[0], scores.mean()))
        time2 = time.time()
        print('cross validation time:', (time2-time1)*1000.0)
        #print(results)
        sys.stdout.flush()
        return results

    def voxelScoring(self, task):
        """Voxel selection worker
        Take the task in, do analysis on voxels specified by the task (voxel id, number of voxels)
        """
        # TODO: use the C++ routines to do the following:
        # 1. correlation computation
        # 2. normalization
        # 3. cross validation
        # input: activity matrices, starting voxel, number of voxels, epochs_per_subj, labels, number of folds
        # output: array of float, accuracy numbers
        # scores = voxelScoringingCPP(self.raw_data, task[0], task[1], self.epochs_per_subj, self.labels, self.num_folds)
        #result = ()
        #for i in range(len(scores)):
        #    result.append((task[0]+i, scores[i]))
        #return result
        time1 = time.time()
        corr = self.correlationComputation(task) # corr is a 3D array in row major,
                                                 # in (selected_voxels, epochs, all_voxels) shape
                                                 # corr[i,e,s+j] = corr[j,e,s+i]
        corr = self.correlationNormalization(corr) # in-place z-score, the result is still in corr
        results = self.crossValidation(task, corr)
        time2 = time.time()
        print('task:', task[0], time2-time1)
        sys.stdout.flush()
        return results

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sys.stdout.flush()
    raw_data = []
    labels = []
    if rank==0:
        activity_data = readActivityData('/Users/yidawang/data/face_scene/raw', 'nii.gz',
                                '/Users/yidawang/data/face_scene/mask.nii.gz')
        epoch_map = []
        for i in range(18):
            epoch_map.append((i, 4, 15))
            epoch_map.append((i, 24, 35))
            epoch_map.append((i, 44, 55))
            epoch_map.append((i, 64, 75))
            epoch_map.append((i, 84, 95))
            epoch_map.append((i, 104, 115))
            epoch_map.append((i, 124, 135))
            epoch_map.append((i, 144, 155))
            epoch_map.append((i, 164, 175))
            epoch_map.append((i, 184, 195))
            epoch_map.append((i, 204, 215))
            epoch_map.append((i, 224, 235))
        raw_data=separateEpochs(activity_data, epoch_map)
        labels = [i%2 for i in range(216)]
    raw_data = comm.bcast(raw_data, root=0)
    labels = comm.bcast(labels, root=0)
    vs = VoxelSelector(raw_data, 12, labels, 18)
    results = vs.run()
    if rank==0:
        print(results[0:100])
