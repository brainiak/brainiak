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

Activity-based voxel selection
"""

# Authors: Yida Wang
# (Intel Labs), 2017

import numpy as np
from sklearn import model_selection
from scipy.stats.mstats import zscore
import logging
from mpi4py import MPI

logger = logging.getLogger(__name__)

__all__ = [
    "MVPAVoxelSelector",
]


class MVPAVoxelSelector:
    """Activity-based voxel selection component of FCMA

    Parameters
    ----------

    raw\_data: list of 4D array
        each element of the list is the raw data of a subject,
        in the shape of [x, y, z, t]

    mask: 3D array

    epoch\_info: list of tuple (label, sid, start, end).
        label is the condition labels of the epochs;
        sid is the subject id, corresponding to the index of raw_data;
        start is the start TR of an epoch (inclusive);
        end is the end TR of an epoch(exclusive).
        Assuming len(labels) labels equals the number of epochs and
        the epochs of the same sid are adjacent in epoch_info

    num\_folds: int
        the number of folds to be conducted in the cross validation

    sl: Searchlight
        the distributed Searchlight object

    Attributes
    ----------

    processed\_data\_: 4D array in shape [brain 3D + epoch]
        contains the averaged and normalized brain data epoch by epoch
        it is generated from raw\_data and epoch\_info

    labels\_: 1D array
        contains the labels of the epochs, extracted from epoch\_info
    """
    def __init__(self,
                 raw_data,
                 mask,
                 epoch_info,
                 num_folds,
                 sl
                 ):
        self.raw_data = raw_data
        self.mask = mask.astype(np.bool)
        self.epoch_info = epoch_info
        self.num_folds = num_folds
        self.sl = sl
        self.processed_data_ = None
        self.labels_ = None
        num_voxels = np.sum(self.mask)
        if num_voxels == 0:
            raise ValueError('Zero processed voxels')

    def _preprocess_data(self):
        """ process the raw data according to epoch info

        This is done in rank 0 which has the raw_data read in
        Average the activity within epochs and z-scoring within subject.
        Write the results to self.processed_data,
        which is a 4D array of averaged epoch by epoch processed data
        Also write the labels to self.label as a 1D numpy array
        """
        logger.info(
            'mask size: %d' %
            np.sum(self.mask)
        )
        num_epochs = len(self.epoch_info)
        (d1, d2, d3, _) = self.raw_data[0].shape
        self.processed_data_ = np.empty([d1, d2, d3, num_epochs])
        self.labels_ = np.empty(num_epochs)
        subject_count = [0]  # counting the epochs per subject for z-scoring
        cur_sid = -1
        # averaging
        for idx, epoch in enumerate(self.epoch_info):
            self.labels_[idx] = epoch[0]
            if cur_sid != epoch[1]:
                subject_count.append(0)
                cur_sid = epoch[1]
            subject_count[-1] += 1
            self.processed_data_[:, :, :, idx] = \
                np.mean(self.raw_data[cur_sid][:, :, :, epoch[2]:epoch[3]],
                        axis=3)
        # z-scoring
        cur_epoch = 0
        for i in subject_count:
            if i > 1:
                self.processed_data_[:, :, :, cur_epoch:cur_epoch + i] = \
                    zscore(self.processed_data_[:, :, :,
                           cur_epoch:cur_epoch + i],
                           axis=3, ddof=0)
            cur_epoch += i
        # if zscore fails (standard deviation is zero),
        # set all values to be zero
        self.processed_data_ = np.nan_to_num(self.processed_data_)

    def run(self, clf):
        """ run activity-based voxel selection

        Sort the voxels based on the cross-validation accuracy
        of their activity vectors within the searchlight

        Parameters
        ----------
        clf: classification function
            the classifier to be used in cross validation

        Returns
        -------
        result_volume: 3D array of accuracy numbers
            contains the voxelwise accuracy numbers obtained via Searchlight
        results: list of tuple (voxel_id, accuracy)
            the accuracy numbers of all voxels, in accuracy descending order
            the length of array equals the number of voxels
        """
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            logger.info(
                'running activity-based voxel selection via Searchlight'
            )
        if rank == 0:
            self._preprocess_data()
        self.sl.distribute([self.processed_data_], self.mask)
        self.sl.broadcast((self.labels_, self.num_folds))
        if rank == 0:
            logger.info(
                'data preparation done'
            )

        # Searchlight kernel function
        def _sfn(l, mask, myrad, bcast_var):
            data = l[0][mask, :].T
            # print(l[0].shape, mask.shape, data.shape)
            skf = model_selection.StratifiedKFold(n_splits=bcast_var[1],
                                                  shuffle=False)
            accuracy = np.mean(model_selection.cross_val_score(clf, data,
                                                               y=bcast_var[0],
                                                               cv=skf,
                                                               n_jobs=1))
            return accuracy
        # obtain a 3D array with accuracy numbers
        result_volume = self.sl.run_searchlight(_sfn)
        # get result tuple list from the volume
        result_list = result_volume[self.mask]
        results = []
        if rank == 0:
            for idx, value in enumerate(result_list):
                if value is None:
                    value = 0
                results.append((idx, value))
            # Sort the voxels
            results.sort(key=lambda tup: tup[1], reverse=True)
            logger.info(
                'activity-based voxel selection via Searchlight is done'
            )
        return result_volume, results
