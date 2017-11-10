#  Copyright 2016 Princeton University
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
"""Event segmentation using a Hidden Markov Model

Given an ROI timeseries, this class uses an annealed fitting procedure to
segment the timeseries into events with stable activity patterns. After
learning the signature activity pattern of each event, the model can then be
applied to other datasets to identify a corresponding sequence of events.

Full details are available in the bioRxiv preprint:
Christopher Baldassano, Janice Chen, Asieh Zadbood,
Jonathan W Pillow, Uri Hasson, Kenneth A Norman
Discovering event structure in continuous narrative perception and memory
Neuron, Volume 95, Issue 3, 709 - 721.e5
http://www.cell.com/neuron/abstract/S0896-6273(17)30593-7
"""

# Authors: Chris Baldassano and Cătălin Iordan (Princeton University)

import numpy as np
from scipy import stats
import logging
import copy
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.exceptions import NotFittedError

from . import _utils as utils  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "EventSegment",
]


class EventSegment(BaseEstimator):
    """Class for event segmentation of continuous fMRI data

    Parameters
    ----------
    n_events: int
        Number of segments to learn

    step_var: Callable[[int], float] : default 4 * (0.98 ** (step - 1))
        The Gaussian variance to use during fitting, as a function of the
        number of steps. Should decrease slowly over time.

    n_iter: int : default 500
        Maximum number of steps to run during fitting

    Attributes
    ----------
    p_start, p_end: length n_events+1 ndarray
        initial and final prior distributions over events

    P: n_events+1 by n_events+1 ndarray
        HMM transition matrix

    ll_ : ndarray with length = number of training datasets
        Log-likelihood for training datasets over the course of training

    segments_:  list of (time by event) ndarrays
        Learned (soft) segmentation for training datasets

    event_var_ : float
        Gaussian variance at the end of learning

    event_pat_ : voxel by event ndarray
        Learned mean patterns for each event
    """

    def _default_var_schedule(step):
        return 4 * (0.98 ** (step - 1))

    def __init__(self, n_events=2,
                 step_var=_default_var_schedule,
                 n_iter=500):
        self.n_events = n_events
        self.step_var = step_var
        self.n_iter = n_iter

    def fit(self, X, y=None):
        """Learn a segmentation on training data

        Fits event patterns and a segmentation to training data. After
        running this function, the learned event patterns can be used to
        segment other datasets using find_events

        Parameters
        ----------
        X: time by voxel ndarray, or a list of such ndarrays
            fMRI data to be segmented. If a list is given, then all datasets
            are segmented simultaneously with the same event patterns

        y: not used (added to comply with BaseEstimator definition)

        Returns
        -------
        self: the EventSegment object
        """

        X = copy.deepcopy(X)
        if type(X) is not list:
            X = check_array(X)
            X = [X]

        n_train = len(X)
        for i in range(n_train):
            X[i] = X[i].T

        self.classes_ = np.arange(self.n_events)
        n_dim = X[0].shape[0]
        for i in range(n_train):
            assert (X[i].shape[0] == n_dim)

        # Double-check that data is z-scored in time
        for i in range(n_train):
            X[i] = stats.zscore(X[i], axis=1, ddof=1)

        # Initialize variables for fitting
        log_gamma = []
        for i in range(n_train):
            log_gamma.append(np.zeros((X[i].shape[1],
                                       self.n_events)))
        step = 1
        best_ll = float("-inf")
        self.ll_ = np.empty((0, n_train))
        while step <= self.n_iter:
            iteration_var = self.step_var(step)

            # Based on the current segmentation, compute the mean pattern
            # for each event
            seg_prob = [np.exp(lg) / np.sum(np.exp(lg), axis=0)
                        for lg in log_gamma]
            mean_pat = np.empty((n_train, n_dim, self.n_events))
            for i in range(n_train):
                mean_pat[i, :, :] = X[i].dot(seg_prob[i])
            mean_pat = np.mean(mean_pat, axis=0)

            # Based on the current mean patterns, compute the event
            # segmentation
            self.ll_ = np.append(self.ll_, np.empty((1, n_train)), axis=0)
            for i in range(n_train):
                logprob = self._logprob_obs(X[i],
                                            mean_pat, iteration_var)
                log_gamma[i], self.ll_[-1, i] = self._forward_backward(logprob)

            # If log-likelihood has started decreasing, undo last step and stop
            if np.mean(self.ll_[-1, :]) < best_ll:
                self.ll_ = self.ll_[:-1, :]
                break

            self.segments_ = [np.exp(lg) for lg in log_gamma]
            self.event_var_ = iteration_var
            self.event_pat_ = mean_pat
            best_ll = np.mean(self.ll_[-1, :])
            logger.debug("Fitting step %d, LL=%f", step, best_ll)

            step += 1

        return self

    def _logprob_obs(self, data, mean_pat, var):
        """Log probability of observing each timepoint under each event model

        Computes the log probability of each observed timepoint being
        generated by the Gaussian distribution for each event pattern

        Parameters
        ----------
        data: voxel by time ndarray
            fMRI data on which to compute log probabilities

        mean_pat: voxel by event ndarray
            Centers of the Gaussians for each event

        var: float or 1D array of length equal to the number of events
            Variance of the event Gaussians. If scalar, all events are
            assumed to have the same variance

        Returns
        -------
        logprob : time by event ndarray
            Log probability of each timepoint under each event Gaussian
        """

        n_vox = data.shape[0]
        t = data.shape[1]

        # z-score both data and mean patterns in space, so that Gaussians
        # are measuring Pearson correlations and are insensitive to overall
        # activity changes
        data_z = stats.zscore(data, axis=0, ddof=1)
        mean_pat_z = stats.zscore(mean_pat, axis=0, ddof=1)

        logprob = np.empty((t, self.n_events))

        if type(var) is not np.ndarray:
            var = var * np.ones(self.n_events)

        for k in range(self.n_events):
            logprob[:, k] = -0.5 * n_vox * np.log(
                2 * np.pi * var[k]) - 0.5 * np.sum(
                (data_z.T - mean_pat_z[:, k]).T ** 2, axis=0) / var[k]

        logprob /= n_vox
        return logprob

    def _forward_backward(self, logprob):
        """Runs forward-backward algorithm on observation log probs

        Given the log probability of each timepoint being generated by
        each event, run the HMM forward-backward algorithm to find the
        probability that each timepoint belongs to each event (based on the
        transition priors in p_start, p_end, and P)

        See https://en.wikipedia.org/wiki/Forward-backward_algorithm for
        mathematical details

        Parameters
        ----------
        logprob : time by event ndarray
            Log probability of each timepoint under each event Gaussian

        Returns
        -------
        log_gamma : time by event ndarray
            Log probability of each timepoint belonging to each event

        ll : float
            Log-likelihood of fit
        """
        logprob = copy.copy(logprob)
        t = logprob.shape[0]
        logprob = np.hstack((logprob, float("-inf") * np.ones((t, 1))))

        # Initialize variables
        log_scale = np.zeros(t)
        log_alpha = np.zeros((t, self.n_events + 1))
        log_beta = np.zeros((t, self.n_events + 1))

        # Set up transition matrix, with final sink state
        # For transition matrix of this form, the transition probability has
        # no impact on the final solution, since all valid paths must take
        # the same number of transitions
        p_start = np.zeros((1, self.n_events + 1))
        p_start[0, 0] = 1
        p_trans = (self.n_events-1)/t
        P = np.vstack((np.hstack((
            (1 - p_trans) * np.diag(np.ones(self.n_events))
            + p_trans * np.diag(np.ones(self.n_events - 1), 1),
            np.append(np.zeros((self.n_events - 1, 1)), [[p_trans]], axis=0))),
                            np.append(np.zeros((1, self.n_events)), [[1]],
                                      axis=1)))
        p_end = np.zeros((1, self.n_events + 1))
        p_end[0, -2] = 1

        # Forward pass
        for t in range(t):
            if t == 0:
                log_alpha[0, :] = self._log(p_start) + logprob[0, :]
            else:
                log_alpha[t, :] = self._log(np.exp(log_alpha[t - 1, :])
                                            .dot(P)) + logprob[t, :]

            log_scale[t] = np.logaddexp.reduce(log_alpha[t, :])
            log_alpha[t] -= log_scale[t]

        # Backward pass
        log_beta[-1, :] = self._log(p_end) - log_scale[-1]
        for t in reversed(range(t - 1)):
            obs_weighted = log_beta[t + 1, :] + logprob[t + 1, :]
            offset = np.max(obs_weighted)
            log_beta[t, :] = offset + self._log(
                np.exp(obs_weighted - offset).dot(P.T)) - log_scale[t]

        # Combine and normalize
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)

        ll = np.sum(log_scale[:(t - 1)]) + np.logaddexp.reduce(
            log_alpha[-1, :] + log_scale[-1] + self._log(p_end), axis=1)

        log_gamma = log_gamma[:, :-1]

        return log_gamma, ll

    def _log(self, x):
        """Modified version of np.log that manually sets values <=0 to -inf

        Parameters
        ----------
        x: ndarray of floats
            Input to the log function

        Returns
        -------
        log_ma: ndarray of floats
            log of x, with x<=0 values replaced with -inf
        """

        xshape = x.shape
        _x = x.flatten()
        y = utils.masked_log(_x)
        return y.reshape(xshape)

    def set_event_patterns(self, event_pat):
        """Set HMM event patterns manually

        Rather than fitting the event patterns automatically using fit(), this
        function allows them to be set explicitly. They can then be used to
        find corresponding events in a new dataset, using find_events().

        Parameters
        ----------
        event_pat: voxel by event ndarray
        """
        if event_pat.shape[1] != self.n_events:
            raise ValueError(("Number of columns of event_pat must match "
                              "number of events"))
        self.event_pat_ = event_pat.copy()

    def find_events(self, testing_data, var=None, scramble=False):
        """Applies learned event segmentation to new testing dataset

        After fitting an event segmentation using fit() or setting event
        patterns directly using set_event_patterns(), this function finds the
        same sequence of event patterns in a new testing dataset.

        Parameters
        ----------
        testing_data: timepoint by voxel ndarray
            fMRI data to segment based on previously-learned event patterns

        var: float or 1D ndarray of length equal to the number of events
            default: uses variance that maximized training log-likelihood
            Variance of the event Gaussians. If scalar, all events are
            assumed to have the same variance. If fit() has not previously
            been run, this must be specifed (cannot be None).

        scramble: bool : default False
            If true, the order of the learned events are shuffled before
            fitting, to give a null distribution

        Returns
        -------
        segments : time by event ndarray
            The resulting soft segmentation. segments[t,e] = probability
            that timepoint t is in event e

        test_ll : float
            Log-likelihood of model fit
        """

        if var is None:
            if not hasattr(self, 'event_var_'):
                raise NotFittedError(("The event patterns must first be set "
                                      "by fit() or set_event_patterns()"))
            else:
                var = self.event_var_

        if scramble:
            mean_pat = self.event_pat_[:, np.random.permutation(self.n_events)]
        else:
            mean_pat = self.event_pat_

        logprob = self._logprob_obs(testing_data.T, mean_pat, var)
        lg, test_ll = self._forward_backward(logprob)
        segments = np.exp(lg)

        return segments, test_ll

    def predict(self, X):
        """Applies learned event segmentation to new testing dataset

        Alternative function for segmenting a new dataset after using
        fit() to learn a sequence of events, to comply with the sklearn
        Classifier interface

        Parameters
        ----------
        X: timepoint by voxel ndarray
            fMRI data to segment based on previously-learned event patterns

        Returns
        -------
        Event label for each timepoint
        """
        check_is_fitted(self, ["event_pat_", "event_var_"])
        X = check_array(X)
        segments, test_ll = self.find_events(X)
        return np.argmax(segments, axis=1)

    def calc_weighted_event_var(self, D, weights, event_pat):
        """Computes normalized weighted variance around event pattern

        Utility function for computing variance in a training set of weighted
        event examples. For each event, the sum of squared differences for all
        timepoints from the event pattern is computed, and then the weights
        specify how much each of these differences contributes to the
        variance (normalized by the number of voxels).

        Parameters
        ----------
        D : timepoint by voxel ndarray
            fMRI data for which to compute event variances

        weights : timepoint by event ndarray
            specifies relative weights of timepoints for each event

        event_pat : voxel by event ndarray
            mean event patterns to compute variance around

        Returns
        -------
        ev_var : ndarray of variances for each event
        """
        Dz = stats.zscore(D, axis=1, ddof=1)
        ev_var = np.empty(event_pat.shape[1])
        for e in range(event_pat.shape[1]):
            # Only compute variances for weights > 0.1% of max weight
            nz = weights[:, e] > np.max(weights[:, e])/1000
            sumsq = np.dot(weights[nz, e],
                           np.sum(np.square(Dz[nz, :] -
                                  event_pat[:, e]), axis=1))
            ev_var[e] = sumsq/(np.sum(weights[nz, e]) -
                               np.sum(np.square(weights[nz, e])) /
                               np.sum(weights[nz, e]))
        ev_var = ev_var / D.shape[1]
        return ev_var
