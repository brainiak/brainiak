#  Copyright 2016 Intel Corporation, Princeton University
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
import numpy as np
import re
import warnings
import os.path
from .fmrisim import generate_stimfunction, double_gamma_hrf
import logging


"""
Some utility functions that can be used by different algorithms
"""

logger = logging.getLogger(__name__)


def from_tri_2_sym(tri, dim):
    """convert a upper triangular matrix in 1D format
       to 2D symmetric matrix


    Parameters
    ----------

    tri: 1D array
        Contains elements of upper triangular matrix

    dim : int
        The dimension of target matrix.


    Returns
    -------

    symm : 2D array
        Symmetric matrix in shape=[dim, dim]
    """
    symm = np.zeros((dim, dim))
    symm[np.triu_indices(dim)] = tri
    return symm


def from_sym_2_tri(symm):
    """convert a 2D symmetric matrix to an upper
       triangular matrix in 1D format

    Parameters
    ----------

    symm : 2D array
          Symmetric matrix


    Returns
    -------

    tri: 1D array
          Contains elements of upper triangular matrix
    """

    inds = np.triu_indices_from(symm)
    tri = symm[inds]
    return tri


def sumexp_stable(data):
    """Compute the sum of exponents for a list of samples

    Parameters
    ----------

    data : array, shape=[features, samples]
        A data array containing samples.


    Returns
    -------

    result_sum : array, shape=[samples,]
        The sum of exponents for each sample divided by the exponent
        of the maximum feature value in the sample.

    max_value : array, shape=[samples,]
        The maximum feature value for each sample.

    result_exp : array, shape=[features, samples]
        The exponent of each element in each sample divided by the exponent
        of the maximum feature value in the sample.

    Note
    ----

        This function is more stable than computing the sum(exp(v)).
        It useful for computing the softmax_i(v)=exp(v_i)/sum(exp(v)) function.
    """
    max_value = data.max(axis=0)
    result_exp = np.exp(data - max_value)
    result_sum = np.sum(result_exp, axis=0)
    return result_sum, max_value, result_exp


def concatenate_not_none(l, axis=0):
    """Construct a numpy array by stacking not-None arrays in a list

    Parameters
    ----------

    data : list of arrays
        The list of arrays to be concatenated. Arrays have same shape in all
        but one dimension or are None, in which case they are ignored.

    axis : int, default = 0
        Axis for the concatenation


    Returns
    -------

    data_stacked : array
        The resulting concatenated array.
    """
    # Get the indexes of the arrays in the list
    mask = []
    for i in range(len(l)):
        if l[i] is not None:
            mask.append(i)

    # Concatenate them
    l_stacked = np.concatenate([l[i] for i in mask], axis=axis)
    return l_stacked


def cov2corr(cov):
    """Calculate the correlation matrix based on a
        covariance matrix

    Parameters
    ----------

    cov: 2D array

    Returns
    -------

    corr: 2D array
        correlation converted from the covarince matrix


    """
    assert cov.ndim == 2, 'covariance matrix should be 2D array'
    inv_sd = 1 / np.sqrt(np.diag(cov))
    corr = cov * inv_sd[None, :] * inv_sd[:, None]
    return corr


class ReadDesign:
    """A class which has the ability of reading in design matrix in .1D file,
        generated by AFNI's 3dDeconvolve.

    Parameters
    ----------

    fname: string, the address of the file to read.

    include_orth: Boollean, whether to include "orthogonal" regressors in
        the nuisance regressors which are usually head motion parameters.
        All the columns of design matrix are still going to be read in,
        but the attribute cols_used will reflect whether these orthogonal
        regressors are to be included for furhter analysis.
        Note that these are not entered into design_task attribute which
        include only regressors related to task conditions.

    include_pols: Boollean, whether to include polynomial regressors in
        the nuisance regressors which are used to capture slow drift of
        signals.

    Attributes
    ----------

    design: 2d array. The design matrix read in from the csv file.

    design_task: 2d array. The part of design matrix corresponding to
        task conditions.

    n_col: number of total columns in the design matrix.

    column_types: 1d array. the types of each column in the design matrix.
        0 for orthogonal regressors (usually head motion parameters),
        -1 for polynomial basis (capturing slow drift of signals),
        values > 0 for stimulus conditions

    n_basis: scalar. The number of polynomial bases in the designn matrix.

    n_stim: scalar. The number of stimulus conditions.

    n_orth: scalar. The number of orthogoanal regressors (usually head
        motions)

    StimLabels: list. The names of each column in the design matrix.
    """
    def __init__(self, fname=None, include_orth=True, include_pols=True):
        if fname is None:
            # fname is the name of the file to read in the design matrix
            self.design = np.zeros([0, 0])
            self.n_col = 0
            # number of columns (conditions) in the design matrix
            self.column_types = np.ones(0)
            self.n_basis = 0
            self.n_stim = 0
            self.n_orth = 0
            self.StimLabels = []
        else:
            # isAFNI = re.match(r'.+[.](1D|1d|txt)$', fname)
            filename, ext = os.path.splitext(fname)
            # We assume all AFNI 1D files have extension of 1D or 1d or txt
            if ext in ['.1D', '.1d', '.txt']:
                self.read_afni(fname=fname)

        self.include_orth = include_orth
        self.include_pols = include_pols
        # The two flags above dictates whether columns corresponding to
        # baseline drift modeled by polynomial functions of time and
        # columns corresponding to other orthogonal signals (usually motion)
        # are included in nuisance regressors.
        self.cols_task = np.where(self.column_types == 1)[0]
        self.design_task = self.design[:, self.cols_task]
        if np.ndim(self.design_task) == 1:
            self.design_task = self.design_task[:, None]
        # part of the design matrix related to task conditions.
        self.n_TR = np.size(self.design_task, axis=0)
        self.cols_nuisance = np.array([])
        if self.include_orth:
            self.cols_nuisance = np.int0(
                np.sort(np.append(self.cols_nuisance,
                                  np.where(self.column_types == 0)[0])))
        if self.include_pols:
            self.cols_nuisance = np.int0(
                np.sort(np.append(self.cols_nuisance,
                                  np.where(self.column_types == -1)[0])))
        if np.size(self.cols_nuisance) > 0:
            self.reg_nuisance = self.design[:, self.cols_nuisance]
            if np.ndim(self.reg_nuisance) == 1:
                self.reg_nuisance = self.reg_nuisance[:, None]
        else:
            self.reg_nuisance = None
        # Nuisance regressors for motion, baseline, etc.

    def read_afni(self, fname):
        # Read design file written by AFNI
        self.n_basis = 0
        self.n_stim = 0
        self.n_orth = 0
        self.StimLabels = []
        self.design = np.loadtxt(fname, ndmin=2)
        with open(fname) as f:
            all_text = f.read()

        find_n_column = re.compile(
            r'^#[ ]+ni_type[ ]+=[ ]+"(?P<n_col>\d+)[*]', re.MULTILINE)
        n_col_found = find_n_column.search(all_text)
        if n_col_found:
            self.n_col = int(n_col_found.group('n_col'))
            if self.n_col != np.size(self.design, axis=1):
                warnings.warn(
                    'The number of columns in the design matrix'
                    + 'does not match the header information')
                self.n_col = np.size(self.design, axis=1)
        else:
            self.n_col = np.size(self.design, axis=1)

        self.column_types = np.ones(self.n_col)
        # default that all columns are conditions of interest

        find_ColumnGroups = re.compile(
            r'^#[ ]+ColumnGroups[ ]+=[ ]+"(?P<CGtext>.+)"', re.MULTILINE)
        CG_found = find_ColumnGroups.search(all_text)
        if CG_found:
            CG_text = re.split(',', CG_found.group('CGtext'))
            curr_idx = 0
            for CG in CG_text:
                split_by_at = re.split('@', CG)
                if len(split_by_at) == 2:
                    # the first tells the number of columns in this condition
                    # the second tells the condition type
                    n_this_cond = int(split_by_at[0])
                    self.column_types[curr_idx:curr_idx + n_this_cond] = \
                        int(split_by_at[1])
                    curr_idx += n_this_cond
                elif len(split_by_at) == 1 and \
                        not re.search('\..', split_by_at[0]):
                    # Just a number, and not the type like '1..4'
                    self.column_types[curr_idx] = int(split_by_at[0])
                    curr_idx += 1
                else:  # must be a single stimulus condition
                    split_by_dots = re.split('\..', CG)
                    n_this_cond = int(split_by_dots[1])
                    self.column_types[curr_idx:curr_idx + n_this_cond] = 1
                    curr_idx += n_this_cond
            self.n_basis = np.sum(self.column_types == -1)
            self.n_stim = np.sum(self.column_types > 0)
            self.n_orth = np.sum(self.column_types == 0)

        find_StimLabels = re.compile(
            r'^#[ ]+StimLabels[ ]+=[ ]+"(?P<SLtext>.+)"', re.MULTILINE)
        StimLabels_found = find_StimLabels.search(all_text)
        if StimLabels_found:
            self.StimLabels = \
                re.split(r'[ ;]+', StimLabels_found.group('SLtext'))
        else:
            self.StimLabels = []


def gen_design(stimtime_files, scan_duration, TR, style='FSL',
               temp_res=0.01,
               hrf_para={'response_delay': 6, 'undershoot_delay': 12,
                         'response_dispersion': 0.9,
                         'undershoot_dispersion': 0.9,
                         'undershoot_scale': 0.035}):
    """ Generate design matrix based on a list of names of stimulus
        timing files. The function will read each file, and generate
        a numpy array of size [time_points \* condition], where
        time_points equals duration / TR, and condition is the size of
        stimtime_filenames. Each column is the hypothetical fMRI response
        based on the stimulus timing in the corresponding file
        of stimtime_files.
        This function uses generate_stimfunction and double_gamma_hrf
        of brainiak.utils.fmrisim.

    Parameters
    ----------

    stimtime_files: a string or a list of string.
        Each string is the name of the file storing
        the stimulus timing information of one task condition.
        The contents in the files will be interpretated
        based on the style parameter.
        Details are explained under the style parameter.

    scan_duration: float or a list (or a 1D numpy array) of numbers.
        Total duration of each fMRI scan, in unit of seconds.
        If there are multiple runs, the duration should be
        a list (or 1-d numpy array) of numbers.
        If it is a list, then each number in the list
        represents the duration of the corresponding scan
        in the stimtime_files.
        If only a number is provided, it is assumed that
        there is only one fMRI scan lasting for scan_duration.

    TR: float.
        The sampling period of fMRI, in unit of seconds.

    style: string, default: 'FSL'
        Acceptable inputs: 'FSL', 'AFNI'
        The formating style of the stimtime_files.
        'FSL' style has one line for each event of the same condition.
        Each line contains three numbers. The first number is the onset
        of the event relative to the onset of the first scan,
        in units of seconds.
        (Multiple scans should be treated as a concatenated long scan
        for the purpose of calculating onsets.
        However, the design matrix from one scan won't leak into the next).
        The second number is the duration of the event,
        in unit of seconds.
        The third number is the amplitude modulation (or weight)
        of the response.
        It is acceptable to not provide the weight,
        or not provide both duration and weight.
        In such cases, these parameters will default to 1.0.

        'AFNI' style has one line for each scan (run).
        Each line has a few triplets in the format of
        stim_onsets*weight:duration
        (or simpler, see below), separated by spaces.
        For example, 3.2\*2.0:1.5 means that one event starts at 3.2s,
        modulated by weight of 2.0 and lasts for 1.5s.
        If some run does not include a single event
        of a condition (stimulus type), then you can put \*,
        or a negative number, or a very large number in that line.
        Either duration or weight can be neglected. In such
        cases, they will default to 1.0.
        For example, 3.0, 3.0\*1.0, 3.0:1.0 and 3.0\*1.0:1.0 all
        means an event starting at 3.0s, lasting for 1.0s, with
        amplitude modulation of 1.0.

    temp_res: float, default: 0.01
        Temporal resolution of fMRI, in second.

    hrf_para: dictionary
        The parameters of the double-Gamma hemodynamic response function.
        To set different parameters, supply a dictionary with
        the same set of keys as the default, and replace the corresponding
        values with the new values.

    Returns
    -------

    design: 2D numpy array
        design matrix. Each time row represents one TR
        (fMRI sampling time point) and each column represents
        one experiment condition, in the order in stimtime_files

    """
    if np.ndim(scan_duration) == 0:
        scan_duration = [scan_duration]
    scan_duration = np.array(scan_duration)
    assert np.all(scan_duration > TR), \
        'scan duration should be longer than a TR'
    if type(stimtime_files) is str:
        stimtime_files = [stimtime_files]
    assert TR > 0, 'TR should be positive'
    assert style == 'FSL' or style == 'AFNI', 'style can only be FSL or AFNI'
    n_C = len(stimtime_files)  # number of conditions
    n_S = np.size(scan_duration)  # number of scans
    if n_S > 1:
        design = [np.empty([int(np.floor(duration / TR)), n_C])
                  for duration in scan_duration]
    else:
        design = [np.empty([int(np.floor(scan_duration / TR)), n_C])]
    scan_onoff = np.insert(np.cumsum(scan_duration), 0, 0)
    if style == 'FSL':
        design_info = _read_stimtime_FSL(stimtime_files, n_C, n_S, scan_onoff)
    elif style == 'AFNI':
        design_info = _read_stimtime_AFNI(stimtime_files, n_C, n_S, scan_onoff)

    response_delay = hrf_para['response_delay']
    undershoot_delay = hrf_para['undershoot_delay']
    response_dispersion = hrf_para['response_dispersion']
    undershoot_dispersion = hrf_para['undershoot_dispersion']
    undershoot_scale = hrf_para['undershoot_scale']
    # generate design matrix
    for i_s in range(n_S):
        for i_c in range(n_C):
            stimfunction = generate_stimfunction(
                onsets=design_info[i_s][i_c]['onset'],
                event_durations=design_info[i_s][i_c]['duration'],
                total_time=scan_duration[i_s],
                weights=design_info[i_s][i_c]['weight'],
                temporal_resolution=1.0/temp_res)
            design[i_s][:, i_c] = double_gamma_hrf(
                stimfunction, TR, response_delay=response_delay,
                undershoot_delay=undershoot_delay,
                response_dispersion=response_dispersion,
                undershoot_dispersion=undershoot_dispersion,
                undershoot_scale=undershoot_scale, scale_function=0,
                temporal_resolution=1.0/temp_res) * temp_res
            # We multiply the resulting design matrix with
            # the temporal resolution to normalize it.
            # We do not use the internal normalization
            # in double_gamma_hrf because it does not guarantee
            # normalizing with the same constant.
    return np.concatenate(design, axis=0)


def _read_stimtime_FSL(stimtime_files, n_C, n_S, scan_onoff):
    """ Utility called by gen_design. It reads in one or more
        stimulus timing file comforming to FSL style,
        and return a list (size of [#run \* #condition])
        of dictionary including onsets, durations and weights of each event.

    Parameters
    ----------

    stimtime_files: a string or a list of string.
        Each string is the name of the file storing the stimulus
        timing information of one task condition.
        The contents in the files should follow the style of FSL
        stimulus timing files, refer to gen_design.

    n_C: integer, number of task conditions

    n_S: integer, number of scans

    scan_onoff: list of numbers.
        The onset of each scan after concatenating all scans,
        together with the offset of the last scan.
        For example, if 3 scans of duration 100s, 150s, 120s are run,
        scan_onoff is [0, 100, 250, 370]


    Returns
    -------

    design_info: list of stimulus information
        The first level of the list correspond to different scans.
        The second level of the list correspond to different conditions.
        Each item in the list is a dictiornary with keys "onset",
        "duration" and "weight". If one condition includes no event
        in a scan, the values of these keys in that scan of the condition
        are empty lists.


    See also
    --------

    gen_design
    """
    design_info = [[{'onset': [], 'duration': [], 'weight': []}
                    for i_c in range(n_C)] for i_s in range(n_S)]
    # Read stimulus timing files
    for i_c in range(n_C):
        with open(stimtime_files[i_c]) as f:
            for line in f.readlines():
                tmp = line.strip().split()
                i_s = np.where(
                    np.logical_and(scan_onoff[:-1] <= float(tmp[0]),
                                   scan_onoff[1:] > float(tmp[0])))[0]
                if len(i_s) == 1:
                    i_s = i_s[0]
                    design_info[i_s][i_c]['onset'].append(float(tmp[0])
                                                          - scan_onoff[i_s])
                    if len(tmp) >= 2:
                        design_info[i_s][i_c]['duration'].append(float(tmp[1]))
                    else:
                        design_info[i_s][i_c]['duration'].append(1.0)
                    if len(tmp) >= 3:
                        design_info[i_s][i_c]['weight'].append(float(tmp[2]))
                    else:
                        design_info[i_s][i_c]['weight'].append(1.0)
    return design_info


def _read_stimtime_AFNI(stimtime_files, n_C, n_S, scan_onoff):
    """ Utility called by gen_design. It reads in one or more stimulus timing
        file comforming to AFNI style, and return a list
        (size of ``[number of runs \* number of conditions]``)
        of dictionary including onsets, durations and weights of each event.

    Parameters
    ----------

    stimtime_files: a string or a list of string.
        Each string is the name of the file storing the stimulus
        timing information of one task condition.
        The contents in the files should follow the style of AFNI
        stimulus timing files, refer to gen_design.

    n_C: integer, number of task conditions

    n_S: integer, number of scans

    scan_onoff: list of numbers.
        The onset of each scan after concatenating all scans,
        together with the offset of the last scan.
        For example, if 3 scans of duration 100s, 150s, 120s are run,
        scan_onoff is [0, 100, 250, 370]


    Returns
    -------

    design_info: list of stimulus information
        The first level of the list correspond to different scans.
        The second level of the list correspond to different conditions.
        Each item in the list is a dictiornary with keys "onset",
        "duration" and "weight". If one condition includes no event
        in a scan, the values of these keys in that scan of the condition
        are empty lists.


    See also
    --------

    gen_design
    """
    design_info = [[{'onset': [], 'duration': [], 'weight': []}
                    for i_c in range(n_C)] for i_s in range(n_S)]
    # Read stimulus timing files
    for i_c in range(n_C):
        with open(stimtime_files[i_c]) as f:
            text = f.readlines()
            assert len(text) == n_S, \
                'Number of lines does not match number of runs!'
            for i_s, line in enumerate(text):
                events = line.strip().split()
                if events[0] == '*':
                    continue
                for event in events:
                    assert event != '*'
                    tmp = str.split(event, ':')
                    if len(tmp) == 2:
                        duration = float(tmp[1])
                    else:
                        duration = 1.0
                    tmp = str.split(tmp[0], '*')
                    if len(tmp) == 2:
                        weight = float(tmp[1])
                    else:
                        weight = 1.0
                    if (float(tmp[0]) >= 0
                            and float(tmp[0])
                            < scan_onoff[i_s + 1] - scan_onoff[i_s]):
                        design_info[i_s][i_c]['onset'].append(float(tmp[0]))
                        design_info[i_s][i_c]['duration'].append(duration)
                        design_info[i_s][i_c]['weight'].append(weight)
    return design_info


def center_mass_exp(a, b, scale=1.0):
    """ Calculate the center of mass of negative exponential distribution
        p(x) = exp(-x / scale) / scale
        in the interval of (a, b). scale is the same scale
        parameter as scipy.stats.expon.pdf

    Parameters
    ----------
    a: float
        The starting point of the interval in which the center of mass
        is calculated for exponential distribution.
    b: float
        The ending point of the interval in which the center of mass
        is calculated for exponential distribution.
    scale: float
        The scale parameter of the exponential distribution. See above.

    Returns
    -------
    m: float
        The center of mass in the interval of (a, b) for exponential
        distribution.
    """
    assert a < b, 'a must be smaller than b'
    assert b > 0, 'b must be larger than 0'
    assert scale > 0, 'scale must be positive'
    assert a >= 0, 'a cannot be smaller than 0'
    if b < np.inf:
        return ((a + scale) * np.exp(-a / scale)
                - (scale + b) * np.exp(-b / scale)) \
            / (np.exp(-a / scale) - np.exp(-b / scale))
    else:
        return a + scale
