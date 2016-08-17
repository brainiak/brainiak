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

import argparse, subprocess

fcma_dict = {'directory': '-d', 'task': '-k', 'nfolds': '-n', 'quiet': '-q', 'leaveout': '-l', 'shuffle': '-f',
             'blockdir': '-b', 'filename': '-m', 'xmask': '-x', 'test': '-c', 'output': '-t', 'ymask': '-y',
             'permutebook': '-p', 'visualizeid': '-v', 'blockfile': '-b', 'step': '-s', 'hold': '-h'}
mpi_dict = {'mpiprocess': '-n', 'mpihost': '-hosts', 'mpihostfile': '-f'}

def generate_arguments(input_dict):
    mpi_args = []
    fcma_args = []
    for key, value in input_dict.items():
        if key is 'fcma':
            continue
        if value is not None:
            if 'mpi' in key:
                mpi_args.append(mpi_dict[key])
                mpi_args.append(value)
            else:
                fcma_args.append(fcma_dict[key])
                if value is True:
                    fcma_args.append('1')
                elif value is False:
                    fcma_args.append('0')
                else:
                    fcma_args.append(value)
    return mpi_args, fcma_args

# this script assumes that FCMA has been installed as "FCMA"
if __name__ == '__main__':
    positional_args = [
        (('fcma',), dict(help='path to FCMA executable file')),
        (('directory',), dict(help='directory of NIfTI files')),
        (('filename',), dict(help='acceptable file type, usually the extension name, e.g. nii.gz')),
        (('task',), dict(help='0 for voxel selection using svm, 1 for smart distance ratio, 2 for searchlight, \
                              3 for correlation sum, 4 for two parts correlation and test, \
                              5 for cross validation of two parts correlation, 6 for one part activation and test, \
                              7 for cross validation of one part activation, 8 for N-1 voxel correlation visualization, \
                              9 for marginal screening')),
        (('output',), dict(help='output file for task 0,1,2,3 in the voxel selection mode, input file for the same \
                              tasks in the test mode')),
    ]
    optional_args = [
    ]

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-b', '--blockfile', type=str,
                            help='block information file, if no block information file, \
                            a block information directory is required')
    arg_parser.add_argument('-e', '--blockdir', type=str,
                            help='block directory name, will check this if -b is not provided')
    arg_parser.add_argument('-s', '--step', type=str,
                            help='step, the number of rows to be assigned per round, default 100')
    arg_parser.add_argument('-l', '--leaveout', type=str,
                            help='leave out id, the first block id that being left out, default -1, \
                            which means don\'t leave out anything')
    arg_parser.add_argument('--hold', type=str,
                            help='number of items that held for test, default -1, \
                            only and must get values when -l is applied')
    arg_parser.add_argument('--test', action='store_true',
                            help='test mode (True) or not (False), default False')
    arg_parser.add_argument('-n', '--nfolds', type=str,
                            help='number of folds in the feature selection, default 0')
    arg_parser.add_argument('-x', '--xmask', type=str,
                            help='the first mask file, default no mask')
    arg_parser.add_argument('-y', '--ymask', type=str,
                            help='the second mask file, default no mask')
    arg_parser.add_argument('-v', '--visualizeid', type=str,
                            help='the block id that you want to visualize the correlation, must be specified in task 8')
    arg_parser.add_argument('-q', '--quiet', action='store_false',
                            help='being quiet (True) or not (False) in test mode for task type 2 4 5 6 7, default True')
    arg_parser.add_argument('--shuffle', type=str,
                            help='randomly shuffle the voxel data, 0-no shuffling, \
                            1-shuffle using a random time seed (results are not repeatable), \
                            2-shuffle using an input permuting book (results are repeatable), default 0')
    arg_parser.add_argument('--permutebook', type=str,
                            help='permuting book, only get values when -f is set to 2; \
                            the book contains #subjects row and #voxels column; \
                            each row contains random shuffling result of 0 to #voxels-1')
    arg_parser.add_argument('--mpiprocess', type=str,
                            help='number of mpi process used, only parse this when task=0')
    arg_parser.add_argument('--mpihost', type=str,
                            help='mpi host names, comma separated host list, only parse this when task=0')
    arg_parser.add_argument('--mpihostfile', type=str,
                            help='mpi host file, file containing the host names, only parse this when task=0')

    pg = arg_parser.add_argument_group('fcma arguments')
    for args, kwargs in positional_args:
        pg.add_argument(*args, **kwargs)
    og = arg_parser.add_argument_group('fcma options')
    for args, kwargs in optional_args:
        og.add_argument(*args, **kwargs)
    args = arg_parser.parse_args()
    input_dict = vars(args)
    (mpi_args, fcma_args) = generate_arguments(input_dict)
    if input_dict['task'] is '0':
        subprocess.check_call(['mpirun'] + mpi_args + ['./FCMA'] + fcma_args, cwd=input_dict['fcma'])
    else:
        subprocess.check_call(['./FCMA'] + fcma_args, cwd=input_dict['fcma'])
