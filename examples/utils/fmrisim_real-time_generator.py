# Generate simulated fMRI data with a few parameters that might be relevant
# for real time analysis
# This code can be run as a function in python or from the command line:
# python fmrisim_real-time_generator --inputDir fmrisim_files/ --outputDir data/
#
# The input arguments are:
# Required:
# inputDir - Specify input data dir where the parameters for fmrisim are
# outputDir - Specify output data dir where the data should be saved
#
# Optional (can be modified by flags from the command line):
# data_dict contains:
#     numTRs - Specify the number of time points
#     multivariate_patterns - Is the difference between conditions univariate
#  (0) or multivariate (1)
#     different_ROIs - Are there different ROIs for each condition (1) or is
# it in the same ROI (0). If it is the same ROI and you are using univariate
# differences, the second condition will have a smaller evoked response than
#  the other.
#     event_duration - How long, in seconds, is each event
#     scale_percentage - What is the percent signal change
#     trDuration - How many seconds per volume
#     save_dicom - Do you want to save data as a dicom (1) or numpy (0)
#     save_realtime - Do you want to save the data in real time (1) or as
# fast as possible (0)?
#     isi - What is the time between each event (in seconds)
#     burn_in - How long before the first event (in seconds)

import os
import glob
import time
import random
import argparse
import datetime
import nibabel  # type: ignore
import numpy as np  # type: ignore
import pydicom as dicom
from brainiak.utils import fmrisim as sim  # type: ignore
import sys

def generate_ROIs(ROI_file,
                  stimfunc,
                  noise,
                  scale_percentage,
                  data_dict):
    # Create the signal in the ROI as specified.

    print('Loading', ROI_file)

    nii = nibabel.load(ROI_file)
    ROI = nii.get_data()

    # Find all the indices that contain signal
    idx_list = np.where(ROI == 1)

    idxs = np.zeros([len(idx_list[0]), 3])
    for idx_counter in list(range(0, len(idx_list[0]))):
        idxs[idx_counter, 0] = int(idx_list[0][idx_counter])
        idxs[idx_counter, 1] = int(idx_list[1][idx_counter])
        idxs[idx_counter, 2] = int(idx_list[2][idx_counter])

    idxs = idxs.astype('int8')

    # How many voxels per ROI
    voxels = int(ROI.sum())

    # Create a pattern of activity across the two voxels
    if data_dict['multivariate_pattern'] is True:
        pattern = np.random.rand(voxels).reshape((voxels, 1))
    else:  # Just make a univariate increase
        pattern = np.tile(1, voxels).reshape((voxels, 1))

    # Multiply each pattern by each voxel time course
    weights = np.tile(stimfunc, voxels) * pattern.T

    # Convolve the onsets with the HRF
    temporal_res = 1 / data_dict['trDuration']
    signal_func = sim.convolve_hrf(stimfunction=weights,
                                   tr_duration=data_dict['trDuration'],
                                   temporal_resolution=temporal_res,
                                   scale_function=1,
                                   )

    # Change the type of noise
    noise = noise.astype('double')

    # Create a noise function (same voxels for signal function as for noise)
    noise_function = noise[idxs[:, 0], idxs[:, 1], idxs[:, 2], :].T

    # Compute the signal magnitude for the data
    sf_scaled = sim.compute_signal_change(signal_function=signal_func,
                                          noise_function=noise_function,
                                          noise_dict=data_dict['noise_dict'],
                                          magnitude=[scale_percentage],
                                          method='PSC',
                                          )

    # Combine the signal time course with the signal volume
    signal = sim.apply_signal(sf_scaled,
                              ROI,
                              )

    # Return signal needed
    return signal


def write_dicom(output_name,
                data):
    # Write the data to a dicom file.
    # Dicom files are difficult to set up correctly, this file will likely
    # crash when trying to open it using dcm2nii. However, if it is loaded in
    # python (e.g., dicom.dcmread) then pixel_array contains the relevant
    # voxel data

    # Convert data from float to in
    dataInts = data.astype(np.int16)

    # Populate required values for file meta information
    file_meta = dicom.Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2'  # '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'

    # Create the FileDataset
    ds = dicom.FileDataset(output_name,
                           {},
                           file_meta=file_meta,
                           preamble=b"\0" * 128)

    # Set image dimensions
    frames, rows, cols = dataInts.shape
    ds.Rows = rows
    ds.Columns = cols
    ds.NumberOfFrames = frames
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.PixelRepresentation = 0

    # Add the data elements -- not trying to set all required here. Check DICOM
    # standard
    ds.PatientName = "sim"
    ds.PatientID = "sim"

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    # Add the data
    ds.PixelData = dataInts.tobytes()

    ds.save_as(output_name)


def generate_data(inputDir,
                  outputDir,
                  data_dict):
    # Generate simulated fMRI data with a few parameters that might be
    # relevant for real time analysis
    # inputDir - Specify input data dir where the parameters for fmrisim are
    # outputDir - Specify output data dir where the data should be saved
    # data_dict contains:
    #     numTRs - Specify the number of time points
    #     multivariate_patterns - Is the difference between conditions
    # univariate (0) or multivariate (1)
    #     different_ROIs - Are there different ROIs for each condition (1) or
    #  is it in the same ROI (0). If it is the same ROI and you are using
    # univariate differences, the second condition will have a smaller evoked
    #  response than the other.
    #     event_duration - How long, in seconds, is each event
    #     scale_percentage - What is the percent signal change
    #     trDuration - How many seconds per volume
    #     save_dicom - Do you want to save data as a dicom (1) or numpy (0)
    #     save_realtime - Do you want to save the data in real time (1) or as
    #  fast as possible (0)?
    #     isi - What is the time between each event (in seconds)
    #     burn_in - How long before the first event (in seconds)

    # If the folder doesn't exist then make it
    if os.path.isdir(data_dir) is False:
        os.makedirs(data_dir, exist_ok=True)

    print('Load template of average voxel value')
    template_nii = nibabel.load(fmrisim_dir + 'sub_template.nii.gz')
    template = template_nii.get_data()

    dimensions = np.array(template.shape[0:3])

    print('Create binary mask and normalize the template range')
    mask, template = sim.mask_brain(volume=template,
                                    mask_self=True,
                                    )

    # Write out the mask as a numpy file
    np.save(data_dir + 'mask.npy', mask.astype(np.uint8))

    # Load the noise dictionary
    print('Loading noise parameters')
    with open(fmrisim_dir + 'sub_noise_dict.txt', 'r') as f:
        noise_dict = f.read()
    noise_dict = eval(noise_dict)
    noise_dict['matched'] = 0  # Increases processing time

    # Add it here for easy access
    data_dict['noise_dict'] = data_dict

    print('Generating noise')
    temp_stimfunction = np.zeros((data_dict['numTRs'], 1))
    noise = sim.generate_noise(dimensions=dimensions,
                               stimfunction_tr=temp_stimfunction,
                               tr_duration=int(data_dict['trDuration']),
                               template=template,
                               mask=mask,
                               noise_dict=noise_dict,
                               )

    # Create the stimulus time course of the conditions
    total_time = int(data_dict['numTRs'] * data_dict['trDuration'])
    onsets_A = []
    onsets_B = []
    curr_time = data_dict['burn_in']
    while curr_time < (total_time - data_dict['event_duration']):

        # Flip a coin for each epoch to determine whether it is A or B
        if np.random.randint(0, 2) == 1:
            onsets_A.append(curr_time)
        else:
            onsets_B.append(curr_time)

        # Increment the current time
        curr_time += data_dict['event_duration'] + data_dict['isi']

    # How many timepoints per second of the stim function are to be generated?
    temporal_res = 1 / data_dict['trDuration']

    # Create a time course of events
    event_durations = [data_dict['event_duration']]
    stimfunc_A = sim.generate_stimfunction(onsets=onsets_A,
                                           event_durations=event_durations,
                                           total_time=total_time,
                                           temporal_resolution=temporal_res,
                                           )

    stimfunc_B = sim.generate_stimfunction(onsets=onsets_B,
                                           event_durations=event_durations,
                                           total_time=total_time,
                                           temporal_resolution=temporal_res,
                                           )

    # Create a labels timecourse
    np.save(data_dir + 'labels.npy', (stimfunc_A + (stimfunc_B * 2)))

    # How is the signal implemented in the different ROIs
    signal_A = generate_ROIs(fmrisim_dir + 'ROI_A.nii.gz',
                             stimfunc_A,
                             noise,
                             data_dict['scale_percentage'],
                             data_dict)
    if data_dict['different_ROIs'] is True:

        signal_B = generate_ROIs(fmrisim_dir + 'ROI_B.nii.gz',
                                 stimfunc_B,
                                 noise,
                                 data_dict['scale_percentage'],
                                 data_dict)

    else:

        # Halve the evoked response if these effects are both expected in the same ROI
        if data_dict['multivariate_pattern'] is False:
            signal_B = generate_ROIs(fmrisim_dir + 'ROI_A.nii.gz',
                                     stimfunc_B,
                                     noise,
                                     data_dict['scale_percentage'] * 0.5,
                                     data_dict)
        else:
            signal_B = generate_ROIs(fmrisim_dir + 'ROI_A.nii.gz',
                                     stimfunc_B,
                                     noise,
                                     data_dict['scale_percentage'],
                                     data_dict)

    # Combine the two signal timecourses
    signal = signal_A + signal_B

    print('Generating TRs in real time')
    for idx in range(data_dict['numTRs']):

        #  Create the brain volume on this TR
        brain = noise[:, :, :, idx] + signal[:, :, :, idx]

        # Convert file to integers to mimic what you get from MR
        brain_int32 = brain.astype(np.int32)

        # Store as dicom or nifti?
        if data_dict['save_dicom'] is True:
            # Save the volume as a DICOM file, with each TR as its own file
            output_file = data_dir + 'rt_' + format(idx, '03d') + '.dcm'
            write_dicom(output_file, brain_int32)
        else:
            # Save the volume as a numpy file, with each TR as its own file
            output_file = data_dir + 'rt_' + format(idx, '03d') + '.npy'
            np.save(output_file, brain_int32)

        print("Generate {}".format(output_file))

        # Sleep until next TR
        if data_dict['save_realtime'] == 1:
            time.sleep(data_dict['trDuration'])


if __name__ == '__main__':
    # Receive the inputs
    argParser = argparse.ArgumentParser(
        'Specify input arguments. Some arguments are parameters that require '
        'an input is provided (noted by "Param"), others are flags that when '
        'provided will change according to the flag (noted by "Flag")')
    argParser.add_argument('--inputDir', '-i', default=None, type=str,
                           help='Param. Input directory for fmrisim parameters')
    argParser.add_argument('--outputDir', '-o', default=None, type=str,
                           help='Param. Output directory for simulated data')
    argParser.add_argument('--numTRs', '-n', default=200, type=int,
                           help='Param. Number of time points')
    argParser.add_argument('--eventDuration', '-d', default=10, type=int,
                           help='Param. Number of seconds per event')
    argParser.add_argument('--signalScale', '-s', default=0.5, type=float,
                           help='Param. Percent signal change')
    argParser.add_argument('--useMultivariate', '-m', default=False,
                           action='store_true',
                           help='Flag. Signal is different between conditions '
                                'in a multivariate, versus univariate, way')
    argParser.add_argument('--useDifferentROIs', '-r', default=False,
                           action='store_true', help='Flag. Use different '
                                                     'ROIs for each condition')
    argParser.add_argument('--saveAsDicom', default=False, action='store_true',
                           help='Flag. Output files in DICOM format rather '
                                'than numpy')
    argParser.add_argument('--saveRealtime', default=False, action='store_true',
                           help='Flag. Save data as if it was coming in at '
                                'the acquisition rate')
    args = argParser.parse_args()

    inputDir = args.inputDir
    outputDir = args.outputDir

    if fmrisim_dir is None or data_dir is None:
        print("Must specify an input and output directory using -i and -o")
        exit(-1)

    data_dict = {}

    ## User controlled settings

    # Specify the number of time points
    data_dict['numTRs'] = args.numTRs

    # How long is each event/block you are modelling (assumes 6s rest between)
    data_dict['event_duration'] = float(args.eventDuration)

    # What is the percent signal change being simulated
    data_dict['scale_percentage'] = args.signalScale

    # Are there different ROIs for each condition (True) or is it in the same
    #  ROI (False).
    # If it is the same ROI and you are using univariate differences,
    # the second condition will have a smaller evoked response than the other.
    data_dict['different_ROIs'] = args.useDifferentROIs

    # Is this a multivariate pattern (1) or a univariate pattern
    data_dict['multivariate_pattern'] = args.useMultivariate

    # Do you want to save data as a dicom (True) or numpy (False)
    data_dict['save_dicom'] = args.saveAsDicom

    # Do you want to save the data in real time (1) or as fast as possible (0)?
    data_dict['save_realtime'] = args.saveRealtime

    ## Default settings

    # How long does each acquisition take
    data_dict['trDuration'] = 2

    # What is the time between each event (in seconds)
    data_dict['isi'] = 6

    # How long before the first event (in seconds)
    data_dict['burn_in'] = 6

    # Run the function if running from command line
    generate_data(inputDir,
                  outputDir,
                  data_dict)
