# Generate simulated fMRI data with a few parameters that might be relevant
# for real time analysis
"""
This code can be run as a function in python or from the command line:
python fmrisim_real-time_generator --outputDir data

The input arguments are:
Required:
outputDir - Specify output data dir where the data should be saved

Optional (can be modified by flags from the command line):
data_dict contains:
numTRs - Specify the number of time points
multivariate_patterns - Is the difference between conditions univariate (0)
or multivariate (1)
different_ROIs - Are there different ROIs for each condition (1) or is
it in the same ROI (0). If it is the same ROI and you are using univariate
differences, the second condition will have a smaller evoked response than
the other.
event_duration - How long, in seconds, is each event
scale_percentage - What is the percent signal change
trDuration - How many seconds per volume
save_dicom - Do you want to save data as a dicom (1) or numpy (0)
save_realtime - Do you want to save the data in real time (1) or as fast as
possible (0)?
isi - What is the time between each event (in seconds)
burn_in - How long before the first event (in seconds)
"""
import os
import time
import argparse
import datetime
import nibabel  # type: ignore
import numpy as np  # type: ignore
import pydicom as dicom
from brainiak.utils import fmrisim as sim
import logging
from nibabel.nifti1 import Nifti1Image
import gzip

from pathlib import Path

__all__ = ["generate_data"]

from importlib.resources import files

logger = logging.getLogger(__name__)

script_datetime = datetime.datetime.now()

default_settings = {
    'ROI_A_file': None,
    'ROI_B_file': None,
    'template_path': None,
    'noise_dict_file': None,
    'numTRs': 200,
    'trDuration': 2,
    'isi': 6,
    'burn_in': 6,
    'event_duration': 10,
    'scale_percentage': 0.5,
    'multivariate_pattern': False,
    'different_ROIs': False,
    'save_dicom': False,
    'save_realtime': False,
}


def _generate_ROIs(ROI_file,
                   stimfunc,
                   noise,
                   scale_percentage,
                   data_dict):
    """Make signal activity for an ROI of data
        Creates the specified evoked response time course, calibrated to the
        expected signal change, for a given ROI

    Parameters
    ----------

    ROI_file : str
        Path to the file of the ROI being loaded in

    stimfunc : 1 dimensional array
        Time course of evoked response. Output from
        fmrisim.generate_stimfunction

    noise : 4 dimensional array
        Volume of noise generated from fmrisim.generate_noise. Although this
        is needed as an input, this is only so that the percent signal change
        can be calibrated. This is not combined with the signal generated.

    scale_percentage : float
        What is the percent signal change for the evoked response

    data_dict : dict
        A dictionary to specify the parameters used for making data,
        specifying the following keys
        numTRs - int - Specify the number of time points
        multivariate_patterns - bool - Is the difference between conditions
        univariate (0) or multivariate (1)
        different_ROIs - bool - Are there different ROIs for each condition (
        1) or is it in the same ROI (0). If it is the same ROI and you are
        using univariate differences, the second condition will have a
        smaller evoked response than the other.
        event_duration - int - How long, in seconds, is each event
        scale_percentage - float - What is the percent signal change
        trDuration - float - How many seconds per volume
        save_dicom - bool - Save to data as a dicom (1) or numpy (0)
        save_realtime - bool - Do you want to save the data in real time (1)
        or as fast as possible (0)?
        isi - float - What is the time between each event (in seconds)
        burn_in - int - How long before the first event (in seconds)

    Returns
    ----------

    signal : 4 dimensional array
    Volume of signal in the specified ROI (noise has not yet been added)

    """

    # Create the signal in the ROI as specified.

    # Load in the template data (it may already be loaded if doing a test)
    if isinstance(ROI_file, str):
        logger.info('Loading', ROI_file)
        nii = nibabel.load(ROI_file)
        ROI = nii.get_fdata()
    else:
        ROI = ROI_file

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


def _write_dicom(output_name,
                 data,
                 image_number=0):
    """Write the data to a dicom file
    Saves the data for one TR to a dicom.

    Dicom files are difficult to set up correctly, this file will likely
    crash when trying to open it using dcm2nii. However, if it is loaded in
    python (e.g., dicom.dcmread) then pixel_array contains the relevant
    voxel data

    Parameters
    ----------

    output_name : str
        Output name for volume being created

    data : 3 dimensional array
        Volume of data to be saved

    image_number : int
        Number dicom to be saved. This is critical for setting up dicom file
        header information.

    """

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
    ds.InstanceNumber = image_number
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [.01, 0, 0, 0, 0, 0]
    ds.PhotometricInterpretation = 'MONOCHROME1'

    # Add the data elements -- not trying to set all required here. Check DICOM
    # standard
    ds.PatientName = "sim"
    ds.PatientID = "sim"

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set creation date/time
    image_datetime = script_datetime + datetime.timedelta(seconds=image_number)
    timeStr = image_datetime.strftime('%H%M%S')
    ds.ContentDate = image_datetime.strftime('%Y%m%d')
    ds.ContentTime = timeStr

    # Add the data
    ds.PixelData = dataInts.tobytes()

    ds.save_as(output_name)


def _get_input_names(data_dict):
    """Get names from dict
        Read in the data_dict to return the relevant file names. Will also
        add the default values if trDuration, isi, or burn_in haven't been set

    Parameters
    ----------

    data_dict : dict
        A dictionary to specify the parameters used for making data,
        specifying the following keys
        numTRs - int - Specify the number of time points
        multivariate_patterns - bool - Is the difference between conditions
        univariate (0) or multivariate (1)
        different_ROIs - bool - Are there different ROIs for each condition (
        1) or is it in the same ROI (0). If it is the same ROI and you are
        using univariate differences, the second condition will have a
        smaller evoked response than the other.
        event_duration - int - How long, in seconds, is each event
        scale_percentage - float - What is the percent signal change
        trDuration - float - How many seconds per volume
        save_dicom - bool - Save to data as a dicom (1) or numpy (0)
        save_realtime - bool - Do you want to save the data in real time (1)
        or as fast as possible (0)?
        isi - float - What is the time between each event (in seconds)
        burn_in - int - How long before the first event (in seconds)

    Returns
    ----------

    ROI_A_file : str
        Path to ROI for condition A

    ROI_B_file : str
        Path to ROI for condition B

    template_path : str
        Path to template file for data

    noise_dict_file : str
        Path to file containing parameters for noise simulation

    """

    # Load in the ROIs
    if data_dict.get('ROI_A_file') is None:
        rf = files('brainiak').joinpath('utils/sim_parameters/ROI_A.nii.gz')
        with rf.open("rb") as f:
            vol = f.read()
        ROI_A_file = Nifti1Image.from_bytes(gzip.decompress(vol)).get_fdata()
    else:
        ROI_A_file = data_dict['ROI_A_file']

    if data_dict.get('ROI_B_file') is None:
        rf = files('brainiak').joinpath('utils/sim_parameters/ROI_B.nii.gz')
        with rf.open("rb") as f:
            vol = f.read()
        ROI_B_file = Nifti1Image.from_bytes(gzip.decompress(vol)).get_fdata()
    else:
        ROI_B_file = data_dict['ROI_B_file']

    # Get the path to the template
    if data_dict.get('template_path') is None:
        rf = files('brainiak').joinpath(
            'utils/sim_parameters/sub_template.nii.gz')
        with rf.open("rb") as f:
            vol = f.read()
        template_path = Nifti1Image.from_bytes(
            gzip.decompress(vol)).get_fdata()
    else:
        template_path = data_dict['template_path']

    # Load in the noise dict if supplied
    if data_dict.get('noise_dict_file') is None:
        rf = files('brainiak').joinpath(
            'utils/sim_parameters/sub_noise_dict.txt')
        with rf.open("rb") as f:
            noise_dict_file = f.read()
    else:
        noise_dict_file = data_dict['noise_dict_file']

    # Return the paths
    return ROI_A_file, ROI_B_file, template_path, noise_dict_file


def generate_data(outputDir,
                  user_settings):
    """Generate simulated fMRI data
    Use a few parameters that might be relevant for real time analysis

    Parameters
    ----------

    outputDir : str
        Specify output data dir where the data should be saved

    user_settings : dict
        A dictionary to specify the parameters used for making data,
        specifying the following keys
        numTRs - int - Specify the number of time points
        multivariate_patterns - bool - Is the difference between conditions
        univariate (0) or multivariate (1)
        different_ROIs - bool - Are there different ROIs for each condition (
        1) or is it in the same ROI (0). If it is the same ROI and you are
        using univariate differences, the second condition will have a
        smaller evoked response than the other.
        event_duration - int - How long, in seconds, is each event
        scale_percentage - float - What is the percent signal change
        trDuration - float - How many seconds per volume
        save_dicom - bool - Save to data as a dicom (1) or numpy (0)
        save_realtime - bool - Do you want to save the data in real time (1)
        or as fast as possible (0)?
        isi - float - What is the time between each event (in seconds)
        burn_in - int - How long before the first event (in seconds)

    """
    data_dict = default_settings.copy()
    data_dict.update(user_settings)

    # If the folder doesn't exist then make it
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    logger.info('Load template of average voxel value')

    # Get the file names needed for loading in the data
    ROI_A_file, ROI_B_file, template_path, noise_dict_file = \
        _get_input_names(data_dict)

    # Load in the template data (it may already be loaded if doing a test)
    if isinstance(template_path, str):
        template_nii = nibabel.load(template_path)
        template = template_nii.get_fdata()
    else:
        template = template_path

    dimensions = np.array(template.shape[0:3])

    logger.info('Create binary mask and normalize the template range')
    mask, template = sim.mask_brain(volume=template,
                                    mask_self=True,
                                    )

    # Write out the mask as a numpy file
    outFile = os.path.join(outputDir, 'mask.npy')
    np.save(outFile, mask.astype(np.uint8))

    # Load the noise dictionary
    logger.info('Loading noise parameters')

    # If this isn't a string, assume it is a resource stream file
    if type(noise_dict_file) is str:
        with open(noise_dict_file, 'r') as f:
            noise_dict = f.read()
    else:
        # Read the resource stream object
        noise_dict = noise_dict_file.decode()

    noise_dict = eval(noise_dict)
    noise_dict['matched'] = 0  # Increases processing time

    # Add it here for easy access
    data_dict['noise_dict'] = noise_dict

    logger.info('Generating noise')
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
    outFile = os.path.join(outputDir, 'labels.npy')
    np.save(outFile, (stimfunc_A + (stimfunc_B * 2)))

    # How is the signal implemented in the different ROIs
    signal_A = _generate_ROIs(ROI_A_file,
                              stimfunc_A,
                              noise,
                              data_dict['scale_percentage'],
                              data_dict)
    if data_dict['different_ROIs'] is True:

        signal_B = _generate_ROIs(ROI_B_file,
                                  stimfunc_B,
                                  noise,
                                  data_dict['scale_percentage'],
                                  data_dict)

    else:

        # Halve the evoked response if these effects are both expected in the
        #  same ROI
        if data_dict['multivariate_pattern'] is False:
            signal_B = _generate_ROIs(ROI_A_file,
                                      stimfunc_B,
                                      noise,
                                      data_dict['scale_percentage'] * 0.5,
                                      data_dict)
        else:
            signal_B = _generate_ROIs(ROI_A_file,
                                      stimfunc_B,
                                      noise,
                                      data_dict['scale_percentage'],
                                      data_dict)

    # Combine the two signal timecourses
    signal = signal_A + signal_B

    logger.info('Generating TRs in real time')
    for idx in range(data_dict['numTRs']):

        #  Create the brain volume on this TR
        brain = noise[:, :, :, idx] + signal[:, :, :, idx]

        # Convert file to integers to mimic what you get from MR
        brain_int32 = brain.astype(np.int32)

        # Store as dicom or nifti?
        if data_dict['save_dicom'] is True:
            # Save the volume as a DICOM file, with each TR as its own file
            output_file = os.path.join(outputDir, 'rt_' + format(idx, '03d')
                                       + '.dcm')
            _write_dicom(output_file, brain_int32, idx+1)
        else:
            # Save the volume as a numpy file, with each TR as its own file
            output_file = os.path.join(outputDir, 'rt_' + format(idx, '03d')
                                       + '.npy')
            np.save(output_file, brain_int32)

        logger.info("Generate {}".format(output_file))

        # Sleep until next TR
        if data_dict['save_realtime'] == 1:
            time.sleep(data_dict['trDuration'])


if __name__ == '__main__':
    # Receive the inputs
    argParser = argparse.ArgumentParser(
        'Specify input arguments. Some arguments are parameters that require '
        'an input is provided (noted by "Param"), others are flags that when '
        'provided will change according to the flag (noted by "Flag")')
    argParser.add_argument('--output-dir', '-o', default=None, type=str,
                           help='Param. Output directory for simulated data')
    argParser.add_argument('--ROI-A-file', default=None, type=str,
                           help='Param. Full path to file for cond. A ROI')
    argParser.add_argument('--ROI-B-file', default=None, type=str,
                           help='Param. Full path to file for cond. B ROI')
    argParser.add_argument('--template-path', default=None, type=str,
                           help='Param. Full path to file for brain template')
    argParser.add_argument('--noise-dict-file', default=None, type=str,
                           help='Param. Full path to file setting noise '
                                'params')
    argParser.add_argument('--numTRs', '-n', default=200, type=int,
                           help='Param. Number of time points')
    argParser.add_argument('--event-duration', '-d', default=10, type=int,
                           help='Param. Number of seconds per event')
    argParser.add_argument('--trDuration', default=2, type=int,
                           help='Param. How many second per volume')
    argParser.add_argument('--isi', default=6, type=int,
                           help='Param. How long in seconds between events')
    argParser.add_argument('--burn-in', default=6, type=int,
                           help='Param. How long before the first event '
                                'begins after the run onset')
    argParser.add_argument('--scale-percentage', '-s', default=0.5, type=float,
                           help='Param. Percent signal change')
    argParser.add_argument('--multivariate-pattern', '-m', default=False,
                           action='store_true',
                           help='Flag. Signal is different between conditions '
                                'in a multivariate, versus univariate, way')
    argParser.add_argument('--different-ROIs', '-r', default=False,
                           action='store_true', help='Flag. Use different '
                                                     'ROIs for each condition')
    argParser.add_argument('--save-dicom', default=False,
                           action='store_true', help='Flag. Output files in '
                                                     'DICOM format rather '
                                                     'than numpy')
    argParser.add_argument('--save-realtime', default=False,
                           action='store_true', help='Flag. Save data as if '
                                                     'it was coming in at '
                                                     'the acquisition rate')
    args = argParser.parse_args()

    # Essential arguments
    outputDir = args.output_dir

    if outputDir is None:
        logger.info("Must specify an output directory using -o")
        exit(-1)

    data_dict = default_settings.copy()

    # User controlled settings

    # Specify the path to the files used for defining ROIs.
    data_dict['ROI_A_file'] = args.ROI_A_file
    data_dict['ROI_B_file'] = args.ROI_B_file

    # Specify where the template
    data_dict['template_path'] = args.template_path
    data_dict['noise_dict_file'] = args.noise_dict_file

    # Specify the number of time points
    data_dict['numTRs'] = args.numTRs

    # How long is each event/block you are modelling (assumes 6s rest between)
    data_dict['event_duration'] = float(args.event_duration)

    # How long does each acquisition take
    data_dict['trDuration'] = args.trDuration

    # What is the time between each event (in seconds)
    data_dict['isi'] = args.isi

    # How long before the first event (in seconds)
    data_dict['burn_in'] = args.burn_in

    # What is the percent signal change being simulated
    data_dict['scale_percentage'] = args.scale_percentage

    # Are there different ROIs for each condition (True) or is it in the same
    #  ROI (False).
    # If it is the same ROI and you are using univariate differences,
    # the second condition will have a smaller evoked response than the other.
    data_dict['different_ROIs'] = args.different_ROIs

    # Is this a multivariate pattern (1) or a univariate pattern
    data_dict['multivariate_pattern'] = args.multivariate_pattern

    # Do you want to save data as a dicom (True) or numpy (False)
    data_dict['save_dicom'] = args.save_dicom

    # Do you want to save the data in real time (1) or as fast as possible (0)?
    data_dict['save_realtime'] = args.save_realtime

    # Run the function if running from command line
    generate_data(outputDir,
                  data_dict)
