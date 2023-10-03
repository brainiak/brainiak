"""-----------------------------------------------------------------------------

sample.py (Last Updated: 01/26/2021)

The purpose of this script is to run a sample project for the BrainIAK Aperture 
paper. This sample project script will be wrapped by the projectInterface and 
it will be made accessible on the Web Server.

The purpose of this *particular* script is to demonstrated how you can use the
various scripts, functions, etc. we have developed for your use! The functions
we will reference live in 'rt-cloud/rtCommon/'.

Finally, this script is called from the projectServer which is started in the
jupyter notebook.

-----------------------------------------------------------------------------"""


# add the full path to the rtcloud
import os
path_to_rtcloud = os.getenv('RTCLOUD_PATH')
if path_to_rtcloud == None:
    print("Please set RTCLOUD_PATH, see instructions")
    raise ValueError

# import other important modules
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import scipy.io as sio
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
from sklearn import svm
import nilearn
from nilearn.masking import apply_mask
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

import warnings # ignore warnings when importing dicomreaders below
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from nibabel.nicom import dicomreaders

# obtain full path for current directory: '.../rt-cloud/projects/sample'
currPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_to_rtcloud)
# import project modules from rt-cloud
from rtCommon.utils import loadConfigFile
from rtCommon.clientInterface import ClientInterface
from rtCommon.imageHandling import readRetryDicomFromDataInterface, getDicomFileName, convertDicomImgToNifti, convertDicomFileToNifti, readNifti, convertDicomFileToNifti

def doRuns(cfg, dataInterface, subjInterface, webInterface):
    """
    This function is called by 'main()' below. Here, we use the 'dataInterface'
    to read in dicoms (presumably from the scanner, but here it's from a folder
    with previously collected dicom files), doing some sort of analysis in the
    cloud, and then sending the info to the web browser.

    INPUT:
        [1] cfg (configuration file with important variables)
        [2] dataInterface (this will allow a script from the cloud to access files
               from the stimulus computer, which receives dicom files directly
               from the Siemens console computer)
        [3] subjInterface - this allows sending feedback (e.g. classification results)
                to a subjectService running on the presentation computer to provide
                feedback to the subject (and optionally get their response).
    OUTPUT:
        None.
    """

    # variables we'll use throughout
    scanNum = cfg.scanNum[0]
    runNum = cfg.runNum[0]

    # obtain the path for the directory where the subject's dicoms live
    cfg.dicomDir = cfg.imgDir
    print("Location of the subject's dicoms: %s\n" %cfg.dicomDir)

    # Initialize a watch for the entire dicom folder using the function 'initWatch'
    # in dataInterface. (Later we will use watchFile() to look for a specific dicom)
    #   INPUT:
    #       [1] cfg.dicomDir (where the subject's dicom files live)
    #       [2] cfg.dicomNamePattern (the naming pattern of dicom files)
    #       [3] cfg.minExpectedDicomSize (a check on size to make sure we don't
    #               accidentally grab a dicom before it's fully acquired)
    dataInterface.initWatch(cfg.dicomDir, cfg.dicomNamePattern,
        cfg.minExpectedDicomSize)

    #We will use the function plotDataPoint in webInterface whenever we
    #  want to send values to the web browser so that they can be plotted in the
    #  --Data Plots-- tab.
    #However at the start of a run we will want to clear the plot, and we can use
    #clearRunPlot(runId), or clearAllPlots() also in the webInterface object.
    print("\n\nClear any pre-existing plot for this run using 'clearRunPlot(runNum)'")
    webInterface.clearRunPlot(runNum)
    print(''
          '###################################################################################')

    # declare the total number of TRs
    num_trainingData = cfg.numTrainingTRs
    num_total_TRs = cfg.numSynthetic

    for this_TR in np.arange(num_total_TRs):
        # declare variables that are needed to use 'readRetryDicomFromDataInterface'
        timeout_file = 5 # small number because of demo, can increase for real-time

        # use 'getDicomFileName' from 'imageHandling.py' to obtain the filename 
        #   of the dicom data you want to get... which is useful considering how
        #   complicated these filenames can be!
        #   INPUT:
        #       [1] cfg (config parameters)
        #       [2] scanNum (scan number)
        #       [3] fileNum (TR number, which will reference the correct file)
        #   OUTPUT:
        #       [1] fullFileName (the filename of the dicom that should be grabbed)
        fileName = getDicomFileName(cfg, scanNum, this_TR)

        # Use 'readRetryDicomFromDataInterface' in 'imageHandling.py' to wait for dicom
        #    files to be written by the scanner (uses 'watchFile' internally) and then
        #    reading the dicom file once it is available.
        #INPUT:
        #    [1] dataInterface (allows a cloud script to access files from the
        #        control room computer)
        #    [2] filename (the dicom file we're watching for and want to load)
        #    [3] timeout (time spent waiting for a file before timing out)
        #OUTPUT:
        #    [1] dicomData (with class 'pydicom.dataset.FileDataset')
        dicomData = readRetryDicomFromDataInterface(dataInterface, fileName,
            timeout_file)

        # We can use the 'convertDicomFileToNifti' function with dicomData as 
        #   input to get the data in NifTi format
        base, ext = os.path.splitext(fileName)
        assert ext == '.dcm'
        niftiFilename = base + '.nii'
        convertDicomFileToNifti(fileName, niftiFilename)
        niftiObject = readNifti(niftiFilename)
        
        # declare various things if it's the first TR
        if this_TR == 0:
            # load the labels and mask
            labels = np.load(os.path.join(cfg.imgDir, 'labels.npy'))
            print(os.path.join(cfg.imgDir, 'labels.npy'))
            ROI_nib = nib.load(os.path.join(currPath,'ROI_mask.nii.gz'))
            ROI_mask = np.array(ROI_nib.dataobj)
            mask_nib = nib.Nifti1Image(ROI_mask.T, affine=niftiObject.affine)
            # declare the number of TRs we will shift the data to account for the hemodynamic lag
            num_shiftTRs = 3
            # shift the labels to account for hemodynamic lag
            shifted_labels = np.concatenate([np.full((num_shiftTRs, 1), np.nan),labels])
            # set up a matrix that will hold all of the preprocessed data
            preprocessed_data = np.full((num_total_TRs, int(ROI_mask.sum())), np.nan)

        # preprocess the training data by applying the mask
        preprocessed_data[this_TR,:] = np.ravel(apply_mask(niftiObject,mask_nib).reshape(int(ROI_mask.sum()),1))

        ## Now we divide into one of three possible steps

        # STEP 1: collect the training data
        if this_TR < num_trainingData:
            print('Collected training data for TR %s' %this_TR)
        # STEP 2: train the SVM model
        elif this_TR == num_trainingData:
            print(''
                  '###################################################################################')
            print('Done collecting training data! \nStart training the classifier.')

            # snapshot of time to keep track of how long it takes to train the classifier
            start_time = time.time()

            scaler = StandardScaler()
            X_train = preprocessed_data[num_shiftTRs:num_trainingData]
            y_train = shifted_labels[num_shiftTRs:num_trainingData].reshape(-1,)
            # we don't want to include rest data
            X_train_noRest = X_train[y_train != 0]
            y_train_noRest = y_train[y_train != 0]
            # and we want to zscore the bold data
            X_train_noRest_zscored = scaler.fit_transform(X_train_noRest)
            clf = svm.SVC(kernel='linear', C=0.01, class_weight='balanced')
            clf.fit(X_train_noRest_zscored, y_train_noRest)

            # print out the amount of time it took to train the classifier
            print('Classifier done training! Time it took: %.2f s' %(time.time() - start_time))
            print(''
                  '###################################################################################')
        elif this_TR > num_trainingData:
            # apply the classifier to new data to obtain prediction IF not a rest trial
            if shifted_labels[this_TR] != 0:
                prediction = clf.predict(scaler.transform(preprocessed_data[this_TR,:].reshape(1,-1)))
                print(f'Plotting classifier prediction for TR {this_TR}: {prediction}')
                webInterface.plotDataPoint(runNum, int(this_TR), float(prediction))

                # To send feedback for the subject at the presentation computer use
                #   the subjInterface as shown below. For this demo there is no
                #   presentation computer so we won't actually it.
                # subjInterface.setResult(runNum, int(this_TR), float(prediction))
            else:
                print(f'Skipping classification because it is a rest trial')


    X_test = preprocessed_data[num_trainingData+1:]
    y_test = shifted_labels[num_trainingData+1:num_total_TRs].reshape(-1,)
    # we don't want to include the rest data
    X_test_noRest = X_test[y_test != 0]
    y_test_noRest = y_test[y_test != 0]
    # and we want to zscore the bold data
    X_test_noRest_zscored = scaler.transform(X_test_noRest)
    accuracy_score = clf.score(X_test_noRest_zscored, y_test_noRest)
    print('Accuracy of classifier on new data: %s' %accuracy_score)
    # print(y_test_noRest)
    # print(clf.predict(X_test_noRest_zscored))
  
    print(""
    "###################################################################################\n"
    "REAL-TIME EXPERIMENT COMPLETE!")

    return

def main(argv=None):
    """
    This is the main function that is called when you run 'sample.py'.

    Here, you will load the configuration settings specified in the toml configuration
    file. It will initiate the class clientInterface which automatically connects
    to the projectServer and allows making requests, such as to get DICOM or NifTi
    data using the dataInterface contained in clientInterface.
    It will then call the function 'doRuns' to actually start doing the experiment.
    """

    # define the parameters that will be recognized later on to set up fileIterface
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=None, type=str,
                           help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default='', type=str,
                           help='Comma separated list of run numbers')
    argParser.add_argument('--scans', '-s', default='', type=str,
                           help='Comma separated list of scan number')
    args = argParser.parse_args(argv)

    # load the experiment configuration file
    cfg = loadConfigFile(args.config)

    # obtain paths for important directories (e.g. location of dicom files)
    if cfg.imgDir is None:
        cfg.imgDir = os.path.join(currPath, 'dicomDir')
    cfg.codeDir = currPath

    # Make an RPC connection to the projectServer
    # The 'dataInterface' class allows you to read and write files from the
    #   control room computer to this script running in the cloud.
    # The 'subjInterface' class will allow us to send classification results
    #   as feedback to the subject at the presentation computer.
    clientRPC = ClientInterface()
    dataInterface = clientRPC.dataInterface
    subjInterface = clientRPC.subjInterface
    webInterface = clientRPC.webInterface
    
    # now that we have the necessary variables, call the function 'doRuns' in order
    #   to actually start reading dicoms and doing your analyses of interest!
    #   INPUT:
    #       [1] cfg (configuration file with important variables)
    #       [2] dataInterface (this will allow a script from the cloud to access files
    #               from the stimulus computer that receives dicoms from the Siemens
    #               console computer)
    #       [3] subjInterface - this allows sending feedback (e.g. classification results)
    #            to a subjectService running on the presentation computer to provide
    #            feedback to the subject (and optionally get their response).
    #       [4] webInterface - this allows updating information on the experimenter webpage.
    #            For example to plot data points, or update status messages.
    doRuns(cfg, dataInterface, subjInterface, webInterface)

    return 0


if __name__ == "__main__":
    """
    If 'sample.py' is invoked as a program, then actually go through all of the portions
    of this script. This statement is not satisfied if functions are called from another
    script using "from sample.py import FUNCTION"
    """
    main()
    sys.exit(0)
