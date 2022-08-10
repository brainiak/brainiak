# Set Up Instructions for the Real-Time fMRI Cloud-Based Framework

Here are instructions you have to follow in order to implement our real-time fMRI cloud-based framework when running the [rtcloud_notebook jupyter notebook](https://github.com/brainiak/brainiak-aperture/blob/master/notebooks/real-time/rtcloud_notebook.ipynb). There are some things that you have to set up only once, whereas there are other things you have to do every time before you launch and run the jupyter notebook.

## Things to do once
Before you can run this notebook, you will have to take the following steps to set up our software framework:

1. Clone the [brainiak aperture repo](https://github.com/brainiak/brainiak-aperture.git) and the [rtcloud framework repo](https://github.com/brainiak/rt-cloud.git). The location of the repositories do not matter but you should make a note of the paths.
2. Follow [Step 1](https://github.com/brainiak/rt-cloud#step-1-install-mini-conda-and-nodejs) of the installation instructions for the rtcloud framework: Check to see if you have conda, Node.js, and NPM installed. Install these packages if necessary.
3. Create a conda environment that will be specific for the rtcloud framework and activate it:
    + `cd /PATH_TO_RTCLOUD/rt-cloud`
    + `conda env create -f environment.yml`
    + `conda activate rtcloud`
4. Install and build node module dependencies.
    + `cd /PATH_TO_RTCLOUD/rt-cloud/web`
    + `npm install`
    + `npm run build`

## Things to do every time
Here are the things that you have to do every time before you launch and run this jupyter notebook:

1. Activate the conda environment for the rtcloud framework:
    + `conda activate rtcloud`
2. On the command line, create a global variable to the full path for the rtcloud framework repo. You must do this in order to use the functions we have created for the framework. And don't forget the forward slash "/" at the end.
    + `export RTCLOUD_PATH=/PATH_TO_RTCLOUD/rt-cloud/`
    + Double check that you did this correctly by typing the following command, which should print the *full* path to the rtcloud framework folder: `ECHO $RTCLOUD_PATH`

## Common Issues

- If you get a blank blue screen when you open the localhost with the web server, then you forgot to follow Step 4 above.
- The `/tmp/notebook-simdata` folder is in your root directory. To get there, do `cd /tmp`. You want to delete this `notebook-simdata` folder whenever you want to similate the synthetic data being produced in real-time.
- If `rtCommon` can't be found, then you forgot to run Step 2 of "Things to do every time" above.
