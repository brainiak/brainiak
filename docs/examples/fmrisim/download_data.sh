#!/bin/bash

# Download the archive containing the example data if we don't have it already
wget -nc https://dataspace.princeton.edu/bitstream/88435/dsp01dn39x4181/2/Corr_MVPA_archive.tar.gz

# If the file doesn't exist, we need to extract it.
test ! -e Corr_MVPA && tar xzkvf Corr_MVPA_archive.tar.gz Corr_MVPA_Data_dataspace/Participant_01_rest_run01.nii && mv Corr_MVPA_Data_dataspace Corr_MVPA

