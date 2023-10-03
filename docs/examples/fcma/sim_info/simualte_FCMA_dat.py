#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:41:27 2020

@author: Peeta Li
@contact: peetal@uoregon.edu
@modulated from https://brainiak.org/tutorials/13-real-time/
"""

# Check pick ~400 brain voxels, from two regions of interest (two cubes). 
# Not choose voxels randomly due to visualization purpose. 
"""
# --------------------------------------------------
# Cherry pick brain voxels for the simulation data  
# --------------------------------------------------
# mPFC MNI cooridnates: (0,54,-6)
# precuneus MNI coordinates: (0,-60,33)

# given MNI coordination, return voxel coordination given a mask dimenison. 
def _MNI_to_voxel_coord(x,y,z,affine):
    import numpy as np
    mni = np.asmatrix([[x],[y],[z],[1]])
    # pusedoinversed affine matrix %*% mni is the voxel coordinate
    voxel = np.asarray(np.asmatrix(np.linalg.pinv(affine.affine)) * mni)
    # transpose the result matrix, choose the first index, round, retrun a list
    center_voxel = [int(c) for c in np.transpose(voxel[0:3])[0].tolist()]
    
    return center_voxel[0], center_voxel[1], center_voxel[2]

# given center voxel coordination, mask and radius, output a roi mask
def _generate_roi_cube(x,y,z,brain_mask,radius):
    # empty the mask
    empty_mask = np.zeros(brain_mask.shape)
    # 
    empty_mask[x-radius:x+radius, y-radius:y+radius, z-radius:z+radius] = 1 
    
    return empty_mask

# load in the mask, can be generated of different sizes. 
dat_dir = '/path/to/FCMA_demo/sim_info'
mask_nii = image.load_img(os.path.join(dat_dir, 'brain_mask.nii.gz'))
mask = mask_nii.get_fdata()

# roi_A (mPFC)
x1, y1, z1 = _MNI_to_voxel_coord(0,54,-6,mask_nii) 
roi_A_mask = _generate_roi_cube(x1,y1,z1,mask,3)
roi_A_mask_nii = image.new_img_like(mask_nii, roi_A_mask, mask_nii.affine)

# roi_B (precuneus)
x2, y2, z2 = _MNI_to_voxel_coord(0,-60,33,mask_nii) 
roi_B_mask = _generate_roi_cube(x2,y2,z2,mask,3)
roi_B_mask_nii = image.new_img_like(mask_nii, roi_B_mask, mask_nii.affine)

# full mask
full_mask = np.zeros(mask.shape)
full_mask[np.where(roi_A_mask == 1)] = 1
full_mask[np.where(roi_B_mask == 1)] = 1
full_mask_nii = image.new_img_like(mask_nii, full_mask, mask_nii.affine)

# write out all the nifts: 
roi_A_mask_nii.to_filename(os.path.join(dat_dir, 'ROI_A_mpfc.nii.gz'))
roi_B_mask_nii.to_filename(os.path.join(dat_dir, 'ROI_B_precuenus.nii.gz'))
full_mask_nii.to_filename(os.path.join(dat_dir, 'full_mask.nii.gz'))
"""

# simulate 10 subjects' fMRI data over 9 small features (ROI) for FCMA demo

import os, random
from nilearn import image, plotting
import numpy as np  
from brainiak.utils import fmrisim as sim 

# specify directory
dat_dir = '/path/to/FCMA_demo/sim_info'

# Specify the volume parameters
trDuration = 1  # seconds
numTRs = 400 # How many TRs will you generate?

# Set up stimulus event time course parameters
event_duration = 15  # How long is each event
isi = 5  # What is the time between each event

# Specify signal magnitude parameters
signal_change = 10 # How much change is there in intensity for the max of the patterns across participants
multivariate_pattern = 1  # Do you want the signal to be a z scored pattern across voxels (1) or a univariate increase (0)

print('Load template of average voxel value')
sub_template_nii = image.load_img(os.path.join(dat_dir, 'brain_template.nii.gz'))
sub_template = sub_template_nii.get_fdata()
dimensions = np.array(sub_template.shape[0:3])

print('Create binary mask and normalize the template range')
mask, template = sim.mask_brain(volume = sub_template, mask_self = True)
mask_cherry = image.load_img(os.path.join(dat_dir, 'full_mask.nii.gz')).get_fdata()

# Load the noise dictionary
print('Loading noise parameters')
with open(os.path.join(dat_dir, 'sub_noise_dict.txt'), 'r') as f:
    noise_dict = f.read()
noise_dict = eval(noise_dict)
noise_dict['matched'] = 0


# stimfunction across two conditions for each subject 
stimfunc_all = []
for sid in range(8): # sid = 0
    # Create the stimulus time course of the conditions
    total_time = int(numTRs * trDuration)
    events = int(total_time / (event_duration + isi))
    onsets_A = []
    onsets_B = []
    randoized_label = np.repeat([1,2],int(events/2)).tolist()
    random.shuffle(randoized_label)
    for event_counter, cond in enumerate(randoized_label):
        
        # Flip a coin for each epoch to determine whether it is A or B
        if cond == 1:
            onsets_A.append(event_counter * (event_duration + isi))
        elif cond == 2:
            onsets_B.append(event_counter * (event_duration + isi))
            
    temporal_res = 1 # How many timepoints per second of the stim function are to be generated?
    
    # Create a time course of events 
    stimfunc_A = sim.generate_stimfunction(onsets=onsets_A,
                                           event_durations=[event_duration],
                                           total_time=total_time,
                                           temporal_resolution=temporal_res,
                                          )
    
    stimfunc_B = sim.generate_stimfunction(onsets=onsets_B,
                                           event_durations=[event_duration],
                                           total_time=total_time,
                                           temporal_resolution=temporal_res,
                                           )
    # stimfunc per subject
    stimfunc_ppt = np.concatenate((stimfunc_A, stimfunc_B), axis = 1)
    
    stimfunc_all.append(stimfunc_ppt)
    
    print('Load ROIs')
    nii_A = image.load_img(os.path.join(dat_dir, 'ROI_A_mpfc.nii.gz'))
    nii_B = image.load_img(os.path.join(dat_dir, 'ROI_B_precuenus.nii.gz'))
    ROI_A = nii_A.get_fdata()
    ROI_B = nii_B.get_fdata()
    
    # How many voxels per ROI
    voxels_A = int(ROI_A.sum())
    voxels_B = int(ROI_B.sum())
    
    # Create a pattern of activity across the two voxels
    print('Creating signal pattern')
  
    pattern_A = np.random.rand(voxels_A).reshape((voxels_A, 1))
    pattern_B = np.random.rand(voxels_B).reshape((voxels_B, 1))
   
    # Multiply each pattern by each voxel time course
    # Noise was added to the design matrix, to make the correlation pattern noise, so FCMA could be challenging. 
    # use normal distributed noise instead of unifrom distributed to make the data noisier 
    weights_A = np.tile(stimfunc_A, voxels_A) * pattern_A.T + np.random.normal(0,1.5, size = np.tile(stimfunc_A, voxels_A).shape) 
    weights_B = np.tile(stimfunc_B, voxels_B) * pattern_B.T + np.random.normal(0,1.5, size = np.tile(stimfunc_B, voxels_B).shape) 
        
    # Convolve the onsets with the HRF
    # TR less than feature is not good, but b/c this is simulated data, can ignore this concer. 
    print('Creating signal time course')
    signal_func_A = sim.convolve_hrf(stimfunction=weights_A,
                                   tr_duration=trDuration,
                                   temporal_resolution=temporal_res,
                                   scale_function=1,
                                   )
    
    signal_func_B = sim.convolve_hrf(stimfunction=weights_B,
                                   tr_duration=trDuration,
                                   temporal_resolution=temporal_res,
                                   scale_function=1,
                                   )
    
    
    # Multiply the signal by the signal change 
    signal_func_A =  signal_func_A * signal_change #+ signal_func_B * signal_change
    signal_func_B =  signal_func_B * signal_change #+ signal_func_A * signal_change

    # Combine the signal time course with the signal volume
    print('Creating signal volumes')
    signal_A = sim.apply_signal(signal_func_A,
                                ROI_A)
    
    signal_B = sim.apply_signal(signal_func_B,
                                ROI_B)
    
    # Combine the two signal timecourses
    signal = signal_A + signal_B
    
    # spare the true noise. 
    #print('Generating noise')
    #noise = sim.generate_noise(dimensions=dimensions,
    #                           stimfunction_tr=np.zeros((numTRs, 1)),
    #                           tr_duration=int(trDuration),
    #                           template=template,
    #                           mask=mask_cherry,
    #                           noise_dict=noise_dict,
    #                           temporal_proportion = 0.5)
    
    
    
    brain = signal #+ noise
    brain_nii = image.new_img_like(template_nii, brain, template_nii.affine)
    out_path = os.path.join(dat_dir, '../simulated_data')
    if os.path.isdir(out_path) is False:
        os.makedirs(out_path, exist_ok=True)
    brain_nii.to_filename(os.path.join(out_path, f"sub_{sid}_sim_dat.nii.gz"))
    
     
# write out the simulated epoch file 
sim.export_epoch_file(stimfunction = stimfunc_all,
                      filename = os.path.join(dat_dir, '../simulated_data/sim_epoch_file.npy'),
                      tr_duration = 1.0,
                      temporal_resolution = 1.0,)

