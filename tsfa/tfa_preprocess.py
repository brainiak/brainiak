import numpy as np
import sys
import os
import nilearn
import nibabel
from nilearn.input_data import NiftiMasker
import scipy.io
import time
import re
import glob

def getAllNifti(path):
    files = []
    for f in glob.glob(path+'*.nii'):
       files.append(f)         
    return files
    
def genIntersectionMask(path,files,second_mask=None,threshold=0):
  #start with the second mask
  new_mask = nibabel.load(second_mask)
  affine = new_mask.get_affine()
  final_mask_data = new_mask.get_data().astype(int)  
  final_mask_data[final_mask_data < threshold] = 0  
  img_data_type = new_mask.header.get_data_dtype()
  
  #then get intersection  
  for mask in files:
    new_mask = nibabel.load(mask)
    mask_data = new_mask.get_data().astype(bool)
    final_mask_data = np.logical_and(final_mask_data, mask_data[:,:,:,0])
    print(mask)
    
  print np.sum(final_mask_data)
  interection_mask = nibabel.Nifti1Image(final_mask_data.astype(img_data_type), affine)
  mask_file = path + 'intersection.img'
  nibabel.save(interection_mask, mask_file)
  return mask_file,interection_mask         

def preprocessRun(in_path,out_path,nifti_files,mask_file,groups):
  nifti_masker = NiftiMasker(mask_img=mask_file, standardize=True, detrend=False, smoothing_fwhm=False)
  cur_mask = nibabel.load(mask_file)
  mask_data = cur_mask.get_data().astype(bool)
  R = np.argwhere(mask_data)
  np.save(out_path + 'R.npy', R)  
  fshort=[nifti.replace(in_path,'')  for nifti in nifti_files]
  group_files = []
  for idxg,group in enumerate(groups):
     group_files.append([])
     name_len = len(group)
     for idxf, nifti in enumerate(fshort):        
        print group, nifti
        if nifti[0:name_len]==group:
           group_files[idxg].append(nifti_files[idxf])
       
     for idxf, nifti in enumerate(group_files[idxg]):        
        all_images = nifti_masker.fit_transform(nifti)  
        out_name = out_path +'group'+str(idxg)+'_s'+str(idxf)+'.npy'
        print nifti
        print all_images.shape
        np.save(out_name, all_images.T)



in_path='/home/hadoop/TFA/tests/115pieman/'
out_path='/home/hadoop/TFA/tests/115pieman_labels/'
second_mask='/home/hadoop/TFA/tests/avg152T1_gray_3mm.nii'
groups=['Intact','Paragraph','Resting','Word']
threshold=100
nifti_files=getAllNifti(in_path)
mask_file,_=genIntersectionMask(in_path,nifti_files,second_mask,threshold)
#mask_file=in_path + 'intersection.img'
preprocessRun(in_path,out_path,nifti_files,mask_file,groups)

