import numpy as np
import sys
import os
import nilearn
import nibabel
from nilearn.input_data import NiftiMasker
import scipy.io
import time

output_file = None
trialNum = None
runNum = None

def strArrayFromFile(fname):
  fid = open(fname)
  arr = [];
  for line in fid:
    runName = line.rstrip()
    arr.append(runName)
  fid.close()
  return arr

def cleanName(name):
   return name.replace('_R_','').replace('_E_','').replace('_0_','')

def preprocessRun(path, subject, run, mask, getIntersection, dataType, getAveraging, useRSA):
  # Load labels and onsets
  label_mat = scipy.io.loadmat(path + 'labels/Trialwise_' + str(subject) + '_' + str(run) + '.mat')
  onsets = np.array([item[0][0] for item in label_mat['onsets'][0]])
  durations = np.array([item[0][0] for item in label_mat['durations'][0]])
  names = np.array([item[0] for item in label_mat['names'][0]])  
  trs = np.floor((onsets / 2.4)  + 1.0 + 2.0).astype(int) 
  def getNum(instr):
    res = re.search('\d+', instr)
    if res:
      return int(res.group(0))
    else:
      return 0
  ids = np.array([getNum(name) for name in names])
  iserror = np.array([(name[4:6] == '_E') for name in names])
  isrerun = np.array([(name[4:6] == '_R') for name in names])
  
  
  if useRSA:
      fmri_prefix = path + 'for_SRM/' + subject
      fmri_path = fmri_prefix + '/betas_run'  + str(run) + '.nii'
      mask_path = fmri_prefix + '/brain_mask.nii' 
  else:  
      #load data for raw mask  
      fmri_prefix = path + 'preprocessed_data/' + subject + '/task_run' + str(run) + '_realigned'
      if getIntersection:      
          mask_path = path + 'anatomical_masks_individual/intersection_' + mask + '.img'      
          fmri_path = fmri_prefix + '_normalized.nii' 
      else:
         mask_prefix = path + 'anatomical_masks_individual/' + mask + '_' + subject
         if mask == 'wholebrain':
             #for wholebrain, *.img
             mask_path = mask_prefix + '.img'
         else:
             #for small ROIs, *.nii
             mask_path = mask_prefix + '.nii'
         fmri_path = fmri_prefix + '.nii' 
  
  nifti_masker = NiftiMasker(mask_img=mask_path, standardize=True, detrend=False, smoothing_fwhm=4.0)  
  all_images = nifti_masker.fit_transform(fmri_path)  
  #find valid voxels' coordinate from mask file. verified the order of voxels matches that of all_images
  cur_mask = nibabel.load(mask_path)
  mask_data = cur_mask.get_data().astype(bool)
  R = np.argwhere(mask_data)
  
  #RSA data is already in pseudo TRs, and cannot getIntersection. save and exit
  if useRSA:
    output_path = path + mask + '_rsa_sort_avg_labels/'
    if not os.path.exists(output_path):
      os.makedir(output_path)
    np.save(output_path + str(subject) + '_' + str(run) + '_images.npy', all_images.T)
    np.save(output_path + str(subject) + '_' + str(run) + '_R.npy', R)
    print(str(subject) + ' ' + str(run))
    return

  # Extract relevant data
  maxidx = trs.shape[0]-1
  trs = np.array([tr if tr <= all_images.shape[0] else (all_images.shape[0]) for tr in trs])  
  relevant = np.zeros(trs.shape).astype(bool)
  numeric_labels = -1 * np.ones(trs.shape).astype(int)
  #to cope with the bug in original names (subject230)
  for idx, name in enumerate(names):
    if names[idx] ==  'FoFy2':
        names[idx] = 'FoFy  2'
    if names[idx] ==  'FyHy3':
        names[idx] = 'FyHy  3' 
    if names[idx] ==  'HyHy5':
        names[idx] = 'HyHy  5'
    if names[idx] ==  'HyHy6':
        names[idx] = 'HyHy  6'  
  
  last_idx = 0
  for idx, name in enumerate(names): 
    clean_name = cleanName(name)
    name_str = clean_name[0:4]
    name_idx = int(clean_name[4:7])
    tr_type = name[5]

    if dataType == 0: 
      def checkCondition(key,name_str,name,tr_type,name_idx,names,last_idx):
          return (name_str == key)
    elif dataType == 1:
       #keep correct one and last rerun 
       if idx > 0 and idx < maxidx:  
           def checkCondition(key,name_str,name,tr_type,name_idx,names,last_idx):    
               return (name_str == key) and (tr_type != 'E') and (names[idx+1][5] != 'R') and (name_idx == last_idx + 1 or last_idx == 0) 
       else:
           def checkCondition(key,name_str,name,tr_type,name_idx,names,last_idx):                            
               return (name_str == key) and (tr_type != 'E')          
    elif dataType == 2:      
       def checkCondition(key,name_str,name,tr_type,name_idx,names,last_idx):
           return (name_str == key) and (tr_type != 'E') 
    else:      
       def checkCondition(key,name_str,name,tr_type,name_idx,names,last_idx):
           return (name_str == key) and (tr_type != 'E')  and (tr_type != 'R')    
    
    for key in name_to_label:
        relevant[idx] = checkCondition(key,name_str,name,tr_type,name_idx,names,last_idx) 
        if relevant[idx]:
            numeric_labels[idx] = name_to_label[key]
            last_idx = name_idx
          #  print subject,run,idx,trs[idx],name,'True'
            break   
    
  print subject,run,all_images.shape[0],trs.shape[0],np.sum(relevant)
  names = np.array([name[0:4] for name in names])
  relevant_trs = trs[relevant]
  relevant_names = names[relevant]
  relevant_numeric_labels = numeric_labels[relevant]
  relevant_images = all_images[relevant_trs-1,:]

    
  # Save relevant data
  prefix = path + mask
  if getIntersection:
     output_path = prefix + '_inter_labels/'
  else:
     output_path = prefix + '_labels/'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  np.save(output_path + str(subject) + '_' + str(run) + '_names.npy', relevant_names)
  np.save(output_path + str(subject) + '_' + str(run) + '_labels.npy', relevant_numeric_labels)
  np.save(output_path + str(subject) + '_' + str(run) + '_images.npy', relevant_images.T)
  np.save(output_path + str(subject) + '_' + str(run) + '_errors.npy', relevant_errors)
  np.save(output_path + str(subject) + '_' + str(run) + '_ids.npy', relevant_ids)
  np.save(output_path + str(subject) + '_' + str(run) + '_trs.npy', relevant_trs)
  np.save(output_path + str(subject) + '_' + str(run) + '_R.npy', R)  
 
 
  # sort  
  if getAveraging or getCommon:
      order = relevant_numeric_labels.argsort()
      sorted_data = relevant_images[order].T
      sorted_labels = relevant_numeric_labels[order] 
      sorted_names = relevant_names[order]    
      (nVoxel, nTR) = sorted_data.shape 
        
      # Save sorted data
      if getIntersection:
         output_path = prefix + '_inter_sort_labels/'
      else:
         output_path = prefix + '_sort_labels/'
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      np.save(output_path + str(subject) + '_' + str(run) + '_names.npy',  sorted_names)
      np.save(output_path + str(subject) + '_' + str(run) + '_labels.npy', sorted_labels)
      np.save(output_path + str(subject) + '_' + str(run) + '_images.npy', sorted_data)
      
      stateCount =  np.bincount(sorted_labels)
      return stateCount
    

def getAvgTR(path, subject, run, mask,getIntersection):
  prefix = path + mask  
  if getIntersection:
     input_path = prefix + '_inter_sort_labels/'
  else:
     input_path = prefix + '_sort_labels/'

  sorted_labels = np.load(input_path + str(subject) + '_' + str(run) + '_labels.npy')
  sorted_data = np.load(input_path + str(subject) + '_' + str(run) + '_images.npy')    
  (nVoxel,origTR) = sorted_data.shape
  
  #average state withrun for srm    
  avg_data = np.zeros((nVoxel, nState))
  for idxS, state in enumerate(states):
      curState = sorted_labels == state
      avg_data[:,idxS] = np.mean(sorted_data[:,curState], axis=1)
   
  if getIntersection:
      output_path = prefix + '_inter_sort_avg_labels/'
  else:
      output_path = prefix + '_sort_avg_labels/'
  if not os.path.exists(output_path):
      os.makedirs(output_path)
  np.save(output_path + str(subject) + '_' + str(run) + '_images.npy', avg_data)
  print(str(subject) + ' ' + str(run))
  
def getCommonTR(path, subject, run, mask, nTR, getIntersection,totalTrPerState, minTrPerState):
  prefix = path + mask     
  if getIntersection:
     input_path = prefix + '_inter_sort_labels/'
  else:
     input_path = prefix + '_sort_labels/'

  sorted_labels = np.load(input_path + str(subject) + '_' + str(run) + '_labels.npy')
  sorted_data = np.load(input_path + str(subject) + '_' + str(run) + '_images.npy') 
   
  (nVoxel,origTR) = sorted_data.shape
  final_data = np.zeros((nVoxel, nTR))
  final_labels = np.zeros((nTR,))
  findalIdx = 0
  origIdx = 0
  for idx, count in enumerate(minTrPerState):
      final_data[:,findalIdx:findalIdx+count] = sorted_data[:,origIdx:origIdx+count]
      final_labels[findalIdx:findalIdx+count] = sorted_labels[origIdx:origIdx+count]
      findalIdx += counto
      origIdx += totalTrPerState[idx]  
  
    # Save sorted data
  if getIntersection:
     output_path = prefix + '_inter_sort_common_labels/'
  else:
     output_path = prefix + '_sort_common_labels/'
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  np.save(output_path + str(subject) + '_' + str(run) + '_labels.npy', final_labels)
  np.save(output_path + str(subject) + '_' + str(run) + '_images.npy', final_data) 
  print(str(subject) + ' ' + str(run))
  

def genIntersectionMask(path, subjects, mask):
  subject = subjects[0]
  
  if mask == 'wholebrain':
     #for wholebrain, *_MNI.img is the one in MNI space
     mask_prefix = path + 'anatomical_masks_individual/' + mask + '_' 
     raw_mask = mask_prefix + subject + '_MNI.img'
  else:
     #for small ROIs, *_norm.nii is the one in MNI space
     mask_prefix = path + 'anatomical_masks_individual/' + mask + '_' 
     raw_mask = mask_prefix + subject + '_norm.nii'
     
  new_mask = nibabel.load(raw_mask)
  affine = new_mask.get_affine()
  final_mask_data = new_mask.get_data().astype(bool)
  img_data_type = new_mask.header.get_data_dtype()  
  for idx,subject in enumerate(subjects[1:],1):
    if mask == 'wholebrain':
        mask_file = mask_prefix + subject + '_MNI.img'
    else:
        mask_file = mask_prefix + subject + '_norm.nii' 
    new_mask = nibabel.load(mask_file)
    mask_data = new_mask.get_data().astype(bool)
    final_mask_data = np.logical_and(final_mask_data, mask_data)
    print(subject)

  interection_mask = nibabel.Nifti1Image(final_mask_data.astype(img_data_type), affine)
  mask_file = path + 'anatomical_masks_individual/intersection_' + mask + '.img'
  nibabel.save(interection_mask, mask_file)
  return interection_mask


start_time = time.time()
getIntersection = True
getAveraging = True
getCommon = False
getBaseProcessing = True
useRSA = False
dataType = 3# 0:all data; 1:keep correct and last rerun; 2:exclude error run; 3:exclude both error and rerun;
name_to_label = {}

if useRSA:
  name_to_label['FoFy'] = 0
  name_to_label['FyFo'] = 1
  name_to_label['HoHy'] = 2
  name_to_label['HyHo'] = 3
  name_to_label['FyFy'] = 4
  name_to_label['FoFo'] = 5
  name_to_label['HyHy'] = 6
  name_to_label['HoHo'] = 7
  name_to_label['HyFy'] = 8
  name_to_label['HoFy'] = 9
  name_to_label['HyFo'] = 10
  name_to_label['HoFo'] = 11
  name_to_label['FyHy'] = 12
  name_to_label['FoHy'] = 13
  name_to_label['FyHo'] = 14
  name_to_label['FoHo'] = 15
else:
  name_to_label['FoFo'] = 0
  name_to_label['FoFy'] = 1
  name_to_label['FoHo'] = 2
  name_to_label['FoHy'] = 3
  name_to_label['FyFo'] = 4
  name_to_label['FyFy'] = 5
  name_to_label['FyHo'] = 6
  name_to_label['FyHy'] = 7
  name_to_label['HoFo'] = 8
  name_to_label['HoFy'] = 9
  name_to_label['HoHo'] = 10
  name_to_label['HoHy'] = 11
  name_to_label['HyFo'] = 12
  name_to_label['HyFy'] = 13
  name_to_label['HyHo'] = 14
  name_to_label['HyHy'] = 15
  
nState = len(name_to_label)
states = np.arange(nState)
#subjects = ['205', '206', '207', '208', '209', '210', '211', '212', '213', '214',  '216', '217', '219', '220', '221', '222', '223', '224', '225', '227', '228', '229', '230', '231', '232']
subjects = ['206', '208', '209', '210', '211', '212', '213', '214',  '216', '217', '219', '220', '221', '222', '223', '224', '225', '227', '228', '229', '230', '231', '232']

nSubj = len(subjects)
nRun = 5
runs = np.arange(nRun) +1
maxTR = 1000

if sys.platform=='darwin':
  path='/Users/megan/Documents/projects/FHSS/'
else:
  path='/data1/state_space/FHSS/'
 
masks = ['wholebrain','OFCKahnt_K2_All','OFCKahnt_K2_1','OFCKahnt_K2_2','AudC','DLPFC','HC','MotC', 'VisC','wAudC','wMotC','wOFCKahnt_K2_1','wOFCKahnt_K2_2','wVisC']

masks = masks[1]
totalTrPerState = np.zeros((nSubj, nRun, nState))
minTrPerState = maxTR * np.ones(nState)

if useRSA:
  if getIntersection or not getAveraging:
    print "RSA data is already in pseduo TRs status! and it is in subject space, cannot getIntersection in this case!!!" 
    exit  
  
if getIntersection:
  interection_mask = genIntersectionMask(path, subjects, mask)

if getBaseProcessing:
    for idxS, subject in enumerate(subjects):
     trPerState = np.array((nRun,nState))
     for idxR, run in enumerate(runs):
         if getCommon:
           totalTrPerState[idxS,idxR,:] = preprocessRun(path, subject, run, mask, getIntersection, dataType,getAveraging, useRSA)                     
           minTrPerState = np.minimum(minTrPerState, totalTrPerState[idxS,idxR,:])   
           print minTrPerState 
         else:
            preprocessRun(path, subject, run, mask, getIntersection, dataType,getAveraging, useRSA)  
  
   #save result to avoid rerun next time   
    if getCommon:
      if getIntersection:
          output_path = path + mask + '_sort_inter_labels/'
      else:
          output_path = path + mask + '_sort_labels/'        
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      np.save(output_path + 'totalTrPerState.npy',  totalTrPerState)
      np.save(output_path + 'minTrPerState.npy',  minTrPerState)
  
  
if not useRSA:
    if getAveraging:
      for idxS, subject in enumerate(subjects):    
        for idxR, run in enumerate(runs):
          getAvgTR(path, subject, run, mask, getIntersection)   
                       
if getCommon:
    #load from previously run results
      if getIntersection:
         input_path = path + mask + '_sort_inter_labels/'
      else:
         input_path = path + mask + '_sort_labels/'
         
      totalTrPerState = np.load(input_path + 'totalTrPerState.npy')
      minTrPerState = np.load(input_path + 'minTrPerState.npy')
      print totalTrPerState.shape
      print minTrPerState.shape
          
      nTR = minTrPerState.sum() 
      for idxS, subject in enumerate(subjects):    
        for idxR, run in enumerate(runs):
          getCommonTR(path, subject, run, mask, nTR, getIntersection,totalTrPerState[idxS,idxR,:], minTrPerState)
  
        
print("--- %s seconds ---" % (time.time() - start_time))     
  
  

