import numpy as np
import sys
import os
import nilearn
import nibabel
from nilearn.input_data import NiftiMasker
import scipy.io
import importlib
import random
import time
from threading import Thread

output_file = None
trialNum = None
runNum = None

from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors
from sklearn import metrics
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV


def trainAndScore(trainData, trainLabels, testData, testLabels, method,featureSelection):

  sample_weight = np.ones(trainLabels.shape)
  for i in range(0,trainLabels.shape[0]):
    sample_weight[i] = 1.0 / np.sum(trainLabels[i] == trainLabels)
  sample_weight = 1.0 * trainLabels.shape[0] * sample_weight / np.linalg.norm(sample_weight)
 
  feature_selection = SelectKBest(f_classif, k=featureSelection[0])
  fs = featureSelection
  #number of voxels
  fs.append(trainData.shape[1])

  # Train classifier and get accuracy
  if method == 'L1LR':
    classifier = linear_model.LogisticRegression(penalty='l1', C=1000.0)
    classifier.fit(trainData, trainLabels )
  elif method == 'L2LR':
    #classifier = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    #classifier.fit(trainData.T, trainLabels )
    lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    classifier = Pipeline([('anova', feature_selection), ('LogisticRegression', lr)])
    classifier.fit(trainData, trainLabels )
  elif method == 'RBFSVM':
    svc = svm.SVC(kernel='rbf')
   #classifier = Pipeline([('anova', feature_selection), ('svc', svc)])
    classifier = svc
    classifier.fit(trainData, trainLabels )
    #classifier.fit(trainData, trainLabels, sample_weight=sample_weight )
  elif method == 'LinearSVM':
    svc = svm.SVC(kernel='linear', probability=True)
    k_grid = dict(anova__k=fs)
   # classifier = svc
   # classifier = Pipeline([('anova', feature_selection), ('svc', svc)])
    classifier = GridSearchCV(Pipeline([('anova', feature_selection), ('svc', svc)]), param_grid=k_grid,cv=5)
    classifier.fit(trainData, trainLabels )
    #print(classifier.best_params_)
  elif method == 'GaussianNB':
    classifier = naive_bayes.GaussianNB()
    classifier.fit(trainData, trainLabels )
  elif method == 'DecisionTree':
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(trainData, trainLabels )
  elif method == 'KNN':
    classifier = neighbors.KNeighborsClassifier(n_neighbors=15)
    classifier.fit(trainData, trainLabels )
  elif method == 'RandomForest':
    classifier = ensemble.RandomForestClassifier()
    classifier.fit(trainData, trainLabels )
  elif method == 'Dummy':
    from sklearn.dummy import DummyClassifier
    classifier = DummyClassifier()
    classifier.fit(trainData, trainLabels)

  if trainAll:
      #test on different subjects      
      allTRs = testData.shape[1]
      nsubj = allTRs/nstate
      result = []
      for subj in np.arange(nsubj):
          y_pred = classifier.predict(testData[:,subj*nstate:(subj+1)*nstate].T)
          result.append(confusion_matrix(testLabels, y_pred))
      return result
  else:
      y_pred = classifier.predict(testData)
  return confusion_matrix(testLabels, y_pred)


def runSrm(path, srm_subjs, expected_run_num, niter, nstate, features, mask, getIntersection, getAveraging, perRunZscore,memOnly):  
  nSubj = len(srm_subjs)    
  runs = np.arange(expected_run_num)+1
  
  if useRSA:
     prefix = path + mask + '_rsa'
  else:
     prefix = path + mask     
     
  #in case stimulus is the same for different subjects
  if not getAveraging and not getCommon:
      if getIntersection:
         input_path = prefix + '_inter_labels/'
      else:
         input_path = prefix + '_labels/'
  
  else:  
      #in case preprocessing was needed to get same stimulus across subjects  
      if getIntersection:
         input_path = prefix + '_inter_sort'
      else:
         input_path = prefix + '_sort'
         
      if getAveraging:
         input_path = input_path + '_avg_labels/'    
             
      if getCommon:
         input_path = input_path + '_common_labels/'

     
  data = np.load(input_path + str(srm_subjs[0]) + '_1_images.npy')
  (nVoxel, nTR) = data.shape
  #print data.shape
  if getAveraging:
     nTR = nstate
   
  R_prefix = path + mask  
  if getIntersection:  
    #same number of voxels across subjects    
  #  fhss = np.zeros((nVoxel, expected_run_num*nTR, nSubj))      
    R=[]
    for idx, subject in enumerate(srm_subjs): 
      R.append(np.load(R_prefix + '_inter_labels/' + str(subject) + '_1_R.npy'))
      data = np.load(input_path + str(subject) + '_1_images.npy')
      if idx == 0:    
          runStartIdx = np.zeros((expected_run_num))
          runEndIdx = np.zeros((expected_run_num))       
          trPerRun = np.zeros((expected_run_num))
          runEndIdx[0] = data.shape[1]    
          trPerRun[0] =  data.shape[1]
      for r_idx, run in enumerate(runs):
          tmp = np.load(input_path + str(subject) + '_' + str(run) + '_images.npy')
          data = np.hstack((data, tmp))         
          if idx == 0:
              runStartIdx[r_idx] = runEndIdx[r_idx-1]  
              trPerRun[r_idx] =  tmp.shape[1]        
              runEndIdx[r_idx] = runStartIdx[r_idx] + tmp.shape[1]
      
      #create fhss after knowing the #TR per subject
      if idx == 0:
          fhss = np.zeros((nVoxel, runEndIdx[-1], nSubj))
      
      fhss[:,:,idx] = data
    
  else:
    # varying number of voxels across subjects
    # the number of TRs per fun might also be different
    fhss = []
    R = []    
    for idx, subject in enumerate(srm_subjs):  
      R.append(np.load(R_prefix + '_labels/' + str(subject) + '_1_R.npy'))
      data = np.load(input_path + str(subject) + '_1_images.npy') 
      # trPerRun, runStartIdx,runEndIdx is the same across subjects
      if idx == 0:    
          runStartIdx = np.zeros((expected_run_num))
          runEndIdx = np.zeros((expected_run_num))       
          trPerRun = np.zeros((expected_run_num))
          runEndIdx[0] = data.shape[1]    
          trPerRun[0] =  data.shape[1]       
      for r_idx, run in enumerate(runs[1:],1):          
          tmp = np.load(input_path + str(subject) + '_' + str(run) + '_images.npy')
          data = np.hstack((data, tmp))
          if idx == 0:
              runStartIdx[r_idx] = runEndIdx[r_idx-1]  
              trPerRun[r_idx] =  tmp.shape[1]        
              runEndIdx[r_idx] = runStartIdx[r_idx] + tmp.shape[1]
         
      fhss.append(data)    
  
  
  if srmTrain5:
      from srm_fhss_train5 import srmFhss
  else:
      from srm_fhss import srmFhss  
  #save data for SRM use    
  output_path = path + mask+'_srm_input'        
  if not os.path.exists(output_path):
      os.makedirs(output_path)
  scipy.io.savemat(output_path + '/fhss.mat', {'fhss': fhss})
  class Args(object):
    #align_algo = 'srm_full'
    def __init__(self,alignAlgo,features, miter,niter,path,expected_run_num,srm_subjs,trPerRun,
                 runEndIdx,runStartIdx,getAveraging,getIntersection,perRunZscore,memOnly,
                 tfaThreshold,tfaK,tfaEmbedded,tfaMethod,tfaLoss,tfaWeight,LW,UW):            
        self.align_algo = alignAlgo
        self.features = features
       # nfeature = nfeature
        self.niter = niter
        self.randseed = random.randint(1, 10)
        self.strfresh = True
        self.path = path
        self.nruns = expected_run_num
        self.nTrain = expected_run_num - 1
        self.subjects = srm_subjs           
        self.trPerRun = trPerRun
        self.runEndIdx = runEndIdx
        self.runStartIdx = runStartIdx
        self.getAveraging = getAveraging
        self.getIntersection = getIntersection
        self.perRunZscore = perRunZscore
        self.memOnly = memOnly
        self.miter = miter
        self.threshold = tfaThreshold
        self.tfaK = tfaK
        self.tfaEmbedded = tfaEmbedded
        self.tfaMethod = tfaMethod
        self.tfaLoss = tfaLoss
        self.tfaWeight = tfaWeight
        self.LW=LW
        self.UW=UW
  
  args = Args(alignAlgo,features, miter,niter,path,expected_run_num,srm_subjs,trPerRun,
              runEndIdx,runStartIdx,getAveraging,getIntersection,perRunZscore,memOnly,
              tfaThreshold,tfaK,tfaEmbedded,tfaMethod,tfaLoss,tfaWeight,LW,UW)
    
  if getIntersection:
     output_path = prefix + '_inter_sort'
  else:
     output_path = prefix + '_sort'
     
  if getAveraging:
      output_path = output_path + '_avg'
  else: 
      output_path = output_path + '_common'

  options = {'input_path'  : path + mask+'_srm_input/',\
             'working_path': path +'srm_working/',\
             'output_path' : output_path + '_srm_labels/'}
  
  
  if memOnly:
     #result is nrun*nsubj list
     #each list element is nVoxel*(trPerRun*nrun) transformed data
     srmResult,bestK = srmFhss(fhss, R,options,args)
     return srmResult,bestK,runStartIdx,runEndIdx,trPerRun
  else:
     bestK = srmFhss(fhss, R, options,args)
     return bestK
     
     
     
def classifySubjectMemAll(path, subjects, method, idxf, runs, states, mask, getIntersection, getAveraging,useRSA,srmResult,featureSelection):
  nruns = len(runs)
  nsubj = len(subjects)
  
  if srmTrain5:
       nVoxel = srmResult[0].shape[0] 
  else:
       nVoxel = srmResult[0][0][0].shape[0]  
  
  trPerRun = nstate
  trainTrPerSubj =  trPerRun*(nruns-1)
  # Cross validation   
  avg_accuracy = np.zeros((nsubj))
  conf_mat = []
  for idxT, testIdx in enumerate(runs):
    condition=runs!=testIdx
    trainIdx = np.extract(condition,runs)
    
    #get testData, testLabels
    testData = np.zeros((nVoxel, trPerRun*nsubj))
    for subj in np.arange(nsubj):
        if srmTrain5:
            testData[:,subj*trPerRun: (subj+1)*trPerRun] = srmResult[subj][:,(testIdx-1)*trPerRun:testIdx*trPerRun].T.copy()
        else:
            testData[:,subj*trPerRun: (subj+1)*trPerRun] = srmResult[idxT][idxf][subj][:,(testIdx-1)*trPerRun:testIdx*trPerRun].T.copy()
            
    testLabels = states
          
    trainData = np.zeros((nVoxel, trainTrPerSubj*nsubj))
    trainLabels = np.zeros((trainTrPerSubj*nsubj))
    
    #get trainData 
    for subj in np.arange(nsubj):
      base = subj*trainTrPerSubj
      for idx1, run in enumerate(trainIdx):
          if srmTrain5:
              trainData[:,base+idx1*trPerRun:base+(idx1+1)*trPerRun] = srmResult[subj][:,(run-1)*trPerRun:run*trPerRun].T.copy()     
          else:
              trainData[:,base+idx1*trPerRun:base+(idx1+1)*trPerRun] = srmResult[idxT][idxf][subj][:,(run-1)*trPerRun:run*trPerRun]              
    
    trainLabels = np.repeat(states,(nruns-1)*nsubj,axis=0)            
    conf_mat.append(trainAndScore(trainData, trainLabels, testData, testLabels, method,featureSelection))
    
  for idx, subject in enumerate(subjects): 
        for test in np.arange(nruns):
            avg_accuracy[idx] = avg_accuracy[idx] + 1.0 * np.sum(np.diag(conf_mat[test][idx])) / np.sum(np.sum(conf_mat[test][idx]))
        print str(method) + ' average CV accuracy precision recall subject ' + str(subject) + ' is '+ str(avg_accuracy[idx] / float(nruns))


def classifySubjectMem(path, subject, subjIdx,method,idxfs, runs, states, mask, getIntersection, getAveraging,useRSA,srmResult,runStartIdx,runEndIdx,trPerRun,featureSelection):
  nruns = len(runs)
  if useRSA:
     prefix = path + mask + '_rsa'
  else:
     prefix = path + mask
   
     
  if getIntersection:    
     diff = '_inter' 
  else: 
     diff = ''
      
  if getAveraging:
     diff = diff + '_sort_avg'
  elif getCommon: 
     diff = diff + '_sort_common'
  else:
     diff = ''
 
  if not getAveraging:
      label_prefix = prefix + diff + '_labels/' + str(subject)    
  
   
  # Cross validation   
  avg_accuracy = 0.0
  for idxT, testIdx in enumerate(runs):
    condition=runs!=testIdx
    trainIdx = np.extract(condition,runs)
    
    #get testData, testLabels
    if srmTrain5:
       allData = srmResult[subjIdx] 
    else:
       allData = srmResult[idxT][idxf][subjIdx]
    (nVoxel, nTR) = allData.shape   
    
       
    testData = allData[:,runStartIdx[testIdx-1]:runEndIdx[testIdx-1]].T.copy()
    if getAveraging:
       testLabels = states
    else:
       testLabels = np.load(label_prefix + '_' + str(testIdx) + '_labels.npy')
          
    
    #get trainData 
    
    run = trainIdx[0]
    startIdx = 0
    endIdx = trPerRun[run-1]
    trainData = allData[:,runStartIdx[run-1]:runEndIdx[run-1]]    
    if getAveraging:
        trainLabels = states
    else:
        trainLabels = np.load(label_prefix + '_' + str(run) + '_labels.npy')
    
          
    for run in trainIdx[1:]:
      startIdx = endIdx      
      endIdx = startIdx + trPerRun[run-1]
      trainData = np.hstack((trainData,allData[:,runStartIdx[run-1]:runEndIdx[run-1]]))
      
      if getAveraging:
          trainLabels = np.hstack((trainLabels, states))
      else:
          trainLabels = np.hstack((trainLabels, np.load(label_prefix + '_' + str(run) + '_labels.npy')))
    
    trainData = trainData.T.copy()
    
    conf_mat = trainAndScore(trainData, trainLabels, testData, testLabels, method,featureSelection)
    avg_accuracy = avg_accuracy + 1.0 * np.sum(np.diag(conf_mat)) / np.sum(np.sum(conf_mat))

  print str(method) + ' average CV accuracy precision recall subject ' + str(subject) + ' is '+ str(avg_accuracy / float(nruns))
  return avg_accuracy / (float(nruns))


def classifySubject(path, subject, subjIdx,method, runs, states, nfeature,mask, getIntersection, getAveraging,useRSA,featureSelection):
  nruns = len(runs)
  trainSetSize = len(runs) - 1
  indices = range(0, len(runs))
  allIndices = np.concatenate([indices,indices])
 
  #find data path prefix based on configurations
  if useRSA:
     prefix = path + mask + '_rsa'
  else:
     prefix = path + mask
  
  if getIntersection:    
     diff = '_inter' 
  else: 
     diff = ''
      
  if getAveraging:
     diff = diff + '_sort_avg'
  elif getCommon: 
     diff = diff + '_sort_common'
  else:
     diff = ''
 
  if not getAveraging:
      label_prefix = prefix + diff + '_labels/' + str(subject) 
  
  if runType == 0:
      data_prefix = prefix + diff + '_labels/' + str(subject)
  else:
      data_prefix = prefix + diff + '_srm_output/' + str(subject) + '_' + str(nfeature)

  # Cross validation
  avg_accuracy = 0.0
  for idx in indices:
    # get indices for training, testing 
    trainIdx = allIndices[idx:idx+trainSetSize]
    testIdx = map(int, list(set(indices)-set(trainIdx)))
   
    #load the data that SRM trained on the trainIdx
    data = []
    labels = []
    for run in runs:
      #train SRM with train runs
      idxT = testIdx[0]+1    
      if runType == 0: 
          tmp_data = np.load(data_prefix + '_' + str(run) + '_images.npy')
      else:
          tmp_data = np.load(data_prefix + '_' + str(run) + '_' + str(idxT) + '_images.npy')
           
           
      if getAveraging:
          tmp_labels = states
      else:
          tmp_labels = np.load(label_prefix + '_' + str(run) + '_labels.npy')
      
      data.append(tmp_data)
      labels.append(tmp_labels)    

    # Create test set
    testData = np.hstack(tuple([data[i] for i in testIdx] )).T.copy()
    testLabels = np.hstack(tuple([labels[i] for i in testIdx] ))

    # Create training set   
    trainData = np.hstack(tuple([data[i] for i in trainIdx] )).T.copy()
    trainLabels = np.hstack(tuple([labels[i] for i in trainIdx] ))
    
    conf_mat = trainAndScore(trainData, trainLabels, testData, testLabels, method,featureSelection)
    avg_accuracy = avg_accuracy + 1.0 * np.sum(np.diag(conf_mat)) / np.sum(np.sum(conf_mat))
    
  print str(method) + ' average CV accuracy precision recall subject '  + str(subject) + ' is '+ str(avg_accuracy / float(nruns))
  sys.stdout.flush()
  return avg_accuracy / (float(nruns))


start_time = time.time()
#subjects who have 5 runs
#subjects = np.array([206, 208, 209, 210, 211, 212, 213, 214, 216, 217, 219, 220, 221, 222, 223, 224, 225, 227, 228, 229, 230, 231, 232])
subjects = []
#each group of subjects have the same stimulus sequence
subjects.append(['206', '210', '214',  '222', '230'])
subjects.append(['208', '212', '216',  '220', '224', '228', '232'])
subjects.append(['209', '213', '217', '221', '225', '229'])
subjects.append(['211', '219', '223', '227', '231'])
nSubj = len(subjects)
nstate = 16
states = np.arange(nstate)
expected_run_num = 5
masks = ['wholebrain','OFCKahnt_K2_All','OFCKahnt_K2_1','OFCKahnt_K2_2','AudC','DLPFC','HC','MotC', 'VisC','wAudC','wMotC','wOFCKahnt_K2_1','wOFCKahnt_K2_2','wVisC']
mask = masks[1]

if sys.platform=='darwin':
  path='/Users/megan/Documents/projects/FHSS/'
else:
  path='/data1/state_space/FHSS/'

niter = 10
miter = 1
tfaThreshold = 0.01
tfaK = 50
tfaEmbedded = False
tfaWeight = 'ols'
alignAlgo = 'srm_full'
tfaMethod = 'dogbox'  # ['trf','dogbox'] for bounded optimization
tfaLoss = 'soft_l1' #['linear','soft_l1','huber','cauchy','arctan']
UW=1.0
LW=0.05
runs = np.arange(expected_run_num)+1
runType = 0
getIntersection = False
getAveraging = False
getCommon = False
useRSA = False #
perRunZscore = True
memOnly = True
trainAll = False
classifier = 'LinearSVM'
if mask == 'wholebrain':
     featureSelection = [500,750,1000, 1500, 2000, 5000]
else:
     featureSelection = [500,750,1000, 1200]
srmTrain5 = False
if useRSA and getIntersection:
  print "RSA data is in subject space, cannot getIntersection in this case!!!" 
  exit('wrong parameters')

if getAveraging:
   #16 pseudo TRs per run
   features = [58,59,60,61,62,63,64]
 #   features = [40,41,42,43,44,45,46,47,48,49] 
 #  features = [52,54,56,58,62]
elif getCommon:
   #89 common TRs per run, 356TRs for train
   features = [100,200,300,356]
else:
   #484 TRs for each subject
   features = [100,200,300,484]
  
#no SRM at all
if runType == 0: 
    for s in subjects:    
       for idx,subject in enumerate(s):
         classifySubject(path, subject, idx,classifier, runs, states, 0,mask, getIntersection, getAveraging,useRSA,featureSelection)
#SRM already ran, use its results directly
elif runType == 1:    
    for nfeature in features:
      print nfeature    
      for idx,subject in enumerate(subjects): 
          classifySubject(path, subject, idx,classifier, runs, states, nfeature, mask, getIntersection, getAveraging,useRSA,featureSelection)
          
     
#run SRM before classification      
else:    
    if memOnly:      
          for s in subjects: 
              srmResult,bestK,runStartIdx,runEndIdx,trPerRun = runSrm(path, s, expected_run_num, niter, nstate, features,mask,getIntersection, getAveraging,perRunZscore,True) 
              
              if trainAll and getIntersection and not useRSA:                             
                  for idxf, nfeature in enumerate(features):
                      print nfeature  
                      start_time1 = time.time() 
                      classifySubjectMemAll(path, s, classifier,idxf,runs, states, mask,getIntersection, getAveraging,useRSA,srmResult,featureSelection)      
                      print("nfeature = %d, total classification took %s seconds" % (nfeature,time.time() - start_time1))                 
              else:                
                  for idxf, nfeature in enumerate(features):
                      print nfeature    
                      for idx,subject in enumerate(s):  
                       start_time1 = time.time()                
                       classifySubjectMem(path, subject, idx,classifier,idxf, runs, states, mask,getIntersection, getAveraging,useRSA,srmResult,runStartIdx,runEndIdx,trPerRun,featureSelection)      
                      print("nfeature = %s, subj = %s, classification took %s seconds" % (str(nfeature), str(subject),time.time() - start_time1))      
 
    else:
        bestK = runSrm(path, subjects, expected_run_num, niter, nstate, features,mask,getIntersection, getAveraging,perRunZscore,False)    
        for nfeature in features:
          print nfeature      
          for idx,subject in enumerate(subjects):  
            start_time1 = time.time()
            classifySubject(path, subject, idx,classifier, runs, states, bestK[idx],mask,getIntersection, getAveraging, useRSA,featureSelection)      
            print("subj = %d, classification took %s seconds" % (subject,time.time() - start_time1))
            
print("--- %s seconds ---" % (time.time() - start_time)) 
