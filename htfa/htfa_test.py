from mpi4py import MPI
import numpy as np
from htfa import HtfaArgs,get_data,read_data,fit_htfa
import nibabel
from nilearn.input_data import NiftiMasker
import scipy.io
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nvoxel=1000
ntr=200
dim=3

cov_vec_size = np.sum(np.arange(dim)+1)

K=5
max_outer_iter=10
max_inner_iter=10
threshold=0.01
nlss_method = 'dogbox' # ['trf','dogbox'] for bounded optimization
nlss_loss = 'soft_l1' #['linear','soft_l1','huber','cauchy','arctan']
weight_method = 'rr' #['ols','rr']
upper_ratio = 1.8
lower_ratio = 0.1
voxel_ratio = 0.25
tr_ratio = 0.1
max_voxel = 5000
max_tr = 500
from_array = False
from_mat = True
nsubjs = 4

if from_array:
    data = []
    R = []
    R0=np.random.rand(nvoxel,dim)
    for i in np.arange(nsubjs): 
        tmp = np.random.rand(nvoxel,ntr)        
        data.append(tmp)       
        R.append(R0)
    my_data,my_R =  get_data(data,R,nsubjs,rank,size)
elif from_mat:
#can also read from *.npy file
    prefix='/home/hadoop/TFA/tests/synth/'
    my_data = []
    my_R = []
    R0 = scipy.io.loadmat(prefix + 'R.mat')
    for idx in np.arange(nsubjs):
        if idx % size == rank:            
            data = scipy.io.loadmat(prefix + 's' +str(idx) + '.mat')
            my_data.append(data['data'])
            my_R.append(R0['R']) 
                          
else:
    prefix = '/home/hadoop/TFA/tests/115pieman_labels/'
    my_data = []
    my_R = []
    R0=np.load(prefix+'R.npy')
    for idx in np.arange(nsubjs):
        if idx % size == rank:
            my_data.append(np.load(prefix+'group1_s'+str(i)+'.npy'))
            my_R.append(R0)
    

args=HtfaArgs(K, nsubjs,max_outer_iter,max_inner_iter,threshold,
              nlss_method,nlss_loss,weight_method,upper_ratio,lower_ratio,
              voxel_ratio,tr_ratio,max_voxel,max_tr)

if rank == 0:
   start_time = time.time() 
                
fit_htfa(my_data,my_R,args)

if rank == 0:
   print("htfa exe time: %s seconds" % (time.time() - start_time))

"""
from tfa import get_factors
from sklearn.metrics import mean_squared_error

prefix='/home/hadoop/TFA/tests/synth/'
R0 = scipy.io.loadmat(prefix + 'R.mat')
R=R0['R']
template=scipy.io.loadmat(prefix + 'template.mat')
centers=template['template_centers']
widths=template['template_widths']
F0=scipy.io.loadmat(prefix + 'F.mat')
F=F0['F']

factors=get_factors(R,centers,widths)
diff=mean_squared_error(F, factors, multioutput='uniform_average')
print F.shape,factors.shape
print F[0:10]
print factors[0:10]
print diff,np.max(F),np.max(factors)
"""
