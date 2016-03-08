from mpi4py import MPI
import numpy as np
from htfa import HtfaArgs,get_data,read_data,fit_htfa

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nvoxel=1000
ntr=200
dim=3

cov_vec_size = np.sum(np.arange(dim)+1)

K=5
max_outer_iter=1
max_inner_iter=1
threshold=0.01
nlss_method = 'dogbox'  # ['trf','dogbox'] for bounded optimization
nlss_loss = 'soft_l1' #['linear','soft_l1','huber','cauchy','arctan']
weight_method = 'rr' #['ols','rr']
upper_ratio = 1.8
lower_ratio = 0.02
voxel_ratio = 0.25
tr_ratio = 0.1
max_voxel = 5000
max_tr = 500
from_array = False
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
#can also read from file
else:
    prefix = '/home/hadoop/TFA/tests/115pieman_labels/'
    data_files = []
    R = []
    R0=np.load(prefix+'R.npy')
    for i in np.arange(nsubjs):
        data_files.append(prefix+'group1_s'+str(i)+'.npy')
        R.append(R0)
    my_data,my_R =  read_data(data_files,R,nsubjs,rank,size)

args=HtfaArgs(K, nsubjs,max_outer_iter,max_inner_iter,threshold,
              nlss_method,nlss_loss,weight_method,upper_ratio,lower_ratio,
              voxel_ratio,tr_ratio,max_voxel,max_tr)
              
fit_htfa(my_data,my_R,args)
