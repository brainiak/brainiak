from mpi4py import MPI
import numpy as np
from utils import from_tri_2_sym, from_sym_2_tri, fast_inv
from tfa import TfaArgs,get_global_prior,converged,get_map_offset,fit_tfa,map_update_posterior


"""
data: is a list, each element is a V*T data of a subject
K: the number of latent factors
R: is a list, each element is a V*dim voxel coordinate matrix
"""


class HtfaArgs(object):
    def __init__(self,K,nsubjs,max_outer_iter=100,max_inner_iter=10,threshold=0.01,
                 nlss_method='trf',nlss_loss='soft_l1',weight_method='rr',upper_ratio=1.8,lower_ratio=0.02,
                 voxel_ratio=0.25,tr_ratio=0.1,max_voxel=5000,max_tr=500):  
        self.K = K 
        self.nsubjs = nsubjs         
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.threshold = threshold        
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.weight_method = weight_method
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.voxel_ratio = voxel_ratio
        self.tr_ratio = tr_ratio
        self.max_voxel = max_voxel
        self.max_tr = max_tr

def get_data(data,R,nsubjs,my_rank,size):
    data = []
    R = []        
    for idx, cur_data in enumerate(data):
      cur_rank = idx % size           
      if cur_rank == my_rank:
         data.append(cur_data)
         R.append(R[idx])
    return data,R
    
def read_data(data_files,R,nsubjs,my_rank,size):
    data = []
    R = []        
    for idx, name in enumerate(data_files):
      cur_rank = idx % size           
      if cur_rank == my_rank:        
         data.append(np.load(name))
         R.append(R[idx])
    return data,R

def get_offset(nsubjs,size,posterior_size):
    gather_size = np.zeros((1,size)).astype(int) 
    gather_offset = np.zeros((1,size)).astype(int) 
       
    for idx, s in enumerate(np.arange(nsubjs)):
      cur_rank = idx % size
      gather_size[0,cur_rank] += posterior_size 
      
    for idx in np.arange(size-1)+1:      
       gather_offset[0,idx] = gather_offset[0,idx - 1] + gather_size[0,idx - 1] 
    
    tuple_size = tuple(map(tuple, gather_size))
    tuple_offset = tuple(map(tuple, gather_offset))
    return tuple_size[0],tuple_offset[0]

def get_sample_size(nlocal_subjs, data):  
    max_sample_tr = np.zeros(nlocal_subjs)
    max_sample_voxel = np.zeros(nlocal_subjs)
    for idx,data in enumerate(data):
        nvoxel = data[idx].shape[0]
        ntr = data[idx].shape[1]
        max_sample_voxel[idx] = np.min(max_voxel,voxel_ratio*nvoxel)
        max_sample_tr[idx] = np.min(max_voxel,voxel_ratio*nvoxel)  

def fit_htfa(data,R,args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
         
    nsubjs = args.subjs
    dim = R[0].shape[1]
    cov_vec_size = np.sum(np.arange(dim)+1)    
    #centers,widths,centerCov,widthVar
    prior_bcast_size = K*(dim+2+cov_vec_size)
    #centers,widths
    posterior_size = K*(dim+1) 
    #map data to processes
    gather_size,gather_offset = get_offset(nsubjs,size,posterior_size)
    max_sample_tr, max_sample_voxel = get_sample_size(nlocal_subjs, data)
    nlocal_subjs = len(data)    
    local_posterior = np.zeros(nlocal_subjs*posterior_size)
        
    if rank == 0:
        idx = np.random.choice(nlocal_subjs,1)
        global_prior,map_offset,global_prior_size = get_global_prior(R[idx],K,dim,cov_vec_size)        
        gather_posterior = np.zeros(nsubjs*posterior_size)
        global_posterior = np.zeros(global_prior_size)
        last_global_posterior = np.zeros(global_prior_size)        
    else:
        global_prior = np.zeros(prior_bcast_size)
        map_offset = get_map_offset(K,dim,cov_vec_size,False)
        gather_posterior = None 
    
    m = 0 
    outer_converged = np.array([0])         
    while m < max_outer_iter and not outer_converged[0]:
       #root broadcast first 4 fields of global_prior to all nodes       
       comm.Bcast(global_prior[0:prior_bcast_size], root=0) 
       #each node loop over its data
       for s,subj_data in enumerate(data):
           local_prior = global_prior[0:posterior_size].copy()            
           tfa_args=TfaArgs(args.max_inner_iter,args.threshold,args.K,
                            args.nlss_method,args.nlss_loss,args.weight_method,
                            args.upper_ratio,args.lower_ratio,max_sample_tr[s],max_sample_voxel[s]) 
           local_posterior[s*posterior_size:(s+1)*posterior_size] = fit_tfa(local_prior,global_prior[0:prior_bcast_size],map_offset,
                                                                            subj_data,R[s],tfa_args)
               
       comm.Gatherv(local_posterior,[gather_posterior,gather_size,gather_offset, MPI.DOUBLE])
       #root updates update global_posterior
       if rank == 0:        
           global_posterior = map_update_posterior(global_prior,gather_posterior,K,nsubjs,dim,map_offset,cov_vec_size)
           if converged(global_posterior[0:posterior_size],global_prior[0:posterior_size],K,dim,threshold):
               outer_converged[0] = 1
           else:
               global_prior = global_posterior
               
       comm.Bcast(outer_converged,root=0)       
       m += 1


       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
