from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy import stats
from utils import from_tri_2_sym, from_sym_2_tri, fast_inv
import numpy as np
import time
import sys
import math
import time

class TfaArgs(object):
    def __init__(self,max_iter,threshold,K,nlss_method,nlss_loss,weight_method,
                 upper_ratio,lower_ratio,max_sample_tr,max_sample_voxel,sample_scaling, bounds):            
        self.miter = max_iter
        self.threshold = threshold
        self.K= K
        self.nlss_method = nlss_method
        self.nlss_loss = nlss_loss
        self.weight_method = weight_method
        self.upper_ratio = upper_ratio
        self.lower_ratio = lower_ratio
        self.max_sample_tr = max_sample_tr
        self.max_sample_voxel = max_sample_voxel
        self.sample_scaling = sample_scaling 
        self.center_width_bounds = bounds 

def map_update(prior_mean,prior_cov,global_cov,new_observation,nsubjs,dim):
    scaled = global_cov/float(nsubjs)    
    common=fast_inv(prior_cov + scaled)
    observation_mean = np.mean(new_observation,axis=1)   
    posterior_mean = prior_cov.dot(common.dot(observation_mean)) + scaled.dot(common.dot(prior_mean))
    posterior_cov = prior_cov.dot(common.dot(scaled))
    return posterior_mean,posterior_cov
    
def map_update_posterior(global_prior,gather_posterior,K,nsubjs,dim,map_offset,cov_vec_size):    
    global_posterior = global_prior.copy()
    prior_centers = global_prior[0:map_offset[1]].copy().reshape(K,dim)    
    prior_widths = global_prior[map_offset[1]:map_offset[2]].copy().reshape(K,1)
    prior_centers_mean_cov = global_prior[map_offset[2]:map_offset[3]].copy().reshape(K,cov_vec_size)
    prior_widths_mean_var = global_prior[map_offset[3]:map_offset[4]].copy().reshape(K,1)
    global_centers_cov = global_prior[map_offset[4]:map_offset[5]].copy().reshape(K,cov_vec_size)
    global_widths_var = global_prior[map_offset[5]:].copy().reshape(K,1)
    center_size = K*dim 
    posterior_size = center_size + K
    for k in np.arange(K):   
        next_centers = np.zeros((dim, nsubjs))
        next_widths = np.zeros(nsubjs)        
        for s in np.arange(nsubjs):   
            center_start = s*posterior_size         
            width_start = center_start + center_size            
            next_centers[:,s] = gather_posterior[center_start+k*dim:center_start+(k+1)*dim].copy()
            next_widths[s] = gather_posterior[width_start+k].copy()        
        
        #centers    
        posterior_mean,posterior_cov = map_update(prior_centers[k].T.copy(), from_tri_2_sym(prior_centers_mean_cov[k],dim),
                                                  from_tri_2_sym(global_centers_cov[k],dim), next_centers,nsubjs,dim)      
        posterior_cov_inv = fast_inv(posterior_cov)
        global_posterior[k*dim:(k+1)*dim] = posterior_mean.T
        global_posterior[map_offset[2]+k*cov_vec_size:map_offset[2]+(k+1)*cov_vec_size] = from_sym_2_tri(posterior_cov)
                        
        #widths
        scaled =global_widths_var[k]/float(nsubjs)
        common=1.0/(prior_widths_mean_var[k] + scaled) 
        observation_mean = np.mean(next_widths)
        tmp = common*scaled
        global_posterior[map_offset[1]+k] = prior_widths_mean_var[k]*common*observation_mean + tmp*prior_widths[k]        
        global_posterior[map_offset[3]+k] = prior_widths_mean_var[k]*tmp
  
    return global_posterior

def converged(prior, posterior,K,dim,threshold):    
    prior_centers = prior[0:K*dim].reshape((K,dim))    
    posterior_centers = posterior[0:K*dim].reshape((K,dim))
    posterior_widths = posterior[K*dim:].reshape((K,1)) 
    #linear assignment on centers 
    cost=distance.cdist(prior_centers,posterior_centers,'euclidean')
    _,col_ind=linear_sum_assignment(cost)
    posterior[0:K*dim]=posterior_centers[col_ind].reshape(K*dim)
    posterior[K*dim:]=posterior_widths[col_ind].reshape(K)
    print 'sorted_posterior\n',np.hstack((posterior[0:K*dim].reshape((K,dim)), 
                                          posterior[K*dim:].reshape((K,1))
                                          ))
    mse = mean_squared_error(prior, posterior, multioutput='uniform_average')
    print "mse is ",mse
    if mse > threshold:
        return False
    else:
        return True

def get_global_prior(R,K,dim,cov_vec_size):
    global_prior = np.zeros(K*(dim+2*cov_vec_size+3))
    centers,widths = init_centers_widths(R, K)
    center_cov= 1.0/math.pow(K,2/3.0)*np.cov(R.T)    
    center_cov_all = np.tile(from_sym_2_tri(center_cov),K)
    width_var = math.pow(np.nanmax(np.std(R,axis=0)),2)
    width_var_all = np.tile(width_var,K)
    map_offset,global_prior_size = get_map_offset(K,dim,cov_vec_size,True)
    #center mean mean
    global_prior[0:map_offset[1]] = centers.ravel()  
    #width mean mean  
    global_prior[map_offset[1]:map_offset[2]] = widths.ravel()
    #center mean cov
    global_prior[map_offset[2]:map_offset[3]] = center_cov_all.ravel()
    #width mean var
    global_prior[map_offset[3]:map_offset[4]] = width_var_all.ravel()
    #center cov
    global_prior[map_offset[4]:map_offset[5]] = center_cov_all.ravel()
    #width var
    global_prior[map_offset[5]:] = width_var_all.ravel()
    return global_prior,map_offset,global_prior_size

def get_map_offset(K,dim,cov_vec_size,is_root):
    if is_root:
        #there are 6 fileds in global prior on root
        nfield = 6 
        field_size = K*np.array([dim,1, cov_vec_size,1,cov_vec_size,1])
        global_prior_size = np.sum(field_size)
    else:    
        #there are 4 fileds in global prior on rest nodes
        nfield = 4
        field_size = K*np.array([dim,1, cov_vec_size, 1])
        
    map_offset = np.zeros(nfield,dtype=np.float)  
    for i in np.arange(nfield-1)+1:
        map_offset[i] = map_offset[i-1]+field_size[i-1] 
    
    if is_root:
        return map_offset,global_prior_size
    else:
        return map_offset           


def init_centers_widths(R, K):
    kmeans = KMeans(init='k-means++', n_clusters=K, n_init=10, random_state=100)
    kmeans.fit(R)
    centers = kmeans.cluster_centers_
    #range of values in each column
    widths = get_diameter(R)*np.ones((K,1)) 
    np.set_printoptions(suppress=True) 
    print 'init_global_template\n',np.hstack((centers,widths)) 
    return centers,widths    

#centers: K*3, widths: K*1
#F: V*K
def get_factors(R,centers,widths):
   K=centers.shape[0]
   F=np.zeros((R.shape[0],K))  
   for k in np.arange(K):
       F[:,k] = np.exp(-1.0/widths[k]*np.sum(np.power(R[:,0]-centers[k,0],2)+
                       np.power(R[:,1]-centers[k,1],2) +
                       np.power(R[:,2]-centers[k,2],2))) 
                         
   return F

#W: K*T
def get_weights(data,F, weight_method):
   beta = np.var(data)
   K = F.shape[1]
   trans_F = F.T.copy()
   test = trans_F.dot(F)   
   if  weight_method == 'rr':       
       W=np.linalg.solve(trans_F.dot(F)+ beta*np.identity(K),trans_F.dot(data))  
   elif  weight_method == 'ols':   
       W=np.linalg.solve(trans_F.dot(F),trans_F.dot(data))  
   return W

def get_diameter(R):
    dim = R.shape[1]
    diameter = np.max(np.ptp(R,axis=0))
    return diameter
   
def get_bounds(R,K,upper_ratio,lower_ratio):
    dim = R.shape[1]
    diameter = get_diameter(R)
    lower = np.zeros((1,dim + 1))
    lower[0,0:dim] = np.nanmin(R, axis=0)
    lower[0,dim] = lower_ratio*diameter
    final_lower = np.repeat(lower,K,axis=0)   
    upper = np.zeros((1,dim + 1))
    upper[0,0:dim] = np.nanmax(R, axis=0)
    upper[0,dim] = upper_ratio*diameter    
    final_upper = np.repeat(upper,K,axis=0)
    bounds =  (final_lower.reshape((K*(dim+1))),final_upper.reshape((K*(dim+1))))
    return bounds

def residual(estimate,K,dim,R, X,W,global_centers,global_widths,
             global_center_mean_cov,global_width_mean_var,sample_scaling,data_sigma):
    centers = estimate[0:K*dim].reshape((K,dim))
    widths = estimate[K*dim:].reshape((K,1))
    F = get_factors(R,centers,widths)    
    err = data_sigma*np.fabs(X - F.dot(W))
    #least_squares requires return value to be at most 1D array
    recon = X.size
    finalErr = np.zeros(err.size)
    finalErr[:] = err.ravel()    
    return np.sum(err,axis=0)    
"""    
    finalErr = np.zeros(recon+2*K)    
    finalErr[0:recon] = err.ravel()
    #center error        
    for k in np.arange(K):
        diff = centers[k]-global_centers[k]
        cov = from_tri_2_sym(global_center_mean_cov[k],dim)
        finalErr[recon + k] = math.sqrt(0.5*sample_scaling*diff.dot(np.linalg.solve(cov,diff.T)))
    #width error
    base = recon+K
    for k in np.arange(K):
        diff = widths[k]-global_widths[k]
        finalErr[base + k] = math.sqrt(0.5*sample_scaling*(1.0/global_width_mean_var[k])*math.pow(diff,2))            
    return finalErr
"""   
def residual_recon(estimate,K,dim,R, X,W,data_sigma):
    centers = estimate[0:K*dim].reshape((K,dim))
    widths = estimate[K*dim:].reshape((K,1))
    F = get_factors(R,centers,widths)
    err = data_sigma*(X - F.dot(W))
    #least_squares requires return value to be at most 1D array
    finalErr = np.zeros(err.size)
    finalErr[:] = err.ravel()    
    return finalErr
    
def get_centers_widths(R,X,W,init_centers,init_widths,args,global_centers,global_widths,
                       global_center_mean_cov,global_width_mean_var):
   K = init_centers.shape[0]
   dim = init_centers.shape[1]
   init_estimate = np.concatenate((init_centers,init_widths),axis=1)    
   data_sigma = 1.0/math.sqrt(2.0)*np.std(X)
   #least_squares only accept x in 1D format 
   init_data = init_estimate.ravel()
   final_estimate = least_squares(residual,init_data,
                                  args=(K,dim,R,X,W,global_centers,global_widths,
                                  global_center_mean_cov,global_width_mean_var,args.sample_scaling,data_sigma), 
                                  method=args.nlss_method, loss=args.nlss_loss, bounds=args.center_width_bounds,
                                  verbose=2) 
   print 'nlss status ',final_estimate.status,final_estimate.success
   return final_estimate.x,final_estimate.cost    

   
def fit_tfa(local_prior,global_prior,map_offset,data,R,args):
    dim = R.shape[1]
    cov_size = (map_offset[3]-map_offset[2])/args.K
    global_centers = global_prior[0:map_offset[1]].copy().reshape(args.K,dim)    
    global_widths = global_prior[map_offset[1]:map_offset[2]].copy().reshape(args.K,1)
    global_center_mean_cov = global_prior[map_offset[2]:map_offset[3]].copy().reshape(args.K,cov_size)
    global_width_mean_var = global_prior[map_offset[3]:].copy().reshape(args.K,1)
    inner_converged = False 
    n = 0
    while n < args.miter and not inner_converged:
        local_posterior = fit_tfa_inner(local_prior,global_centers,global_widths,
                                        global_center_mean_cov,global_width_mean_var,data,R,args)        
        if converged(local_prior,local_posterior,args.K,dim,args.threshold):
             inner_converged = True                   
        else:
             local_prior = local_posterior
        n += 1    
    return local_posterior

def fit_tfa_inner(local_prior,global_centers,global_widths,global_center_mean_cov,
                  global_width_mean_var,data,R,args):                 
    dim = R.shape[1]
    nvoxel = data.shape[0]
    ntr = data.shape[1]  
    voxel_indices = np.random.choice(nvoxel, args.max_sample_voxel, replace=False)
    sample_voxels = np.zeros(nvoxel).astype(bool)
    sample_voxels[voxel_indices] = True  
    trs_indices = np.random.choice(ntr, args.max_sample_tr, replace=False)    
    sample_trs = np.zeros(ntr).astype(bool)    
    sample_trs[trs_indices] = True
    curr_data = data[voxel_indices]
    curr_data = curr_data[:,trs_indices].copy()
    curr_R = R[voxel_indices].copy()
    centers = local_prior[0:args.K*dim].copy().reshape(args.K,dim)
    widths = local_prior[args.K*dim:].copy().reshape(args.K,1)    
    F = get_factors(curr_R,centers,widths)
    W = get_weights(curr_data,F, args.weight_method)   
    local_posterior,tfa_cost = get_centers_widths(curr_R,curr_data,W,centers,widths,args,global_centers,global_widths,
                                          global_center_mean_cov,global_width_mean_var)  
    print 'tfa cost ',tfa_cost
    sys.stdout.flush()  
    return local_posterior
    
 

