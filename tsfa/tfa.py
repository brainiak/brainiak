from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy import stats
from utils import from_tri_2_sym, from_sym_2_tri, fast_inverse
import numpy as np
import time
import sys
import math

class TfaArgs(object):
    def __init__(self,miter,threshold,K,nlss_method,nlss_loss,weight_method,
                 upper_ratio,lower_ratio,max_sample_tr,max_sample_voxel):            
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

def map_udpate(prior_mean,prior_cov,global_cov,new_observation,nsubjs):
    scaled = global_cov/(nsubjs.astype(float))
    common=fast_inverse(prior_cov + scaled)
    posterior_mean = prior_cov.dot(common.dot(np.mean(new_observation,axis=1))) + scaled.dot(common.dot(prior_mean))
    posterior_cov = prior_cov.dot(common.dot(scaled))
    return posterior_mean,posterior_cov
    
def map_update_posterior(global_prior,gather_posterior,K,nsubjs,dim,map_offset,cov_vec_size):
    global_posterior = global_prior.copy()
    prior_centers = global_prior[0:map_offset[1]]    
    prior_widths = global_prior[map_offset[1]:map_offset[2]]
    prior_center_cov = global_prior[map_offset[4]:map_offset[5]]
    prior_width_var = global_prior[map_offset[5]:map_offset[6]]
    global_center_cov = global_prior[map_offset[6]:map_offset[7]]
    global_width_var = global_prior[map_offset[7]:]
    center_sie = K*dim 
    posterior_size = center_size + K
    for k in np.arange(K):   
        next_centers = np.zeros((dim, nsubj))
        next_widths = np.zeros(1,nsubjs)
        for s in np.arange(nsubjs):
            center_start = s*posterior_size
            width_start = center_start + center_sie
            next_centers[:,s] = gather_posterior[center_start:width_start]
            next_widths[:,s] = gather_posterior[width_start:width_start+posterior_size]
        end
        
        #centers    
        posterior_mean,posterior_cov = map_update(prior_centers[k,:].T.copy(), from_tri_2_sym(prior_center_cov[k,:],K), from_tri_2_sym(global_center_cov[k,:],K), next_centers,nsubjs)      
        posterior_cov_inv = fast_inverse(posterior_cov)
        global_posterior[k*center_sie:(k+1)*center_size] = posterior_mean.T
        global_posterior[map_offset[4]+k*cov_vec_size:map_offset[4]+(k+1)*cov_vec_size] = from_sym_2_tri(posterior_cov)
        global_posterior[map_offset[2]+k*cov_vec_size:map_offset[2]+(k+1)*cov_vec_size] = from_sym_2_tri(posterior_cov_inv)
                
        #widths
        global_posterior[map_offset[1]+k],global_posterior[map_offset[5]+k] = map_update(prior_widths[k], prior_widths_var[k], global_widths_var[k], next_widths,nsubjs)        
        global_posterior[map_offset[3]+k] = 1.0/lobal_posterior[map_offset[5]+k]

    return global_posterior

def converged(prior, posterior,K,dim,threshold):
    posterior_2d = posterior.reshape(K,dim+1)
    cost=distance.cdist(prior.reshape(K,dim+1),posterior_2d,'euclidean')
    _,col_ind=linear_sum_assignment(cost)
    sorted_posterior=posterior_2d[col_ind].ravel()
    if mean_squared_error(prior, sorted_posterior, multioutput='uniform_average') > threshold:
        return False
    else:
        return True

def get_global_prior(R,K,dim,cov_vec_size):
    global_prior = np.zeros(K*(dim+2*cov_vec_size+3))
    centers,widths = init_centers_widths(R, K)
    center_cov= 1.0/math.pow(K,2/3.0)*np.cov(R.T)
    center_cov_inv = fast_inverse(center_cov)
    center_cov_all = np.tile(from_sym_2_tri(center_cov),K)
    center_cov_inv_all = np.tile(from_sym_2_tri(center_cov_inv),K)
    width_var = np.nanmax(np.std(R,axis=0))   
    width_var = math.pow(width_var,2)
    width_var_inv = 1.0/width_var
    width_var_all = np.tile(width_var,K)
    width_var_inv_all = np.tile(width_var_inv,K)
    map_offset,global_prior_size = get_map_offset(K,dim,cov_vec_size,True)
    #center mean mean
    global_prior[0:map_offset[1]] = centers.ravel()  
    #width mean mean  
    global_prior[map_offset[1]:map_offset[2]] = widths.ravel()
    global_prior[map_offset[2]:map_offset[3]] = center_cov_inv_all.ravel()
    global_prior[map_offset[3]:map_offset[4]] = width_var_inv_all.ravel()
    #center mean cov
    global_prior[map_offset[4]:map_offset[5]] = center_cov_all.ravel()
    #width mean var
    global_prior[map_offset[5]:map_offset[6]] = width_var_all.ravel()
    #center cov
    global_prior[map_offset[6]:map_offset[7]] = center_cov_all.ravel()
    #width var
    global_prior[map_offset[7]:] = width_var_all.ravel()
    return global_prior,map_offset,global_prior_size

def get_map_offset(K,dim,cov_vec_size,is_root):
    if is_root:
        #there are 8 fileds in global prior on root
        nfield = 8 
        field_size = K*np.array([dim,1, cov_vec_size, 1,cov_vec_size,1,cov_vec_size,1])
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
    kmeans = KMeans(init='k-means++', n_clusters=K, n_init=10)
    kmeans.fit(R)
    centers = kmeans.cluster_centers_
    #range of values in each column
    widths = (1.0/np.max(np.ptp(R,axis=0)))*np.ones((K,1))
    return centers,widths    

#centers: K*3, widths: K*1
#F: V*K
def get_factors(R,centers,widths):
   dist = distance.cdist(R,centers,'euclidean')
   F = (np.exp(-dist.T*widths)).T.copy()
   F = stats.zscore(F,axis=1, ddof=1)
   return F

#W: K*T
def get_weights(data,F, weight_method):
   beta = np.var(data)
   K = F.shape[1]
   trans_F = F.T.copy()
   if  weight_method == 'rr':
       W=fast_inverse(trans_F.dot(F)+ beta*np.identity(K)).dot(trans_F.dot(data))  
   elif  weight_method == 'ols':   
       W=fast_inverse(trans_F.dot(F)).dot(trans_F.dot(data)) 
   return W
   
def get_bounds(R,K,upper_ratio,lower_ratio):
    dim = R.shape[1]
    diameter = np.max(np.ptp(R,axis=0))
    lower = np.zeros((1,dim + 1))
    lower[0,0:dim] = np.nanmin(R, axis=0)
    lower[0,dim] = 1.0/(upper_ratio*diameter)
    final_lower = np.repeat(lower,K,axis=0)   
    upper = np.zeros((1,dim + 1))
    upper[0,0:dim] = np.nanmax(R, axis=0)
    upper[0,dim] = 1.0/(lower_ratio*diameter)
    final_upper = np.repeat(upper,K,axis=0) 
    return (final_lower.reshape((K*(dim+1))),final_upper.reshape((K*(dim+1))))
   
def residual(estimate,K,dim,R, W,Z):
    data = estimate.reshape((K,dim+1))
    centers = np.zeros((K,dim))
    widths = np.zeros((K,1))
    centers = data[:,0:dim]
    widths[:,0] = data[:,dim]
    F = get_factors(R,centers,widths)
    err = W - F.dot(Z)
    #least_squares requires return value to be at most 1D array
    finalErr = np.zeros((err.size))
    finalErr[:] = err.ravel()
    return finalErr

def residual_full(estimate,K,dim,R,X,S,W, weight_method):
    data = estimate.reshape((K,dim+1))
    centers = data[:,0:dim].reshape(K,dim)
    widths[:,0] = data[:,dim].reshape(k1,)
    F = get_factors(R,centers,widths)
    Z = get_weights(W,F, weight_method)
    err = X - F.dot(Z.dot(S))
    #least_squares requires return value to be at most 1D array
    finalErr = np.linalg.norm(err, axis=1)
    return finalErr

def get_centers_widths(R,W,Z,init_centers,init_widths,nlss_method,loss,upper_ratio,lower_ratio):
   K = init_centers.shape[0]
   dim = init_centers.shape[1]
   init_estimate = np.concatenate((init_centers,init_widths),axis=1)
   bounds =get_bounds(R,K,upper_ratio,lower_ratio)   
   #least_squares only accept x in 1D format 
   init_data = init_estimate.ravel()
   final_estimate = least_squares(residual,init_data,args=(K,dim,R,W,Z), method=nlss_method, loss=loss, bounds=bounds)  
   print 'tfa cost ',final_estimate.cost
   sys.stdout.flush()
   result = final_estimate.x.reshape((K,dim+1))
   centers = np.zeros((K,dim))
   widths = np.zeros((K,1))
   centers = result[:,0:dim]
   widths[:,0] = result[:,dim]
   return centers,widths,final_estimate.cost


def get_centers_widths_full(R,X,S,W,init_centers,init_widths,nlss_method,loss,upper_ratio,lower_ratio, weight_method):
   K = init_centers.shape[0]
   dim = init_centers.shape[1]
   init_estimate = np.concatenate((init_centers,init_widths),axis=1)
   bounds =get_bounds(R,K,upper_ratio,lower_ratio)   
   #least_squares only accept x in 1D format 
   init_data = init_estimate.ravel()   
   final_estimate = least_squares(residual_full,init_data,args=(K,dim,R,X,S,W, weight_method), method=nlss_method, loss=loss, bounds=bounds)  
   print 'tfa cost ',final_estimate.cost
   sys.stdout.flush()
   result = final_estimate.x.reshape((K,dim+1))
   centers = np.zeros((K,dim))
   widths = np.zeros((K,1))
   centers = result[:,0:dim]
   widths[:,0] = result[:,dim]
   return centers,widths,final_estimate.cost
   
def fit_tfa(curr_prior,data,R,args):
    dim = R.shape[1]
    inner_converged = False 
    n = 0
    while n < max_inner_iter and not inner_converged:
        curr_posterior = fit_tfa_inner(curr_prior,global_prior,data,R,args)        
        if converged(curr_prior,curr_posterior,args.K,dim,args.threshold):
             inner_converged = True                   
        else:
             curr_prior = curr_posterior
        n += 1    
    return curr_posterior

def fit_tfa_inner(curr_prior,global_prior,data,R,args):
    dim = R.shape[1]
    ntr = data.shape[0]
    nvoxel = data.shape[1]
    tr_indices = np.random.choice(ntr, max_sample_tr, replace=False)
    voxel_indices = np.random.choice(nvoxel, max_sample_voxel, replace=False)
    curr_data = data[voxel_indices,tr_indices].copy
    curr_R = R[voxel_indices].copy
    centers = curr_prior[0:args.K*dim].reshape(K,dim)
    widths = curr_prior[args.K*dim:].reshape(K,1)
    F = get_factors(curr_R,centers,widths)
    W = get_weights(curr_data,F, weight_method)
    centers,widths,_ = get_centers_widths(curr_R,curr_data,W,centers,widths,args.nlss_method,
                                          args.nlss_loss,args.upper_ratio,args.lower_ratio)
    curr_posterior = np.hstack((centers.reshape(1,args.K*dim),widths.reshape(1,args.K)).copy
    return curr_posterior
    
def tsfa_fit_tfa(miter, weight_method,threshold,nlss_method,loss,upper_ratio,lower_ratio,X,S,W,R,centers,widths,full):
    if full:
        centers,widths,obj = get_centers_widths_full(R,X,S,W,centers,widths,nlss_method,loss,upper_ratio,lower_ratio, weight_method) 
        F = get_factors(R,centers,widths)
        Z = get_weights(W,F, weight_method)
        W = F.dot(Z)  
    else:  
        j = 0
        lastObj = np.finfo(float).max
        obj = 0
        while j < miter and (np.fabs(obj - lastObj)  > threshold):
            lastObj = obj
            F = get_factors(R,centers,widths)
            Z = get_weights(W,F, weight_method)
            centers,widths,obj = get_centers_widths(R,W,Z,centers,widths,nlss_method,loss,upper_ratio,lower_ratio)  
            j += 1   

    return W,F,Z,centers,widths   

