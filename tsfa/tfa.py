from sklearn.cluster import KMeans
from scipy.optimize import least_squares
from scipy.spatial import distance
from scipy import stats
import numpy as np
import time
import sys


def initTfa(R, K):
    kmeans = KMeans(init='k-means++', n_clusters=K, n_init=10)
    kmeans.fit(R)
    centers = kmeans.cluster_centers_
    #range of values in each column
    widths = (1.0/np.max(np.ptp(R,axis=0)))*np.ones((K,1))
    return centers,widths    

def getFactors(R,centers,widths):
   dist = distance.cdist(R,centers,'euclidean')
   F = (np.exp(-dist.T*widths)).T
   F = stats.zscore(F,axis=1, ddof=1)
   return F

def getWeights(W,F,weightMethod):
   beta = np.var(W)
   K = F.shape[1]
   transF = F.T
   if weightMethod == 'rr':
       Z=np.linalg.pinv(transF.dot(F)+ beta*np.identity(K)).dot(transF.dot(W))  
   elif weightMethod == 'ols':   
       Z=np.linalg.pinv(transF.dot(F)).dot(transF.dot(W)) 
   return Z
   
def getBounds(R,K,UW,LW):
    dim = R.shape[1]
    diameter = np.max(np.ptp(R,axis=0))
    lower = np.zeros((1,dim + 1))
    lower[0,0:dim] = np.nanmin(R, axis=0)
    lower[0,dim] = 1.0/(UW*diameter)
    finalLower = np.repeat(lower,K,axis=0)   
    upper = np.zeros((1,dim + 1))
    upper[0,0:dim] = np.nanmax(R, axis=0)
    upper[0,dim] = 1.0/(LW*diameter)
    finalUpper = np.repeat(upper,K,axis=0) 
    return (finalLower.reshape((K*(dim+1))),finalUpper.reshape((K*(dim+1))))
   
#def residual(estimate,K,dim,R, W,Z ):
def residual(estimate,K,dim,R,X,S,W,weightMethod):
    data = estimate.reshape((K,dim+1))
    centers = np.zeros((K,dim))
    widths = np.zeros((K,1))
    centers = data[:,0:dim]
    widths[:,0] = data[:,dim]
    F = getFactors(R,centers,widths)
    Z = getWeights(W,F,weightMethod)
    err = X - F.dot(Z.dot(S))
    #least_squares requires return value to be at most 1D array
    finalErr = np.linalg.norm(err, axis=1)
    #finalErr = np.zeros((err.size))
    #finalErr[:] = err.ravel()
    return finalErr

#def getCentersWidths(R,W,Z,initCenters,initWidths,method,loss,UW,LW):
def getCentersWidths(R,X,S,W,initCenters,initWidths,method,loss,UW,LW,weightMethod):
   K = initCenters.shape[0]
   dim = initCenters.shape[1]
   initEstimate = np.concatenate((initCenters,initWidths),axis=1)
   bounds =getBounds(R,K,UW,LW)      
   #least_squares only accept x in 1D format 
   initData = initEstimate.ravel()
   #finalEstimate = least_squares(residual,initData,args=(K,dim,R,W,Z), method=method, loss=loss, bounds=bounds)  
   finalEstimate = least_squares(residual,initData,args=(K,dim,R,X,S,W,weightMethod), method=method, loss=loss, bounds=bounds)  
   print 'tfa cost ',finalEstimate.cost
   sys.stdout.flush()
   result = finalEstimate.x.reshape((K,dim+1))
   centers = np.zeros((K,dim))
   widths = np.zeros((K,1))
   centers = result[:,0:dim]
   widths[:,0] = result[:,dim]
   return centers,widths,finalEstimate.cost
   

def fitTfa(miter,weightMethod,threshold,method,loss,UW,LW,X,S,W,R,centers,widths):
    """
    j = 0
    lastObj = np.finfo(float).max
    obj = 0
    while j < miter and (np.fabs(obj - lastObj)  > threshold):
        lastObj = obj
        F = getFactors(R,centers,widths)
        Z = getWeights(W,F,weightMethod)
        centers,widths,obj = getCentersWidths(R,W,Z,centers,widths,method,loss,UW,LW)  
        j += 1                
    """
    
    centers,widths,obj = getCentersWidths(R,X,S,W,centers,widths,method,loss,UW,LW,weightMethod)           
    F = getFactors(R,centers,widths)
    Z = getWeights(W,F,weightMethod)
    W = F.dot(Z)
    return W,F,Z   
                

