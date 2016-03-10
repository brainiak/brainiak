import numpy as np
from numpy.linalg import solve 
import pip


"""
convert an upper/lower triangular matrix in 1D to a full 2D symmetric matrix
"""
def from_tri_2_sym(tri,dim):
   symm = np.zeros((dim,dim))
   symm[np.triu_indices(dim)] = tri
   return symm

def from_sym_2_tri(symm):
   inds = np.triu_indices_from(symm)   
   tri = symm[inds]
   return tri   
   
def fast_inv(a):    
    identity = np.identity(a.shape[1],dtype=a.dtype)    
    inva = np.linalg.solve(a,identity)
    return inva    

   
