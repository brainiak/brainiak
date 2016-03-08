import numpy as np
from numpy.linalg import solve 
import pip


"""
convert an upper/lower triangular matrix in 1D to a full 2D symmetric matrix
"""
<<<<<<< HEAD
=======


>>>>>>> dc0a1e7ff8110e6f2bc513bc94dc48abbec9c997
def from_tri_2_sym(tri,K):
   symm = np.zeros((K,K))
   inds = np.triu_indices_from(symm)   
   symm[inds] = tri   
   symm[(inds[1], inds[0])] = tri
   return symm

def from_sym_2_tri(symm):
   inds = np.triu_indices_from(symm)   
   tri = symm[inds]
   return tri   
   
def fast_inverse(a):    
    identity = np.identity(a.shape[1],dtype=a.dtype)    
    inva = np.linalg.solve(a,identity)
    return inva    

   
