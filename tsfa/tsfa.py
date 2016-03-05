#!/usr/bin/env python

# Inference code for non-probabilistic Shared Response Model with varying number of voxels across subjects

# A Reduced-dimension fMRI Shared Response Model
# Po-Hsuan Chen, Janice Chen, Yaara Yeshurun-Dishon, Uri Hasson, James Haxby, Peter Ramadge
# Advances in Neural Information Processing Systems (NIPS), 2015.

# movie_data is a list of two dimensional arrays
# movie_data[m] is the data for subject m of size nvoxel x nTR
# nvoxels can be different across subjects

# By Cameron PH Chen @ Princeton

import numpy as np
import sys
import os
import time
from scipy import stats
from tfa import initTfa, fitTfa


def tsfa(movie_data, R, options, args):
    print 'Start TSFA K = ' + str(args.nfeature) + '\n'
    sys.stdout.flush()
    X = movie_data
    nsubjs = len(X)
    niter = args.niter
    miter = args.miter

    for m in range(nsubjs):
        assert X[0].shape[1] == X[m].shape[1], 'numbers of TRs are different among subjects'

    nTR = X[0].shape[1]

    align_algo = args.align_algo
    nfeature = args.nfeature
    K = args.tfaK

    for m in xrange(nsubjs):
        X[m] = stats.zscore(X[m].T, axis=0, ddof=1).T

    W = []
    centers = []
    widths = []
    F = []
    Z = []
    for m in xrange(nsubjs):
        nvoxel = X[m].shape[0]
        dim = R[m].shape[1]
        W.append(np.zeros(shape=(nvoxel, nfeature)))
        F.append(np.zeros(shape=(nvoxel, K)))
        Z.append(np.zeros(shape=(K, nfeature)))
        centers.append(np.zeros((K, dim)))
        widths.append(np.zeros((K, 1)))
    
    S = np.zeros((nfeature, nTR))   
    np.random.seed(args.randseed)
    #initialization        
    for m in xrange(nsubjs):
        nvoxel = X[m].shape[0]
            
        # initialize with random orthogonal matrix
        A = np.mat(np.random.random((nvoxel, nfeature)))
        Q, R_qr = np.linalg.qr(A)
        W[m] = Q
        S = S + W[m].T.dot(X[m])
        
        #initialize tfa
        centers[m],widths[m] = initTfa(R[m], K)
        
    S = S/float(nsubjs)

    for i in range(niter):
        for m in range(nsubjs):
            print '.',
            sys.stdout.flush()

            Am = X[m].dot(S.T)
            pert = np.zeros((Am.shape))
            np.fill_diagonal(pert, 1)
            Um, sm, Vm = np.linalg.svd(Am+0.001*pert, full_matrices=False)

            W[m] = Um.dot(Vm)  # W = UV^T
            
            if args.tfaEmbedded:   
                W[m],F[m],Z[m] = fitTfa(miter,args.tfaWeight,args.threshold,args.tfaMethod, args.tfaLoss,args.UW,args.LW, X[m],S,W[m],R[m],centers[m],widths[m])

        S = np.zeros((nfeature, nTR))
        for m in range(nsubjs):
            S = S + W[m].T.dot(X[m])
        S = S/float(nsubjs)  
        print '\n'   
        
    
    if not args.tfaEmbedded: 
        start_time = time.time()
        for m in range(nsubjs): 
            W[m],F[m],Z[m] = fitTfa(miter,args.tfaWeight,args.threshold,args.tfaMethod, args.tfaLoss,args.UW,args.LW, X[m],S,W[m],R[m],centers[m],widths[m])
        print("total fitTfa took %s seconds ---" % (time.time() - start_time))
        sys.stdout.flush()
  
    print 'tsfa cost ',obj_func(X, W, S,nsubjs)

    return W,S


def obj_func(X, W, S, nsubjs):
    obj_val_tmp = 0
    for m in range(nsubjs):
        obj_val_tmp += np.linalg.norm(X[m] - W[m].dot(S), 'fro')**2    
    return obj_val_tmp 

def align(movie_data, R, options, args, lrh):
    print 'SRM start\n',
    sys.stdout.flush()
    # assume data is zscored   
    W, S = tsfa(movie_data, R, options, args)    
    print 'SRM end\n'

    return W, S    
    
