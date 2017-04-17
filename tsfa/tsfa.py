import numpy as np
import sys
import os
import time
from scipy import stats
from tfa import init_centers_widths, fitTfa



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

    T = []
    centers = []
    widths = []
    F = []
    W = []
    for m in xrange(nsubjs):
        nvoxel = X[m].shape[0]
        dim = R[m].shape[1]
        T.append(np.zeros(shape=(nvoxel, nfeature)))
        F.append(np.zeros(shape=(nvoxel, K)))
        W.append(np.zeros(shape=(K, nfeature)))
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
        T[m] = Q
        S = S + W[m].T.dot(X[m])
        
        #initialize tfa
        centers[m],widths[m] = init_centers_widths(R[m], K)
        
    S = S/float(nsubjs)

    for i in range(niter):
        for m in range(nsubjs):
            print '.',
            sys.stdout.flush()

            Am = X[m].dot(S.T)
            pert = np.zeros((Am.shape))
            np.fill_diagonal(pert, 1)
            Um, sm, Vm = np.linalg.svd(Am+0.001*pert, full_matrices=False)

            T[m] = Um.dot(Vm)  # W = UV^T
            
            if args.tfaEmbedded:   
                T[m],F[m],W[m] = tsfa_fit_tfa(miter,args.tfa_weight,args.threshold,args.tfa_method, args.tfa_loss,args.UW,args.LW, X[m],S,T[m],R[m],centers[m],widths[m],True)

        S = np.zeros((nfeature, nTR))
        for m in range(nsubjs):
            S = S + T[m].T.dot(X[m])
        S = S/float(nsubjs)  
        print '\n'   
        
    
    if not args.tfaEmbedded: 
        start_time = time.time()
        for m in range(nsubjs): 
            T[m],F[m],W[m] = tsfa_fit_tfa(miter,args.tfa_weight,args.threshold,args.tfa_method, args.tfa_loss,args.UW,args.LW, X[m],S,T[m],R[m],centers[m],widths[m],True)
        print("total fitTfa took %s seconds ---" % (time.time() - start_time))
        sys.stdout.flush()
  
    print 'tsfa cost ',obj_func(X, T, S,nsubjs)

    return T,F,W,S


def obj_func(X, T, S, nsubjs):
    obj_val_tmp = 0
    for m in range(nsubjs):
        obj_val_tmp += np.linalg.norm(X[m] - T[m].dot(S), 'fro')**2    
    return obj_val_tmp 

def align(movie_data, R, options, args, lrh):
    print 'SRM start\n',
    sys.stdout.flush()
    # assume data is zscored   
    T,F,W,S = tsfa(movie_data, R, options, args)    
    print 'SRM end\n'

    return T,F,W,S    
    
