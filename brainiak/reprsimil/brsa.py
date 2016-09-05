#  Copyright 2016 Mingbo Cai, Princeton Neuroscience Instititute,
#  Princeton University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Bayesian Representational Similarity Analysis (BRSA)
This implementation is based on the following publications:
.. [Cai2016] "Unbiased Bayesian estimation of
      neural representational similarity structure",
   M.B. Cai, N. Schuck, J. Pillow, Y. Niv,
   Neural Information Processing System 29, 2016.
"""

# Authors: Mingbo Cai
# Princeton Neuroscience Institute, Princeton University, 2016

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import warnings
import time
from sklearn.base import BaseEstimator
from sklearn.utils import assert_all_finite
import logging
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)

__all__ = [
    "BRSA",
]


class BRSA(BaseEstimator):
    """Bayesian representational Similarity Analysis (BRSA)
    Given the time series of preprocessed neural imaging data and
    the hypothetical neural response (design matrix) to
    each experimental condition of interest,
    calculate the shared covariance matrix of
    the voxels(recording unit)' response to each condition,
    and the relative SNR of each voxels.
    The relative SNR could be considered as the degree of contribution
    of each voxel to this shared covariance matrix.
    A correlation matrix converted from the covariance matrix
    will be provided as a quantification of neural representational similarity.

    .. math:: \textbf{Y = \textbf{X} \cdot \mbox{\boldmath{$\beta$}}
        + \mbox{\boldmath{$\epsilon$}}
    .. math:: \mbox{\boldmath{$\beta$}}_i \sim \textbf{N}(0,(s_{i}
        \sigma_{i})^2 \textbf{U})

    Parameters
    ----------
    n_iter : int, default: 200
        Number of maximum iterations to run the algorithm.
    rank : int, default: None
        The rank of the covariance matrix.
        If not provided, the covariance matrix will be assumed
        to be full rank. When you have many conditions
        (e.g., calculating the similarity matrix of responses to each event),
        you might want to start with specifying a lower rank and use metrics
        such as AIC or BIC to decide the optimal rank.
    GP_space: boolean, default: False
        Whether to impose a Gaussion Process (GP) prior on the log(pseudo-SNR).
        If true, the GP has a kernel defined over spatial coordinate.
        This is experimental and slow. I find that when SNR is genrally low,
        smoothness can be overestimated. But I think this is better than
        not imposing any regularization.
    GP_inten: boolean, defualt: False
        Whether to include a kernel definved over the intensity of image.
        GP_space should be True as well if you want to use this,
        because the smoothness should be primarily in space.
        Smoothness in intensity is just complementary.
    verbose : boolean, default: False
        Verbose mode flag.
    pad_DC: boolean, default: False
        A column of all 1's will be padded to end of the design matrix
        to account for residual baseline component in the signal.
        We recommend removing DC component in your data but still
        set this as True. If you include a baseline column yourself,
        then you should check this as False.
        In future version, we will include a seperate input
        argument for all regressors you are not interested in,
        such as DC component and motion parameters.
    Attributes
    ----------
    U_ : The shared covariance matrix, shape=[condition,condition].
    L_ : The Cholesky factor of the shared covariance matrix
        (lower-triangular matrix), shape=[condition,condition].
    C_: the correlation matrix derived from the shared covariance matrix,
        shape=[condition,condition]
    nSNR_ : array, shape=[voxels,]
        The pseuso-SNR of all voxels.
        They are normalized such that the geometric mean is 1
    sigma_ : array, shape=[voxels,]
        The estimated standard deviation of the noise in each voxel
        Assuming AR(1) model, this means the standard deviation
        of the refreshing noise.
    rho_ : array, shape=[voxels,]
        The estimated autoregressive coefficient of each voxel
    lGPspace_ : number, only if GP_space or GP_inten is True
        the length scale of Gaussian Process prior of log(SNR)
    bGP_ : number, only if GP_space or GP_inten is True.
        the standard deviation of the GP prior
    lGPinten_: number, only if GP_inten is True
        the length scale in fMRI intensity of the GP prior of log(SNR)
    .. note::
    """

    def __init__(
            self, n_iter=50, rank=None, GP_space=False, GP_inten=False,
            tolx=2e-3, verbose=False, pad_DC=False, epsilon=0.001,
            space_smooth_range=None, inten_smooth_range=None,
            inten_weight=0.1):
        self.n_iter = n_iter
        self.rank = rank
        self.GP_space = GP_space
        self.GP_inten = GP_inten
        self.tolx = tolx
        self.verbose = verbose
        self.pad_DC = pad_DC
        self.epsilon = epsilon
        # This is a tiny ridge added to the Gaussian Process
        # covariance matrix template to gaurantee that it is invertible.
        # Mathematically it means we assume that this proportion of the
        # variance is always independent between voxels for the log(SNR2).
        self.space_smooth_range = space_smooth_range
        self.inten_smooth_range = inten_smooth_range
        self.inten_weight = inten_weight
        # The kernel of the Gaussian Process is a weighting of a kernel
        # defined on spatial coordinate and a kernel defined on
        # image intensity. We think it should be primarily based on
        # spatial coordinate. So the default weight for the kernel
        # defined on image intensity is 0.1.

        return

    def fit(self, X, design, scan_onsets=None, coords=None,
            inten=None, inten_weight=None, verbose=False):
        """Compute the Bayesian RSA
        Parameters
        ----------
        X : 2-D numpy array, shape=[time_points, voxels]
            In the equation, this is actually Y.
        design: 2-D numpy array, shape=[time_points, conditions]
            This is the design matrix. We will automatically pad a column
            of all one's if pad_DC is True
        scan_onsets: optional, an 1-D numpy array, shape=[runs,]
            this specifies the indices of X which correspond to the onset
            of each scanning run. If you do not provide,
            we assume all data are from the same run.
            This only makes a difference for the inverse
            of the temporal covariance matrix of noise
        coords: optional, 2-D numpy array, shape=[voxels,3]
            This is the coordinate of each voxel,
            used for implementing Gaussian Process prior.
        inten: optional, 1-D numpy array, shape=[voxel,]
            This is the average fMRI intensity in each voxel.
            It should be calculated from your data without any preprocessing
            such as z-scoring. Because it should reflect
            whether a voxel is bright (grey matter) or dark (white matter).
            A Gaussian Process kernel defined on both coordinate and intensity
            imposes a smoothness prior on adjcent voxels
            but with the same tissue type. The Gaussian Process
            is experimental and has shown good performance on
            some visual datasets.
        """

        if self.verbose:
            logger.debug('Running Bayesian RSA')

        assert not self.GP_inten or (self.GP_inten and self.GP_space),\
            'You must speficiy GP_space to True'\
            'if you want to use GP_inten'

        # Check input data
        assert_all_finite(X)
        assert X.ndim == 2, 'The data should be 2 dimension ndarray'

        assert np.all(np.std(X, axis=0) > 0),\
            'The time courses of some voxels do not change at all.'\
            ' Please make sure all voxels are within the brain'

        # check design matrix
        assert_all_finite(design)
        assert design.ndim == 2,\
            'The design matrix should be 2 dimension ndarray'
        assert np.all(np.std(design, axis=0) >= 0) or self.pad_DC,\
            'You already included DC component in the '\
            'design matrix. Please set pad_DC as False'
        assert np.size(design, axis=0) == np.size(X, axis=0),\
            'Design matrix and data do not '\
            'have the same number of time points.'
        if self.pad_DC and verbose:
            logger.debug('Padding one more column of 1 to '
                         'the end of design matrix.')
            design = np.concatenate((design,
                                     np.ones([design.shape[0], 1])), axis=1)
        assert self.rank is None or self.rank <= design.shape[1],\
            'Your design matrix has fewer columns than the rank you set'

        # check scan_onsets validity
        assert scan_onsets is None or\
            (np.max(scan_onsets) <= X.shape[0] and np.min(scan_onsets) >= 0),\
            'Some scan onsets provided are out of the range of time points.'

        # check the size of coords and inten
        if self.GP_space:
            if self.verbose:
                logger.debug('Fitting with Gaussian Process prior on log(SNR)')
            assert coords.shape[0] == X.shape[1],\
                'The voxel number of coords does not match that of data X'
            if self.GP_inten:
                assert inten.shape[0] == X.shape[1],\
                    'The voxel number of intensity does not '\
                    'match that of data X'
                if inten_weight is None:
                    inten_weight = self.inten_weight
                assert inten_weight >= 0 and inten_weight <= 1,\
                    'The weight of intensity kernel is '\
                    'out of the range of (0,1).'\
                    'I suggest using a small value, such as 0.1.'
            else:
                inten_weight = 0.0

        # Run Bayesian RSA
        if not self.GP_space:
            self.U_, self.L_, self.nSNR_, self.sigma_, self.rho_ = \
                self._fit_RSA_UV(design, X, scan_onsets=scan_onsets)
        elif not self.GP_inten or inten_weight == 0:
            self.U_, self.L_, self.nSNR_, self.sigma_, self.rho_,\
                self.lGPspace_, self.bGP_ = self._fit_RSA_UV(
                    design, X, scan_onsets=scan_onsets, coords=coords)
        else:
            self.U_, self.L_, self.nSNR_, self.sigma_, self.rho_, \
                self.lGPspace_, self.bGP_, self.lGPinten_ = \
                self._fit_RSA_UV(design, X, scan_onsets=scan_onsets,
                                 coords=coords, inten=inten,
                                 inten_weight=inten_weight)

        if self.pad_DC:
            self.U_ = self.U_[:-1, :-1]
            self.L_ = self.L_[:-1, :self.rank]
        sqrt_diag = np.diag(self.U_)**0.5
        self.C_ = self.U_ / np.tile(sqrt_diag, [np.size(sqrt_diag), 1])\
            / np.transpose(np.tile(sqrt_diag, [np.size(sqrt_diag), 1]))
        return self

    # The following 2 functions below generate templates used
    # for constructing inverse of covariance matrix of AR(1) noise

    def _D_gen(self, TR):
        if TR > 0:
            return np.diag(np.ones(TR - 1), -1) \
                + np.diag(np.ones(TR - 1), 1)
        else:
            return np.empty([0, 0])

    def _F_gen(self, TR):
        if TR > 0:
            F = np.eye(TR)
            F[0, 0] = 0
            F[TR - 1, TR - 1] = 0
            return F
        else:
            return np.empty([0, 0])

    def _prepare_data(self, design, Y, n_T, n_V, scan_onsets=None):
        """Prepares different forms of products of design matrix and data,
        or between themselves. These products are reused a lot during fitting.
        So we pre-calculate them. Because of the fact that these are reused,
        it is in principle possible to update the fitting as new data come in,
        by just incrementally adding the products of new data and
        their corresponding part of design matrix
        """
        if scan_onsets is None:
            # assume that all data are acquired within the same scan.
            D = np.diag(np.ones(n_T - 1), -1) + np.diag(np.ones(n_T - 1), 1)
            F = np.eye(n_T)
            F[0, 0] = 0
            F[n_T - 1, n_T - 1] = 0
        else:
            # Each value in the scan_onsets tells the index at which
            # a new scan starts. For example, if n_T = 500, and
            # scan_onsets = [0,100,200,400], this means that the time points
            # of 0-99 are from the first scan, 100-199 are from the second,
            # 200-399 are from the third and 400-499 are from the fourth
            run_TRs = np.diff(np.append(scan_onsets, n_T))
            logger.debug('I infer that the number of volumes'
                         ' in each scan are: {}'.format(run_TRs))

            D_ele = list(map(self._D_gen, run_TRs))
            F_ele = list(map(self._F_gen, run_TRs))
            D = []
            for d_ele in D_ele:
                D = scipy.linalg.block_diag(D, d_ele)
            F = []
            for f_ele in F_ele:
                F = scipy.linalg.block_diag(F, f_ele)
            # D and F above are templates for constructing
            # the inverse of temporal covariance matrix of noise

        XTY = np.dot(design.T, Y)
        XTDY = np.dot(np.dot(design.T, D), Y)
        XTFY = np.dot(np.dot(design.T, F), Y)

        YTY_diag = np.zeros([np.size(Y, axis=1)])
        YTDY_diag = np.zeros([np.size(Y, axis=1)])
        YTFY_diag = np.zeros([np.size(Y, axis=1)])
        for i_V in range(n_V):
            YTY_diag[i_V] = np.dot(Y[:, i_V].T, Y[:, i_V])
            YTDY_diag[i_V] = np.dot(np.dot(Y[:, i_V].T, D), Y[:, i_V])
            YTFY_diag[i_V] = np.dot(np.dot(Y[:, i_V].T, F), Y[:, i_V])

        XTX = np.dot(design.T, design)
        XTDX = np.dot(np.dot(design.T, D), design)
        XTFX = np.dot(np.dot(design.T, F), design)
        return XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX, XTDX, XTFX

    def _fit_RSA_UV(self, design, Y, param0_fitU=None, param0_fitV=None,
                    scan_onsets=None, coords=None, inten=None,
                    inten_weight=0.0):
        """ The major utility of fitting Bayesian RSA. In the current version,
            The initial parameters param0_fitU and param0_fitV are not provided.
            But we keep this option of providing initial parameters
            for debugging purpose.
        """
        GP_inten = self.GP_inten
        if inten_weight == 0:
            GP_inten = False 
        GP_space = self.GP_space
        rank = self.rank
        n_V = np.size(Y,axis=1)
        n_T = np.size(Y,axis=0)
        n_C = np.size(design,axis=1)
        l_idx = np.tril_indices(n_C)
        
        if self.verbose:
            t_start = time.time()
        
        if rank != None: # the rank of covariance matrix is specified
            idx_rank = np.where(l_idx[1]<rank)
            l_idx = (l_idx[0][idx_rank],l_idx[1][idx_rank])
        else:
            rank = n_C # if not specified, we assume the user wants to estimate a full rank matrix
            if self.verbose:
                print('Please be aware that you did not specify the rank of covariance matrix you want to estimate')
                print('I will assume that the covariance matrix shared among voxels is of full rank')
                print('Rank =',rank)
                print('Please be aware that estimating a matrix of high rank can be extremely slow')
                print('If you have a good reason to specify a lower rank than the number of experiment conditions, do so.')

        n_l = np.size(l_idx[0]) # the number of parameters for L    

        XTY,XTDY,XTFY,YTY_diag, YTDY_diag, YTFY_diag, XTX, XTDX, XTFX = self._prepare_data(design,Y,n_T,n_V,scan_onsets)

        
        if GP_inten != False and inten != None and np.var(inten)>0: 
            # We ought to make sure that there is indeed variation in image intensity
            if np.size(inten) != n_V:
                raise IndexError('The size of intensity vector does not match the number of voxels')
            inten_diff2 = np.zeros([n_V,n_V])
            inten_tile = np.tile(inten,[n_V,1])
            inten_diff2 = (inten_tile-inten_tile.T)**2
            GP_inten = True
            # set the hyperparameter for the GP process:
            if self.inten_smooth_range is None:
                inten_smooth_range = np.max(inten_diff2)**0.5/2.0
            else:
                inten_smooth_range = self.inten_smooth_range
        else:
            inten_diff2 = None
            GP_inten = False
            inten_smooth_range = None

        if GP_space != False and coords != None:
            GP_space = True


            if coords.ndim > 2:
                raise IndexError('The coordinate matrix should not have more than 2 dimensions')
            if coords.ndim == 2:
                if np.size(coords,1)!=n_V:
                    if np.size(coords,0)==n_V:
                        coords = coords.T # we want to make the shape of coordinates 3 x n_V
                    else:
                        raise IndexError('None of the dimensions of the coordinate matches the number of voxels')
            elif coords.ndim == 1:
                coords = np.expand_dims(coords,0)
            dist2 = np.zeros([n_V,n_V])

            for dim in range(np.size(coords,0)):
                coord_tile = np.tile(coords[dim,:],[n_V,1])
                dist2 = dist2 + (coord_tile-coord_tile.T)**2
            
            # set the hyperparameter for the GP process:
            if self.space_smooth_range is None:
                space_smooth_range = np.max(dist2)**0.5/2.0
            else:
                space_smooth_range = self.space_smooth_range
                
            if GP_inten:
                
                n_smooth = 2
            else:
                n_smooth = 1

            


            # This is the upper bound of the variance of log(s^2). Because it is on the log. This already is a very large bound
            current_GP = np.zeros(n_smooth)
        else:
            n_smooth = 0
            dist2 = None
            GP_space = False
            GP_inten = False
            space_smooth_range = None
            inten_smooth_range = None


        if (param0_fitU is None) or (param0_fitV is None):
            beta_hat = np.linalg.lstsq(design,Y)[0]
            residual = Y - np.dot(design,beta_hat)
            cov_point_est = np.cov(beta_hat)
            eye = np.eye(n_C)
            vec_I = eye[np.tril_indices(n_C)]
#             current_vec_U_chlsk_l_AR1 = np.linalg.cholesky(cov_point_est+np.eye(n_C)*1e-6)[l_idx]
            # We add a tiny diagonal element to the point estimation of covariance, just in case
            # the user provides data in which n_C is smaller than n_T
#             current_vec_U_chlsk_l_AR1 = eye[l_idx]
            # The initial value for the Cholesky factor does not seem to matter much
            # So I go with random initialization now.
            current_vec_U_chlsk_l_AR1 = np.random.randn(n_l)

            
            param0 = np.zeros(n_l+2)
            param0_fitU = np.zeros(n_l+n_V)
            param0_fitV = np.zeros(n_V+n_smooth-1)

            param0_fitU[0:n_l] = current_vec_U_chlsk_l_AR1
            param0_fitU[n_l:n_l+n_V] = np.sum(residual[0:-1,:]*residual[1:,:],axis=0)/\
                    np.sum(residual[0:-1,:]*residual[0:-1,:],axis=0) 
            log_sigma2_est = np.log(np.var(residual[1:,:]-residual[0:-1,:]*param0_fitU[n_l:n_l+n_V],axis=0))
            param0_fitU[n_l:n_l+n_V] = np.tan(param0_fitU[n_l:n_l+n_V]*np.pi/2)
            # up to this point, the GP parameters are not set.

            param0[0:n_l] = current_vec_U_chlsk_l_AR1
            param0[n_l] = np.median(log_sigma2_est)
            param0[n_l+1] = np.median(param0_fitU[n_l:n_l+n_V])

            res = scipy.optimize.minimize(self._loglike_y_U_AR1_singpara, param0,
                                      args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, XTY, XTDY, XTFY, l_idx,n_C,n_T,n_V,rank),
                                      method='BFGS',jac=True,tol = 1e-2, #
                                      options = {'xtol':1e-3, 'disp':self.verbose})


            
            current_a1 = np.ones(n_V)*res.x[n_l+1]
            current_logSigma2 = log_sigma2_est 
            # log_sigma2_est is obtained before the above fitting. Although it should overestimate
            # the variance, setting it this way would allow it to track more closely to log(sigma^2) for each voxel
            current_logSNR2 = -current_logSigma2
            norm_factor = np.mean(current_logSNR2)
            current_logSNR2 = current_logSNR2 - norm_factor
            current_vec_U_chlsk_l_AR1 = res.x[0:n_l] * np.exp(norm_factor/2.0)
            
            


            param0_fitU[0:n_l] = current_vec_U_chlsk_l_AR1
            param0_fitU[n_l:n_l+n_V] = current_a1 
            param0_fitV[0:n_V-1] = current_logSNR2[0:n_V-1]

        else:
            current_vec_U_chlsk_l_AR1 = param0_fitU[0:n_l]
            current_a1 = param0_fitU[n_l:n_l+n_V]
            current_logSNR2 = np.empty(n_V)
            current_logSNR2[0:n_V-1] = param0_fitV[0:n_V-1]
            current_logSNR2[-1] = -np.sum(current_logSNR2[0:n_V-1])
            if GP_space:
                current_GP = param0_fitV[n_V-1:n_V-1+n_smooth]
        
        param1_fitV = param0_fitV.copy()
        param1_fitU = param0_fitU.copy()
        
        tolx = self.tolx*5
        # if on average, each parameter's change is smaller than this threshold, we stop fitting
        
        # We cannot start fitting smoothness parameter just after fitting the single parameter version.
        # The reason is that we would have SNR equal in all voxels to start with, and that will favor a
        # smooth distance that is very large.
        # Therefore, we first fit one iteration without the smoothness parameter\
        if GP_space:
            init_iter = 40
            if self.verbose:
                print('initial fitting without GP prior')
            
            for it in range(0,init_iter):
                res_fitV = scipy.optimize.minimize(self._loglike_y_AR1_diagV_fitV, param0_fitV[:n_V-1],
                           args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                                 XTY, XTDY, XTFY, current_vec_U_chlsk_l_AR1, current_a1, \
                                 l_idx,n_C,n_T,n_V,rank,False,False),
                           method='BFGS',jac=True,tol = 10**-2, #
                           options = {'xtol':10**-3, 'disp':self.verbose,'maxiter':4}) 

                current_logSNR2[0:n_V-1] = res_fitV.x[0:n_V-1]
                current_logSNR2[-1] = - np.sum(current_logSNR2[0:n_V-1])
                param1_fitV[:n_V-1] = current_logSNR2[:n_V-1]
                
                norm_fitVchange = np.linalg.norm(param1_fitV[:n_V-1]-param0_fitV[:n_V-1])
                if self.verbose:
                    print("norm of parameter change after fitting V: ",norm_fitVchange)
                    print('E[log(SNR2)^2]:',np.mean(current_logSNR2**2))
                    # print('E[SNR^2]:',np.mean(np.exp(current_logSNR2/2)))
                


                
                

                

                

                
                
                
                
                if np.any(np.logical_not(np.isfinite(current_logSigma2))):
                    print(current_logSigma2)
                    print(current_logSNR2)
                    print('log(sigma^2) has non-finite number')
                
                param0_fitV = param1_fitV.copy()


                # fit U
                param0_fitU[0:n_l] = current_vec_U_chlsk_l_AR1
                param0_fitU[n_l:n_l+n_V] = current_a1
                res_fitU = scipy.optimize.minimize(self._loglike_y_AR1_diagV_fitU, param0_fitU,
                           args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                                 XTY, XTDY, XTFY, current_logSNR2,  \
                                 l_idx,n_C,n_T,n_V,rank),\
                           method='BFGS',jac=True,tol = 10**-2, \
                           options = {'xtol':10**-3, 'disp':self.verbose,'maxiter':3}) 
                current_vec_U_chlsk_l_AR1 = res_fitU.x[0:n_l]
                current_a1 = res_fitU.x[n_l:n_l+n_V]
                norm_fitUchange = np.linalg.norm(res_fitU.x-param0_fitU)
                if self.verbose:
                    print("norm of parameter change after fitting U: ",norm_fitUchange)
                param0_fitU = res_fitU.x.copy()

                if norm_fitVchange/np.sqrt(param0_fitV.size-n_smooth) < tolx and norm_fitUchange/np.sqrt(param0_fitU.size) < tolx:
                    break

            current_GP[0] = np.log(np.min(dist2[np.tril_indices_from(dist2, k=-1)]))
#             current_GP[0] = np.log(np.max(dist2)/4.0)
            # Initialize it with the smallest distance. 
            if GP_inten:
                current_GP[1] = np.log(np.maximum( np.percentile(inten_diff2[np.tril_indices_from(inten_diff2,k=-1)],5), 0.5))

            param0_fitV[n_V-1:n_V-1+n_smooth] = current_GP
            if self.verbose:
                print('current GP[0]:',current_GP[0])
        test_grad = self.verbose
        if test_grad:
            if self.verbose:
                print('testing gradient for fitV')
            ll0, deriv0 = self._loglike_y_AR1_diagV_fitV(param1_fitV, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, current_vec_U_chlsk_l_AR1, current_a1, l_idx,n_C,n_T,n_V,rank,GP_space,GP_inten,\
                dist2,inten_diff2,space_smooth_range,inten_smooth_range,inten_weight)
            perturb = 1e-4
            
            param_perturb = param1_fitV.copy()
            param_perturb[0] = param_perturb[0]+perturb
            ll1, deriv1 = self._loglike_y_AR1_diagV_fitV(param_perturb, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, current_vec_U_chlsk_l_AR1, current_a1, l_idx,n_C,n_T,n_V,rank,GP_space,GP_inten,\
                dist2,inten_diff2,space_smooth_range,inten_smooth_range,inten_weight)
            if self.verbose:
                print('perturbing log_SNR2')
                print('numerical difference:', ll1-ll0)
                print('analytical difference:', deriv0[0]*perturb)

            
            
            if GP_space:
                
                param_perturb = param1_fitV.copy()
                param_perturb[n_V-1] = param_perturb[n_V-1]+perturb
                ll1, deriv1 = self._loglike_y_AR1_diagV_fitV(param_perturb, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, \
                     YTFY_diag, XTY, XTDY, XTFY, current_vec_U_chlsk_l_AR1, current_a1,l_idx,n_C,n_T,n_V,\
                     rank,GP_space,GP_inten,dist2,inten_diff2,space_smooth_range,inten_smooth_range,inten_weight)
                if self.verbose:
                    print('perturbing GP[0]')
                    print('numerical difference:', ll1-ll0)
                    print('analytical difference:', deriv0[n_V-1]*perturb)

                
                
            if GP_inten:
                
                param_perturb = param1_fitV.copy()
                param_perturb[n_V] = param_perturb[n_V]+perturb
                ll1, deriv1 = self._loglike_y_AR1_diagV_fitV(param_perturb, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, \
                     YTFY_diag, XTY, XTDY, XTFY, current_vec_U_chlsk_l_AR1, current_a1,l_idx,n_C,n_T,n_V,\
                     rank,GP_space,GP_inten,dist2,inten_diff2,space_smooth_range,inten_smooth_range,inten_weight)
                if self.verbose:
                    print('perturbing GP[1]')
                    print('numerical difference:', ll1-ll0)
                    print('analytical difference:', deriv0[n_V]*perturb)

            if self.verbose:
                print('testing gradient for fitU')

            ll0, deriv0 = self._loglike_y_AR1_diagV_fitU(param1_fitU, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, current_logSNR2,  l_idx,n_C,n_T,n_V,rank)
            perturb = 1e-4

            
            param_perturb = param1_fitU.copy()
            param_perturb[0] = param_perturb[0]+perturb
            ll1, deriv1 = self._loglike_y_AR1_diagV_fitU(param_perturb, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, current_logSNR2, l_idx,n_C,n_T,n_V,rank)
            if self.verbose:
                print('perturbing L')
                print('numerical difference:', ll1-ll0)
                print('analytical difference:', deriv0[0]*perturb)

            
            param_perturb = param1_fitU.copy()
            param_perturb[n_l] = param_perturb[n_l]+perturb
            ll1, deriv1 = self._loglike_y_AR1_diagV_fitU(param_perturb, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, current_logSNR2, l_idx,n_C,n_T,n_V,rank)
            if self.verbose:
                print('perturbing a1')
                print('numerical difference:', ll1-ll0)
                print('analytical difference:', deriv0[n_l]*perturb)   

        tolx = self.tolx
        n_iter = self.n_iter
        if self.verbose:
            print('start real fitting')
        for it in range(0,n_iter):

            # fit V

            res_fitV = scipy.optimize.minimize(self._loglike_y_AR1_diagV_fitV, param0_fitV,
                                           args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                                                 XTY, XTDY, XTFY, current_vec_U_chlsk_l_AR1, current_a1, \
                                                 l_idx,n_C,n_T,n_V,rank,GP_space,GP_inten,dist2,inten_diff2,\
                                                 space_smooth_range,inten_smooth_range,inten_weight),\
                                           method='BFGS',jac=True,tol = 10**(-2-2/n_iter*(it+1)), \
                                           options = {'xtol':10**(-3-3/n_iter*(it+1)), 'disp':self.verbose,'maxiter':6}) 
            param1_fitV = res_fitV.x.copy()
            current_logSNR2[0:n_V-1] = res_fitV.x[0:n_V-1]
            current_logSNR2[n_V-1] = -np.sum(current_logSNR2[0:n_V-1])
            
            
            if GP_space:
                current_GP = res_fitV.x[n_V-1:n_V-1+n_smooth]
            
            fitVchange = param1_fitV[:n_V-1]-param0_fitV[:n_V-1]
            norm_fitVchange = np.linalg.norm(fitVchange)
            
            param0_fitV = param1_fitV.copy()
            if self.verbose:
                print("norm of parameter change after fitting V: ",norm_fitVchange)
                print('E[log(SNR2)^2]:',np.mean(current_logSNR2**2))


            # fit U
            
            
            
        
            param0_fitU[0:n_l] = current_vec_U_chlsk_l_AR1
            param0_fitU[n_l:n_l+n_V] = current_a1

            res_fitU = scipy.optimize.minimize(self._loglike_y_AR1_diagV_fitU, param0_fitU,
                                           args=(XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                                                 XTY, XTDY, XTFY, current_logSNR2,  \
                                                 l_idx,n_C,n_T,n_V,rank),\
                                           method='BFGS',jac=True,tol = 10**(-2-2/n_iter*(it+1)),
                                           options = {'xtol':10**(-2-3/n_iter*(it+1)), 'disp':self.verbose,'maxiter':6}) 
            current_vec_U_chlsk_l_AR1 = res_fitU.x[0:n_l]
            current_a1 = res_fitU.x[n_l:n_l+n_V]

            fitUchange = res_fitU.x-param0_fitU
            norm_fitUchange = np.linalg.norm(fitUchange)
            if self.verbose:
                print("norm of parameter change after fitting U: ",norm_fitUchange)
            param0_fitU = res_fitU.x.copy()

#             if norm_fitVchange/np.sqrt(param0_fitV.size-n_smooth) < tolx and norm_fitUchange/np.sqrt(param0_fitU.size) < tolx\
#                 and np.linalg.norm(fitUchange[0:n_l])/np.sqrt(n_l) < tolx:
            if np.max(np.abs(fitVchange)) < tolx and np.max(np.abs(fitUchange)) < tolx:
                break

            if GP_space:
                if self.verbose:
                    print('current GP[0]:',current_GP[0])
                    print('gradient for GP[0]:',res_fitV.jac[n_V-1])
                if GP_inten:
                    if self.verbose:
                        print('current GP[1]:',current_GP[1])
                        print('gradient for GP[1]:',res_fitV.jac[n_V])

        Uest_chlsk_l_AR1_UV = np.zeros([n_C,rank])
        Uest_chlsk_l_AR1_UV[l_idx] = current_vec_U_chlsk_l_AR1

        est_cov_AR1_UV = np.dot(Uest_chlsk_l_AR1_UV,Uest_chlsk_l_AR1_UV.T)

        
        opt_rho1_AR1_UV = 2/np.pi * np.arctan(current_a1)
        opt_SNR_AR1_UV = np.exp(current_logSNR2/2.0)
        if GP_space:
            opt_space_smooth_r = np.exp(current_GP[0]/2.0)
            if GP_inten:
                opt_intensity_kernel_r = np.exp(current_GP[1]/2.0)
        
        
        if self.verbose:
            t_finish = time.time()
            print('total time of fitting: ',t_finish-t_start,' seconds')

        # Calculating opt_sigma_AR1_UV
        YTAY = YTY_diag - opt_rho1_AR1_UV*YTDY_diag + opt_rho1_AR1_UV**2 * YTFY_diag
        XTAX = XTX[np.newaxis,:,:] - opt_rho1_AR1_UV[:,np.newaxis,np.newaxis]*XTDX[np.newaxis,:,:] \
            + opt_rho1_AR1_UV[:,np.newaxis,np.newaxis]**2 * XTFX[np.newaxis,:,:]
        # dimension: space*feature*feature
        XTAY = XTY - opt_rho1_AR1_UV*XTDY + opt_rho1_AR1_UV**2 * XTFY
        # dimension: feature*space
        LTXTAY = np.dot(Uest_chlsk_l_AR1_UV.T,XTAY)
        # dimension: rank*space

        LAMBDA_i = np.zeros([n_V,rank,rank])
        for i_v in range(n_V):
            LAMBDA_i[i_v,:,:] = np.dot(np.dot(Uest_chlsk_l_AR1_UV.T,XTAX[i_v,:,:]),Uest_chlsk_l_AR1_UV) * \
                opt_SNR_AR1_UV[i_v]**2
        LAMBDA_i += np.eye(rank)
        # dimension: space*rank*rank
        LAMBDA = np.linalg.inv(LAMBDA_i)
        # dimension: space*rank*rank
        YTAXL_LAMBDA = np.einsum('ijk,ki->ij',LAMBDA,LTXTAY)
        # dimension: space*rank
        YTAXL_LAMBDA_LT = np.dot(YTAXL_LAMBDA,Uest_chlsk_l_AR1_UV.T)
        # dimension: space*feature

        opt_sigma_AR1_UV = ((YTAY - opt_SNR_AR1_UV**2 * np.sum(YTAXL_LAMBDA_LT * XTAY.T, axis=1)) / n_T) **0.5
        
        
        
        if GP_inten:
            K_major = np.exp(- (dist2/opt_space_smooth_r**2 + inten_diff2/opt_intensity_kernel_r**2)/2.0) 
        elif GP_space:
            K_major = np.exp(- dist2/opt_space_smooth_r**2/2.0) 
        if GP_space:
            K = K_major + np.diag(np.ones(n_V)*self.epsilon)
            opt_std_log_SNR2 = (np.dot(current_logSNR2,np.dot(np.linalg.inv(K),current_logSNR2)) / (n_V-1))**0.5
        
        
        if not GP_space:
            return est_cov_AR1_UV, Uest_chlsk_l_AR1_UV, opt_SNR_AR1_UV, opt_sigma_AR1_UV, opt_rho1_AR1_UV
        else:
            if not GP_inten:
                return est_cov_AR1_UV, Uest_chlsk_l_AR1_UV, opt_SNR_AR1_UV, opt_sigma_AR1_UV, opt_rho1_AR1_UV, opt_space_smooth_r, opt_std_log_SNR2
            else:
                return est_cov_AR1_UV, Uest_chlsk_l_AR1_UV, opt_SNR_AR1_UV, opt_sigma_AR1_UV, opt_rho1_AR1_UV, opt_space_smooth_r, opt_std_log_SNR2, opt_intensity_kernel_r
            
            
            
    
    # We fit two parts of the parameters iteratively.
    # The following are the corresponding negative log likelihood functions.        
            
    def _loglike_y_AR1_diagV_fitU(self, param, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, XTY, XTDY, XTFY, log_SNR2,  l_idx,n_C,n_T,n_V,rank): 
    #This code calculates the log likelihood of data given cholesky decomposition of U, noise level, and AR(1) parameters,
    #The log of the square of signal to noise level in each voxel (the ratio of the diagonal elements in V and 
    #the noise variance) are fixed. This likelihood is iteratively optimized with the one with _fitV.
    
    #In this version, we assume that beta is independent between voxels and noise is also independent
    #By the assumption that noise is independent, we only need to pass the products X'X, X'Y and Y'Y, instead of X and Y
    #Y'Y is passed in the form of its diagonal elements. 
    #DiagV means we assume that the variance of beta can be different between voxels. This means that V is a diagonal
    #matrix instead of an identity matrix. The parameter includes the lower triangular part of the cholesky decomposition
    #of U flattened, then tan(rho1*pi/2) where rho1 is 
    #each voxel's autoregressive coefficient (assumging AR(1) model). Such parametrization avoids the need of boundaries
    #for parameters.

        LL = 0.0 # log likelihood

        n_l = np.size(l_idx[0]) # the number of parameters in the index of lower-triangular matrix
        # we replace all (n_C+1)*n_C/2 with n_l. This change allows for parametrizing only
        # part of the lower triangular matrix (non-full rank covariance matrix)


        L = np.zeros([n_C,rank]) #lower triagular matrix L, cholesky decomposition of U
        L[l_idx] = param[0:n_l]

        
#         sigma2 = np.exp(log_sigma2) #variance of noise (sigma^2)
        a1 = param[n_l:n_l+n_V]
        rho1 = 2.0/np.pi * np.arctan(a1) #auto-regressive coefficients



        SNR2 = np.exp(log_SNR2)
        # each element of SNR2 is the ratio of the diagonal element on V to the variance of the fresh noise in each voxel 

        #derivatives
        deriv_L = np.zeros(np.shape(L))
        deriv_a1 = np.zeros(np.shape(rho1))

        YTAY = YTY_diag - rho1*YTDY_diag + rho1**2 * YTFY_diag
        XTAX = XTX[np.newaxis,:,:] - rho1[:,np.newaxis,np.newaxis]*XTDX[np.newaxis,:,:] \
            + rho1[:,np.newaxis,np.newaxis]**2 * XTFX[np.newaxis,:,:]
        # dimension: space*feature*feature
        XTAY = XTY - rho1*XTDY + rho1**2 * XTFY
        # dimension: feature*space
        LTXTAY = np.dot(L.T,XTAY)
        # dimension: rank*space 
        LAMBDA_i = np.zeros([n_V,rank,rank])
        for i_v in range(n_V):
            LAMBDA_i[i_v,:,:] = np.eye(rank) + np.dot(np.dot(L.T,XTAX[i_v,:,:]),L) * SNR2[i_v]
        # dimension: space*rank*rank
        LAMBDA = np.linalg.inv(LAMBDA_i)
        # dimension: space*rank*rank
        YTAXL_LAMBDA = np.einsum('ijk,ki->ij',LAMBDA,LTXTAY)
        # dimension: space*rank
        YTAXL_LAMBDA_LT = np.dot(YTAXL_LAMBDA,L.T)
        # dimension: space*feature (feature can be larger than rank)
        
        sigma2 = (YTAY - SNR2 * np.sum(YTAXL_LAMBDA_LT * XTAY.T, axis=1)) / n_T
        
        
        # dimension: space,
        LL = -np.sum(np.log(sigma2))*n_T*0.5 + np.sum(np.log(1-rho1**2))*0.5 -np.sum(np.log(np.linalg.det(LAMBDA_i)))*0.5 \
            - n_T/2.0 # - np.sum(np.log(1+a1**2)) + np.sum(log_sigma2)/2.0 # + np.sum(log_SNR2)/0.2
        XTAXL = np.dot(XTAX,L)
        # dimension: space*feature*rank
        deriv_L = -np.einsum('ijk,ikl,i',XTAXL,LAMBDA,SNR2) - np.einsum('ijk,ik,il,i',XTAXL,YTAXL_LAMBDA,YTAXL_LAMBDA,SNR2**2/sigma2) \
            + np.dot(XTAY/sigma2*SNR2,YTAXL_LAMBDA)
        # dimension: feature*rank
        tmp = -XTDX+2*rho1[:,np.newaxis,np.newaxis]*XTFX
        # dimension: space*feature*feature
        deriv_a1 = 2.0/(np.pi*(1+a1**2)) * (-rho1/(1-rho1**2)  \
            - np.einsum('...ij,...ji',np.dot(LAMBDA,L.T),np.dot(tmp,L)) *SNR2/2.0 \
            + np.sum((-XTDY+2.0*rho1*XTFY)*YTAXL_LAMBDA_LT.T,axis=0)/sigma2*SNR2 \
            - np.einsum('...i,...ij,...j',YTAXL_LAMBDA_LT,tmp,YTAXL_LAMBDA_LT) \
            /sigma2/2.0*(SNR2**2.0) -(-YTDY_diag + 2.0*rho1*YTFY_diag) / (sigma2*2.0) ) # - 2.0*a1/(1+a1**2)
        # dimension: space,


        deriv = np.zeros(np.size(param))
        deriv[0:n_l] = deriv_L[l_idx]
        deriv[n_l:n_l+n_V] = deriv_a1


        return -LL, -deriv        
    
    
    
    
    
    
    
    
    def _loglike_y_AR1_diagV_fitV(self,param, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, XTY, XTDY, XTFY, L_l, a1,\
                              l_idx,n_C,n_T,n_V,rank,GP_space,GP_inten,\
                              dist2=None,inten_dist2=None,space_smooth_range=None,inten_smooth_range=None,inten_weight=0.0): 
        
        #This code calculates the log likelihood of data given the log of the square of signal to noise ratio in each voxel.
        #This likelihood is iteratively optimized with the one with _fitU. 
        #The cholesky factor of U and autoregressive coefficient in AR(1) parameters are fixed. 
        #Because the ML estimate of the variance of noise in each voxel (sigma^2) given other parameters has analytic form,
        #we do not need to explicitly parametrize it. Just set it to the ML value.
        

        #We assume that beta is independent between voxels and noise is also independent
        #By the assumption that noise is independent, we only need to pass the products X'X, X'Y and Y'Y, instead of X and Y
        #Y'Y is passed in the form of its diagonal elements. 
        #DiagV means we assume that the beta amplitude can be different between voxels. This means that V is a diagonal
        #matrix instead of an identity matrix. The parameter includes the log of the variance of noise
        #(the independent fresh noise) and log of the square of pseudo-SNR. (log(s^2)). s^2 is the ratio of the diagonal elements
        #of V over sigma^2. Such parametrization avoids the need of boundaries
        #for parameters.
        #L_l is the lower triangular part of L, a1 is tan(rho1*pi/2), where rho1 is the autoregressive coefficient in each voxel
        # we now add GP prior to log(s2). But I think we only need to do this to fitV version, because log(s2)
        # are fixed in fitU.
        
        scale_SNR = 20.0 
        scale_log_SNR2 =10.0
        # When GP prior is not imposed, we still want to impose weakly informative prior on log(SNR^2).
        # This avoids divergence since we now impose the restriction of sum(log_SNR2)=0 within this function
        # Additional prior can be imposed on SNR to further avoid divergence. From my experimenting, 
        # the divergence only occurs when I parametrize n_V-1 numbers of log_SNR2 instead of n_V numbers of log_SNR2.
        # The prior we imposed is half-cauchy (for SNR), which is flatter than Gaussian at 0.
        
        
        LL = 0.0
        n_l = np.size(l_idx[0]) # the number of parameters in the index of lower-triangular matrix
        # we replace all (n_C+1)*n_C/2 with n_l. This change allows for parametrizing only
        # part of the lower triangular matrix (non-full rank covariance matrix)
        if rank == None:
            rank = int((2*n_C+1 - np.sqrt(n_C**2*4+n_C*4+1-8*n_l))/2)

        L = np.zeros([n_C,rank])
        L[l_idx] = L_l

        log_SNR2 = np.empty(n_V)
        log_SNR2[0:n_V-1] = param[0:n_V-1]
        log_SNR2[-1] = -np.sum(log_SNR2[0:n_V-1])
        # instead of correcting log_SNR2 to be zero-mean after fitting, we put the restriction here
        # The cons: we impose correlation between these free parameters, which can lead to divergence in fitting
        
        SNR2 = np.exp(log_SNR2)
        SNR = np.exp(log_SNR2/2.0)
        # We will need SNR (s) itself when imposing smoothness prior
        rho1 = 2.0/np.pi * np.arctan(a1)

        YTAY = YTY_diag - rho1*YTDY_diag + rho1**2 * YTFY_diag
        XTAX = XTX[np.newaxis,:,:] - rho1[:,np.newaxis,np.newaxis]*XTDX[np.newaxis,:,:] \
            + rho1[:,np.newaxis,np.newaxis]**2 * XTFX[np.newaxis,:,:]
        # dimension: space*feature*feature
        XTAY = XTY - rho1*XTDY + rho1**2 * XTFY
        # dimension: feature*space
        LTXTAY = np.dot(L.T,XTAY)
        
        LAMBDA_i = np.zeros([n_V,rank,rank])
        for i_v in range(n_V):
            LAMBDA_i[i_v,:,:] = np.dot(np.dot(L.T,XTAX[i_v,:,:]),L) * SNR2[i_v]
        LAMBDA_i += np.eye(rank) 
        # dimension: space*rank*rank
        LAMBDA = np.linalg.inv(LAMBDA_i)
        # dimension: space*rank*rank
        YTAXL_LAMBDA = np.einsum('ijk,ki->ij',LAMBDA,LTXTAY)
        # dimension: space*rank
        YTAXL_LAMBDA_LT = np.dot(YTAXL_LAMBDA,L.T)
    
        sigma2 = (YTAY - SNR2 * np.sum(YTAXL_LAMBDA_LT * XTAY.T, axis=1)) / n_T
        
        
#         if np.sum(log_SNR2!=0) == 0:
#             # all SNR2 become 1. No need to incorporate GP prior.
#             GP_space = False
        
        # Flags for whethe to incorportate GP smoothness prior in space or tissue type
        if GP_space :
            b_space = param[n_V-1]
            l2_space = np.exp(b_space)
            dl2_db_space = l2_space
            # partial derivative


            if GP_inten:
                b_inten = param[n_V]
                l2_inten = np.exp(b_inten)
                dl2_db_inten = l2_inten



        deriv_log_SNR2 = np.zeros(np.shape(SNR2))
        

        if GP_inten:
            K_major = np.exp(- (dist2/l2_space * (1-inten_weight) + inten_dist2/l2_inten * inten_weight)/2.0) 
#             K_major = np.exp(- (dist2*l_2_space + inten_dist2*l_2_inten)/2.0) # *tau2
        elif GP_space:
            K_major = np.exp(- dist2/l2_space/2.0)
#             K_major = np.exp(- dist2*l_2_space/2.0) # *tau2

        if GP_space:
            
            K_tilde = K_major + np.diag(np.ones(n_V)*self.epsilon) # *tau2
            # Note that the K_tilder here is the correct K divided by the variance tau^2
            # So it does not depend on the variance of the GP.
            L_K = np.linalg.cholesky(K_tilde)
#             inv_L_K = np.linalg.inv(L_K)
            inv_L_K = np.linalg.solve(L_K,np.identity(n_V))
            inv_K_tilde = np.dot(inv_L_K.T,inv_L_K)
            log_det_K_tilde = np.sum(np.log(np.diag(L_K)**2))

            invK_tilde_log_SNR2 = np.dot(inv_K_tilde,log_SNR2)
            log_SNR2_invK_tilde_log_SNR2 = np.dot(log_SNR2,invK_tilde_log_SNR2)

        
        # dimension: space,
        LL = -np.sum(np.log(sigma2))*n_T*0.5 + np.sum(np.log(1-rho1**2))*0.5 -np.sum(np.log(np.linalg.det(LAMBDA_i)))*0.5 \
            -n_T*0.5
        

        # dimension: space*feature*rank
        deriv_log_SNR2 = (-rank + np.trace(LAMBDA,axis1=1,axis2=2))*0.5 + YTAY / (sigma2*2.0) - n_T*0.5 \
            - np.einsum('ij,ijk,ik->i',YTAXL_LAMBDA_LT,XTAX,YTAXL_LAMBDA_LT) / (sigma2*2.0) * (SNR2**2.0) \
            
        # dimension: space,



        if GP_space:

            tau2 = log_SNR2_invK_tilde_log_SNR2 / n_V
#             LL = LL - log_SNR2_invK_tilde_log_SNR2/2.0/tau2 - log_det_K_tilde/2.0 -n_V/2.0*np.log(tau2) - np.log(2*np.pi)*n_V/2.0
            LL = LL - 0.5 - log_det_K_tilde/2.0 -n_V/2.0*np.log(tau2) - np.log(2*np.pi)*n_V/2.0
            # If we parametrize tau^2 with its log, then we do not need to take log again above.
            
            deriv_log_SNR2 = deriv_log_SNR2 - invK_tilde_log_SNR2/tau2

            
            
            
            
            dK_tilde_dl2_space = dist2*(K_major)/2.0/l2_space * (1-inten_weight)


            deriv_b_space =  0.5 * np.dot(np.dot(invK_tilde_log_SNR2.T,dK_tilde_dl2_space),invK_tilde_log_SNR2)*dl2_db_space/tau2 \
                - 0.5*np.sum(inv_K_tilde*dK_tilde_dl2_space)*dl2_db_space 

                
            if inten_weight != 1:
                LL += scipy.stats.halfcauchy.logpdf(l2_space**0.5,scale=space_smooth_range)
                deriv_b_space -= 2*l2_space**0.5/(l2_space + space_smooth_range**2) * dl2_db_space 
            
            
            if GP_inten:
                
                dK_tilde_dl2_inten = inten_dist2*(K_major)/2.0/l2_inten * inten_weight


                deriv_b_inten =  0.5 * np.dot(np.dot(invK_tilde_log_SNR2.T,dK_tilde_dl2_inten),invK_tilde_log_SNR2)*dl2_db_inten/tau2 \
                    - 0.5*np.sum(inv_K_tilde*dK_tilde_dl2_inten)*dl2_db_inten 
                if inten_weight != 0:
                    LL += scipy.stats.halfcauchy.logpdf(l2_inten**0.5,scale=inten_smooth_range)
                    deriv_b_inten -= 2*l2_inten**0.5/(l2_inten + inten_smooth_range**2) * dl2_db_inten 

                
                # deriv_b_inten = 0.5 * np.dot(np.dot(invK_tilde_SNR,dK_tilde_dl_2_inten),invK_tilde_SNR)*dl_2_db_inten/tau2 \
                #     - 0.5*np.sum(inv_K_tilde*dK_tilde_dl_2_inten)*dl_2_db_inten - 2.0*b_inten/(1+b_inten**2) \
                #     -2.0/l_2_inten*dl_2_db_inten    
#         else:
#             LL += np.sum(scipy.stats.norm.logpdf(log_SNR2,scale=scale_log_SNR2))
#             deriv_log_SNR2 -= log_SNR2 / (scale_log_SNR2**2)
#             tau2 = np.sum(log_SNR2**2) / n_V
#             LL = LL - 0.5  -n_V/2.0*np.log(tau2) - np.log(2*np.pi)*n_V/2.0
#             deriv_log_SNR2 -= log_SNR2/tau2
            # Prior to control the magnitude of log(SNR^2)
#             LL += np.sum(scipy.stats.halfcauchy.logpdf(SNR,scale=scale_SNR))
#             deriv_log_SNR2 -= SNR*2 / (SNR2+scale_SNR**2) 
        
        
        deriv = np.zeros(np.size(param))
        deriv[0:n_V-1] = deriv_log_SNR2[0:n_V-1] - deriv_log_SNR2[n_V-1]
        if GP_space:
            deriv[n_V-1] = deriv_b_space
            if GP_inten:
                deriv[n_V] = deriv_b_inten
    


        return -LL, -deriv
    
    
    
    
    
    def _loglike_y_U_AR1_singpara(self,param, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, XTY, XTDY, XTFY, l_idx,n_C,n_T,n_V,rank): 
        #In this version, we assume that beta is independent between voxels and noise is also independent
        #By the assumption that noise is independent, we only need to pass the products X'X, X'Y and Y'Y, instead of X and Y
        #We can even just pass the diagonal of Y'Y. Different from loglike_y_U_sigma_ind, this function treats sigma as part 
        #of parameters
        # singpara version uses single parameter of sigma^2 and rho1 to all voxels. We want to see if this reduces the bias in L
        LL = 0.0

        n_l = np.size(l_idx[0]) # the number of parameters in the index of lower-triangular matrix
        
        if rank == None:
            rank = int((2*n_C+1 - np.sqrt(n_C**2*4+n_C*4+1-8*n_l))/2)

        L = np.zeros([n_C,rank])
        L[l_idx] = param[0:n_l]
        
        log_sigma2 = param[n_l]
        sigma2 = np.exp(log_sigma2)
        a1 = param[n_l+1]
        rho1 = 2.0/np.pi * np.arctan(a1)



        XTAX = XTX - rho1*XTDX + rho1**2 * XTFX
        LAMBDA_i = np.eye(rank) + np.dot(np.dot(np.transpose(L),XTAX),L) / sigma2


        XTAY = XTY - rho1*XTDY + rho1**2 * XTFY
        LTXTAY = np.dot(L.T, XTAY)

        YTAY = YTY_diag - rho1*YTDY_diag + rho1**2 * YTFY_diag

        LAMBDA_LTXTAY = np.linalg.solve(LAMBDA_i,LTXTAY)
        L_LAMBDA_LTXTAY = np.dot(L, LAMBDA_LTXTAY)


        LL = LL + np.sum(LTXTAY * LAMBDA_LTXTAY) / (sigma2**2 * 2.0) \
            - np.sum(YTAY) / (sigma2 * 2.0)



        deriv_L = np.dot(XTAY,LAMBDA_LTXTAY.T) / sigma2**2 \
            - np.dot(np.dot(XTAX, L_LAMBDA_LTXTAY),LAMBDA_LTXTAY.T) / sigma2**3 

        deriv_log_sigma2 = np.sum(YTAY) / (sigma2 * 2.0) - np.sum(XTAY * L_LAMBDA_LTXTAY) / (sigma2**2) \
            + np.sum(np.dot(XTAX, L_LAMBDA_LTXTAY) * L_LAMBDA_LTXTAY) / (sigma2**3 * 2.0)
         

        deriv_a1 = 2.0/(np.pi*(1+a1**2)) * (- rho1 /(1 - rho1**2) \
            + np.sum((-XTDY+2*rho1*XTFY) * L_LAMBDA_LTXTAY)/(sigma2**2) \
            - np.sum(np.dot((-XTDX+2*rho1*XTFX), L_LAMBDA_LTXTAY) * L_LAMBDA_LTXTAY) / (sigma2**3 * 2.0) \
            - np.sum(-YTDY_diag + 2*rho1*YTFY_diag) / (sigma2*2.0)) 




        LL = LL + np.size(YTY_diag) * (- log_sigma2 * n_T *0.5 + np.log(1 - rho1**2)*0.5 - np.log(np.linalg.det(LAMBDA_i)) * 0.5)

        deriv_L = deriv_L - np.transpose(np.linalg.solve(LAMBDA_i, np.dot(L.T, XTAX))) / sigma2 * np.size(YTY_diag)
        deriv_log_sigma2 = deriv_log_sigma2 + (rank - n_T - np.trace(np.linalg.inv(LAMBDA_i))) * 0.5 * np.size(YTY_diag)
        deriv_a1 = deriv_a1 - np.trace(np.linalg.solve(LAMBDA_i,np.dot(np.dot(L.T, (-XTDX + 2*rho1*XTFX)),L))) / (sigma2*2) * np.size(YTY_diag)

    
        deriv = np.zeros(np.size(param))
        deriv[0:n_l] = deriv_L[l_idx]
        deriv[n_l] = deriv_log_sigma2
        deriv[n_l+1] = deriv_a1



        return -LL, -deriv
    
    
    
    
    
    
    