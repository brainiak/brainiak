import numpy as np
import pandas as pd
import nibabel as nib
import nilearn as nl
from nilearn.input_data import NiftiMasker
import nilearn.plotting as niplot
import matplotlib as mlab
import matplotlib.pyplot as plt
import holoviews as hv
import timecorr as tc
import os
from brainiak.factoranalysis.htfa import HTFA
import warnings
import seaborn as sns

hv.extension('bokeh')
hv.output(size=200)

def opts(debug=False):
    '''
    Return a dictionary of parameters to pass to
    brainiak.factoranalysis.htfa.HTFA
    
    inputs:
      debug: set to True (default) to generate a quick test fit and False to
             generate a (slower) more accurate fit.
    '''
    if debug:
        return {'K': 10,
                'max_global_iter': 3,
                'max_local_iter': 3,
                'voxel_ratio': 0.1,
                'tr_ratio': 0.1,
                'max_voxel_scale': 0.1,
                'max_tr_scale': 0.1,
                'verbose': True}
    else:
        return {'K': 50,
                'max_global_iter': 10,
                'max_local_iter': 5,
                'voxel_ratio': 0.1,
                'tr_ratio': 0.1,
                'max_voxel_scale': 0.25,
                'max_tr_scale': 0.25,
                'verbose': True}

def opts2str(params):
    '''
    convert params to filename
    '''
    return str(params).replace('{', '').replace('}', '').replace("'", '').replace(': ', '-').replace(', ', '_')

def htfa2dict(htfa):
    '''
    turn htfa object into a pickleable dictionary
    '''
    return {'K': htfa.K, # init params
         'n_subj': htfa.n_subj,
         'max_global_iter': htfa.max_global_iter,
         'max_local_iter': htfa.max_local_iter,
         'threshold': htfa.threshold,
         'nlss_method': htfa.nlss_method,
         'nlss_loss': htfa.nlss_loss,
         'jac': htfa.jac,
         'x_scale': htfa.x_scale,
         'tr_solver': htfa.tr_solver,
         'weight_method': htfa.weight_method,
         'upper_ratio': htfa.upper_ratio,
         'lower_ratio': htfa.lower_ratio,
         'tr_ratio': htfa.tr_ratio,
         'voxel_ratio': htfa.voxel_ratio,
         'max_voxel': htfa.max_voxel,
         'max_tr': htfa.max_tr,
         'verbose': htfa.verbose,
         'prior_bcast_size': htfa.prior_bcast_size,
         'prior_size': htfa.prior_size,
         'local_posterior_': htfa.local_posterior_, # inferred params
         'local_weights_': htfa.local_weights_,
         'global_centers_cov_': htfa.global_centers_cov,
         'global_centers_cov_scaled': htfa.global_centers_cov_scaled,
         'global_posterior_': htfa.global_posterior_,
         'global_prior_': htfa.global_prior_,
         'global_widths_var': htfa.global_widths_var,
         'global_widths_var_scaled': htfa.global_widths_var_scaled,
         'map_offset': htfa.map_offset,
         'n_dim': htfa.n_dim
         }


def dict2htfa(htfa_dict):
    htfa = HTFA(K=htfa_dict['K'],
                n_subj=htfa_dict['n_subj'],
                max_voxel=htfa_dict['max_voxel'],
                max_tr=htfa_dict['max_tr'],
                verbose=False)
    for k in htfa_dict.keys():
        setattr(htfa, k, htfa_dict[k])
    return htfa

def global_params(htfa):
    centers = htfa.get_centers(htfa.global_posterior_)
    widths = htfa.get_widths(htfa.global_posterior_)
    return centers, widths

def local_params(htfa, n_timepoints):
    centers = [htfa.get_centers(x) for x in np.array_split(htfa.local_posterior_, htfa.n_subj)]
    widths = [htfa.get_widths(x) for x in np.array_split(htfa.local_posterior_, htfa.n_subj)]
        
    inds = np.hstack([0, np.cumsum(np.multiply(htfa.K, n_timepoints))])
    weights = [htfa.local_weights_[inds[i]:inds[i+1]].reshape([htfa.K, 
                                                               n_timepoints[i]]).T for i in np.arange(htfa.n_subj)]
    return centers, widths, weights

def plot_nodes(htfa, n_timepoints, cmap='Spectral', global_scale=100, local_scale=25):
    colors = np.repeat(np.vstack([[0, 0, 0], sns.color_palette(cmap, htfa.n_subj)]), htfa.K, axis=0)
    colors = [colors[i, :] for i in range(colors.shape[0])] # make colors into a list
    
    global_centers, global_widths = global_params(htfa)
    local_centers, local_widths, _ = local_params(htfa, n_timepoints)
    
    centers = np.vstack([global_centers, np.vstack(local_centers)])
    widths = np.vstack([global_widths, np.vstack(local_widths)])
    widths /= np.max(widths)
    
    scales = np.repeat(np.hstack([np.array(global_scale), np.array(htfa.n_subj*[local_scale])]), htfa.K)
    sizes = widths.T * scales
    return nl.plotting.plot_connectome(np.eye(htfa.K * (1 + htfa.n_subj)), centers, node_color=colors, node_size=sizes)
    

def nii2cmu(nifti_file, mask_file=None):
    '''
    inputs:
      nifti_file: a filename of a .nii or .nii.gz file to be converted into
                  CMU format
                  
      mask_file: a filename of a .nii or .nii.gz file to be used as a mask; all
                 zero-valued voxels in the mask will be ignored in the CMU-
                 formatted output.  If ignored or set to None, no voxels will
                 be masked out.
    
    outputs:
      Y: a number-of-timepoints by number-of-voxels numpy array containing the
         image data.  Each row of Y is an fMRI volume in the original nifti
         file.
      
      R: a number-of-voxels by 3 numpy array containing the voxel locations.
         Row indices of R match the column indices in Y.
    '''
    def fullfact(dims):
        '''
        Replicates MATLAB's fullfact function (behaves the same way)
        '''
        vals = np.asmatrix(range(1, dims[0] + 1)).T
        if len(dims) == 1:
            return vals
        else:
            aftervals = np.asmatrix(fullfact(dims[1:]))
            inds = np.asmatrix(np.zeros((np.prod(dims), len(dims))))
            row = 0
            for i in range(aftervals.shape[0]):
                inds[row:(row + len(vals)), 0] = vals
                inds[row:(row + len(vals)), 1:] = np.tile(aftervals[i, :], (len(vals), 1))
                row += len(vals)
            return inds
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        img = nib.load(nifti_file)
        mask = NiftiMasker(mask_strategy='background')
        if mask_file is None:
            mask.fit(nifti_file)
        else:
            mask.fit(mask_file)
    
    hdr = img.header
    S = img.get_sform()
    vox_size = hdr.get_zooms()
    im_size = img.shape
    
    if len(img.shape) > 3:
        N = img.shape[3]
    else:
        N = 1
    
    Y = np.float32(mask.transform(nifti_file)).copy()
    vmask = np.nonzero(np.array(np.reshape(mask.mask_img_.dataobj, (1, np.prod(mask.mask_img_.shape)), order='C')))[1]
    vox_coords = fullfact(img.shape[0:3])[vmask, ::-1]-1
    
    R = np.array(np.dot(vox_coords, S[0:3, 0:3])) + S[:3, 3]
    
    return {'Y': Y, 'R': R}

def cmu2nii(Y, R, template=None):
    '''
    inputs:
      Y: a number-of-timepoints by number-of-voxels numpy array containing the
         image data.  Each row of Y is an fMRI volume in the original nifti
         file.
      
      R: a number-of-voxels by 3 numpy array containing the voxel locations.
         Row indices of R match the column indices in Y.
      
      template: a filename of a .nii or .nii.gz file to be used as an image
                template.  Header information of the outputted nifti images will
                be read from the header file.  If this argument is ignored or
                set to None, header information will be inferred based on the
                R array.
    
    outputs:
      nifti_file: a filename of a .nii or .nii.gz file to be converted into
                  CMU format
                  
      mask_file: a filename for a .nii or .nii.gz file to be used as a mask; all
                 zero-valued voxels in the mask will be ignored in the CMU-
                 formatted output
    
    outputs:
      img: a nibabel Nifti1Image object containing the fMRI data
    '''
    Y = np.array(Y, ndmin=2)
    img = nib.load(template)
    S = img.affine
    locs = np.array(np.dot(R - S[:3, 3], np.linalg.inv(S[0:3, 0:3])), dtype='int')
    
    data = np.zeros(tuple(list(img.shape)[0:3]+[Y.shape[0]]))
    
    # loop over data and locations to fill in activations
    for i in range(Y.shape[0]):
        for j in range(R.shape[0]):
            data[locs[j, 0], locs[j, 1], locs[j, 2], i] = Y[i, j]
    
    return nib.Nifti1Image(data, affine=img.affine)

def animate_connectome(nodes, connectomes, cthresh='75%', figdir='frames', force_refresh=False): #move to helpers
    '''
    inputs:
      nodes: a K by 3 array of node center locations
      
      connectomes: a T by [((K^2 - K)/2) + K] array of per-timepoint connectomes.
                   Each timepoint's connectime is represented in a vectorized
                   format *including the diagonal* (i.e., self connections).
      
      figdir: where to save temporary files and final output
      
      force_refresh: if True, overwrite existing temporary files.  If False,
                     re-use existing temporary files and generate only the
                     temporary files that do not yet exist.
    
    outputs:
      ani: a matplotlib FuncAnimation object
    '''
    
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    
    #save a jpg file for each frame (this takes a while, so don't re-do already made images)
    def get_frame(t, fname):
        if force_refresh or not os.path.exists(fname):
            nl.plotting.plot_connectome(tc.vec2mat(connectomes[t, :]),
                                        nodes,
                                        node_color='k',
                                        edge_threshold=cthresh,
                                        output_file=fname)
    
    timepoints = np.arange(connectomes.shape[0])
    fnames = [os.path.join(figdir, str(t) + '.jpg') for t in timepoints]
    tmp = [get_frame(t, f) for t, f in zip(timepoints, fnames)]
    
    #create a movie frame from each of the images we just made
    fig = plt.figure()
    
    def get_im(fname):
        #print(fname)
        plt.axis('off')
        return plt.imshow(plt.imread(fname), animated=True)
    
    ani = mlab.animation.FuncAnimation(fig, get_im, fnames, interval=50)
    return ani


def mat2chord(connectome, cthresh=0.05):
    '''
    inputs:
      connectome: K by K connectivity matrix
      
      cthresh: only show connections in the top (cthresh*100)%; default = 0.25
    
    outputs:
      chord: a holoviews.Chord object displaying the connectome as a chord
             diagram
    '''
    
    def mat2links(x, ids):
        links = []
        for i in range(x.shape[0]):
            for j in range(i):
                links.append({'source': ids[i], 'target': ids[j], 'value': np.abs(x[i, j]), 'sign': np.sign(x[i, j])})
        return pd.DataFrame(links)
    
    K = connectome.shape[0]
    nodes = pd.DataFrame({'ID': range(K), 'Name': [f'Node {i}' for i in range(K)]})
    
    links = mat2links(connectome, nodes['ID'])
    chord = hv.Chord((links, hv.Dataset(nodes, 'ID'))).select(value=(cthresh, None))
    chord.opts(
        hv.opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=hv.dim('source').str(), labels='Name', node_color=hv.dim('ID').str())
    )
    return chord

def animate_chord(x, cthresh=0.05):
    '''
    inputs:
      connectomes: a T by [((K^2 - K)/2) + K] array of per-timepoint connectomes.
                   Each timepoint's connectime is represented in a vectorized
                   format *including the diagonal* (i.e., self connections).
      
      cthresh: only show connections in the top (cthresh*100)%; default = 0.25
    
    outputs:
      hmap: a holoviews.HoloMap object containing an interactive animation
    '''
    warnings.simplefilter('ignore') #suppress BokehUserWarning for node colors
    
    hv.output(max_frames=x.shape[0])
    renderer = hv.renderer('bokeh')
    return hv.HoloMap({t: mat2chord(tc.vec2mat(x[t, :]), cthresh=cthresh) for t in range(x.shape[0])}, kdims='Time (TRs)')