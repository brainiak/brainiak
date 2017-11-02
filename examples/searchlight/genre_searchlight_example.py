# The following code is designed to perform a searchlight at every voxel in the brain looking at the difference in pattern similarity between musical genres (i.e. classical and jazz). In the study where the data was obtained, subjects were required to listen to a set of 16 songs twice (two runs) in an fMRI scanner. The 16 songs consisted of 8 jazz songs and 8 classical songs. The goal of this searchlight is to find voxels that seem to represent distinct information about these different musical genres. Presumably, these voxels would be found in the auditory cortex which happens to be the most organized system in the brain for processing sound information. 

import numpy as np
import time
from mpi4py import MPI
from nilearn.image import load_img
import sys
from brainiak.searchlight.searchlight import Searchlight
from scipy import stats
from scipy.sparse import random
import os

# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Generate random data
if rank == 0:
    np.random.seed(0)
    data1_rand = np.random.rand(91,109,91,16)
    data2_rand = np.random.rand(91,109,91,16)
    classical = np.random.rand(2600)
    jazz = np.random.rand(2600)
    d1_reshape = np.reshape(data1_rand,(91*109*91,16))
    d2_reshape = np.reshape(data2_rand,(91*109*91,16))
    a1 = load_img('a1plus_2mm.nii.gz')
    a1_vec = np.reshape(a1.get_data(),(91*109*91))
    a1_idx = np.nonzero(a1_vec)
    for i in range(8):
        d1_reshape[a1_idx[0],i] += classical
        d1_reshape[a1_idx[0],i+8] += jazz
        d2_reshape[a1_idx[0],i] += classical
        d2_reshape[a1_idx[0],i+8] += jazz
    data1 = np.reshape(d1_reshape,(91,109,91,16))
    data2 = np.reshape(d2_reshape,(91,109,91,16))

    # Flatten data, then zscore data, then reshape data back into MNI coordinate space
    data1 = stats.zscore(np.reshape(data1,(91*109*91,16)))
    data1 = np.reshape(data1,(91,109,91,16))
    data2 = stats.zscore(np.reshape(data2,(91*109*91,16)))
    data2 = np.reshape(data2,(91,109,91,16))
else:
    data1 = None
    data2 = None

# Load mask 
mask_img = load_img('MNI152_T1_2mm_brain_mask.nii')
mask_img = mask_img.get_data()

# Definte function that takes the difference between within vs. between genre comparisons
def corr2_coeff(AB,msk,myrad,bcast_var):
    if not np.all(msk):
        return None
    A,B = (AB[0], AB[1])
    A = A.reshape((-1,A.shape[-1]))
    B = B.reshape((-1,B.shape[-1]))
    corrAB = np.corrcoef(A.T,B.T)[16:,:16]
    classical_within = np.mean(corrAB[0:8,0:8])
    jazz_within = np.mean(corrAB[8:16,8:16])
    classJazz_between = np.mean(corrAB[8:16,0:8])
    jazzClass_between = np.mean(corrAB[0:8,8:16])
    within_genre = np.mean([classical_within,jazz_within])
    between_genre = np.mean([classJazz_between,jazzClass_between])
    diff = within_genre - between_genre
    return diff

comm.Barrier()
begin_time = time.time()
comm.Barrier()

# Create and run searchlight
sl = Searchlight(sl_rad=1,max_blk_edge=5)
sl.distribute([data1,data2],mask_img)
sl.broadcast(None)
global_outputs = sl.run_searchlight(corr2_coeff)

comm.Barrier()
end_time = time.time()
comm.Barrier()

# Plot searchlight results
if rank == 0:
    print('Searchlight Done: ', end_time - begin_time)
    maxval = np.max(global_outputs[np.not_equal(global_outputs,None)])
    minval = np.min(global_outputs[np.not_equal(global_outputs,None)])
    global_outputs = np.array(global_outputs, dtype=np.float)
    print(global_outputs)

    # Save searchlight images
    out_dir = "searchlight_images"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    import matplotlib.pyplot as plt
    for (cnt, img) in enumerate(global_outputs):
        plt.imshow(img,vmin=minval,vmax=maxval)
        plt.colorbar()
        plt.savefig('searchlight_images/' + 'img' + str(cnt) + '.png')
        plt.clf()


