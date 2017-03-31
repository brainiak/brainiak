import numpy as np
from nilearn.image import load_img
import sys
from brainiak.brainiak.searchlight.searchlight import Searchlight

# Take subject ID as input
subj = sys.argv[1]

datadir = '/Users/jamalw/Desktop/PNI/music_event_structures/'

# Load functional data and mask data
data1 = load_img(datadir + 'subjects/' + subj + '/avg_reorder1.nii')
data2 = load_img(datadir + 'subjects/' + subj + '/avg_reorder2.nii')
mask_img = load_img(datadir + 'MNI152_T1_2mm_brain_mask.nii')
data1 = data1.get_data()
data2 = data2.get_data()
mask_img = mask_img.get_data()

# Flatten data, then zscore data, then reshape data back into MNI coordinate space
data1 = stats.zscore(np.reshape(data1,(91*109*91,16)))
data1 = np.reshape(data1,(91,109,91,16))
data2 = stats.zscore(np.reshape(data2,(91*109*91,16)))
data2 = np.reshape(data2,(91,109,91,16))

# Definte function that takes the difference between within vs. between genre comparisons
def corr2_coeff(AB,msk,myrad,bcast_var):
    if not np.all(msk):
        return None
    A,B = (AB[0], AB[1])
    A = A.reshape((-1,A.shape[-1]))
    B = B.reshape((-1,B.shape[-1]))
    corrAB = np.corrcoef(A.T,B.T)[16:,:16]
    classical_within  = np.mean(corrAB[0:8,0:8])
    jazz_within       = np.mean(corrAB[8:16,8:16])
    classJazz_between = np.mean(corrAB[8:16,0:8])
    jazzClass_between = np.mean(corrAB[0:8,8:16])
    within_genre =  np.mean([classical_within,jazz_within])
    between_genre = np.mean([classJazz_between,jazzClass_between])
    diff = within_genre - between_genre
    return diff

# Create and run searchlight
sl = Searchlight(sl_rad=1,max_blk_edge=5)
sl.distribute([data1,data2],mask_img)
sl.broadcast(None)
print('Running Searchlight...')
global_outputs = sl.run_searchlight(corr2_coeff)

# Plot searchlight results
maxval = np.max(global_outputs[np.not_equal(global_outputs,None)])
minval = np.min(global_outputs[np.not_equal(global_outputs,None)])
global_outputs = np.array(global_outputs, dtype=np.float)
print(global_outputs)
import matplotlib.pyplot as plt
for (cnt, img) in enumerate(global_outputs):
  plt.imshow(img,vmin=minval,vmax=maxval)
  plt.colorbar()
  plt.savefig(datadir + 'searchlight_images/' + 'img' + str(cnt) + '.png')
  plt.clf()


