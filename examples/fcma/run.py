from brainiak.fcma.fcma import *
import sys

# mpirun -np 2 python run.py /Users/yidawang/data/face_scene/raw nii.gz /Users/yidawang/data/face_scene/mask.nii.gz data/fs_epoch_labels.npy
if __name__ == '__main__':
    data_dir = sys.argv[1]
    extension = sys.argv[2]
    mask_file = sys.argv[3]
    epoch_file = sys.argv[4]
    raw_data, labels = prepareData(data_dir, extension, mask_file, epoch_file)
    epochs_per_subj = 12
    num_subjs = 18
    vs = VoxelSelector(raw_data, epochs_per_subj, labels, num_subjs)
    results = vs.run()
    if MPI.COMM_WORLD.Get_rank()==0:
        print(results[0:100])
