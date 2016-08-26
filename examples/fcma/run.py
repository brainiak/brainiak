from brainiak.fcma.fcma import *
from mpi4py import MPI
import sys

# mpirun -np 2 python run.py /Users/yidawang/data/face_scene/raw nii.gz /Users/yidawang/data/face_scene/mask.nii.gz
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sys.stdout.flush()
    raw_data = []
    labels = []
    if rank==0:
        activity_data = readActivityData(sys.argv[1], sys.argv[2], sys.argv[3])
        epoch_map = np.load('data/fs_epoch_labels.npy') # a list of numpy array in shape (condition, nEpochs, nTRs)
        raw_data, labels=separateEpochs(activity_data, epoch_map)
    raw_data = comm.bcast(raw_data, root=0)
    labels = comm.bcast(labels, root=0)
    vs = VoxelSelector(raw_data, 12, labels, 18)
    results = vs.run()
    if rank==0:
        print(results[0:100])
