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
        epoch_map = []
        for i in range(18):
            epoch_map.append((i, 4, 15))
            epoch_map.append((i, 24, 35))
            epoch_map.append((i, 44, 55))
            epoch_map.append((i, 64, 75))
            epoch_map.append((i, 84, 95))
            epoch_map.append((i, 104, 115))
            epoch_map.append((i, 124, 135))
            epoch_map.append((i, 144, 155))
            epoch_map.append((i, 164, 175))
            epoch_map.append((i, 184, 195))
            epoch_map.append((i, 204, 215))
            epoch_map.append((i, 224, 235))
        raw_data=separateEpochs(activity_data, epoch_map)
        labels = [i%2 for i in range(216)]
    raw_data = comm.bcast(raw_data, root=0)
    labels = comm.bcast(labels, root=0)
    vs = VoxelSelector(raw_data, 12, labels, 18)
    results = vs.run()
    if rank==0:
        print(results[0:100])
