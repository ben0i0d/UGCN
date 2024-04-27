import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from numpy.lib.format import open_memmap

paris = {
    'xview': (
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
        (12, 0), 
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (20, 20),(22, 7), (23, 24), 
        (24, 11)
    ),
    'xsub': (
        (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
        (12, 0), 
        (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (20, 20),(22, 7), (23, 24), 
        (24, 11)
    )
}

sets = {
    'train', 'val'
}

datasets = {
    'xsub', 'xview'
}

# bone
def gen_bone(dataset, set):
    print(dataset, set)
    data = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = np.zeros((N, 3, T, V, M), dtype='float32')
    for v1, v2 in tqdm(paris[dataset]):
        fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]
    np.save('./data/{}/{}_bone.npy'.format(dataset, set), fp_sp)

processes = []

for dataset in datasets:
    for set in sets:
        process = multiprocessing.Process(target=gen_bone, args=(dataset, set))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()