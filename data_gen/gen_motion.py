import os
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

sets = {
    'train', 'val'
}

datasets = {
    'xsub', 'xview',
}

parts = {
    'joint', 'bone'
}


def gen_motion(dataset, set,part):
    print(dataset, set, part)
    data = open_memmap('./data/{}/{}_{}.npy'.format(dataset, set, part),mode='r')
    N, C, T, V, M = data.shape
    fp_sp = np.zeros((N, 3, T, V, M), dtype='float32')
    for t in tqdm(range(T - 1)):
        fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]
    fp_sp[:, :, T - 1, :, :] = 0
    np.save('./data/{}/{}_{}_motion.npy'.format(dataset, set, part), fp_sp)

for dataset in datasets:
    for set in sets:
        for part in parts:
            gen_motion(dataset, set, part)
