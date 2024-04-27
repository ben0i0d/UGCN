import os
import numpy as np
import multiprocessing
from numpy.lib.format import open_memmap

sets = {
    'train', 'val'
}

datasets = {
    'xsub', 'xview',
}

def merge_joint_bone_data(dataset, set):
    print(dataset, set)
    data_jpt = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set), mode='r')
    data_bone = open_memmap('./data/{}/{}_bone.npy'.format(dataset, set), mode='r')
    N, C, T, V, M = data_jpt.shape
    data_jpt_bone = open_memmap('./data/{}/{}_joint_bone.npy'.format(dataset, set), dtype='float32', mode='w+', shape=(N, 6, T, V, M))
    data_jpt_bone[:, :C, :, :, :] = data_jpt
    data_jpt_bone[:, C:, :, :, :] = data_bone

processes = []

for dataset in datasets:
    for set in sets:
        process = multiprocessing.Process(target=merge_joint_bone_data, args=(dataset, set))
        processes.append(process)
        process.start()

for process in processes:
    process.join()
