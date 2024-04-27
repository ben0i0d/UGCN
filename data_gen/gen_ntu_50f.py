import os
import sys
import pickle

import torch
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from NTUDatasets import NTUMotionProcessor

from tqdm import tqdm

max_body = 2
num_joint = 25
max_frame = 50
batch_size = 64

def gendata(dataset_path, out_path, benchmark, part='eval'):
    dataset = NTUMotionProcessor(
        '{}/{}_joint.npy'.format(os.path.join(dataset_path, benchmark), part),
        '{}/{}_label.pkl'.format(os.path.join(dataset_path, benchmark), part),
        data_type='relative',
        t_length=max_frame,
        y_rotation=True,
        sampling='resize',
        displacement=1,
        mmap=True)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=False)

    f_position = open_memmap('{}/{}_position.npy'.format(out_path, part),dtype='float32',mode='w+',shape=(dataset.N, 3, max_frame, num_joint, max_body))

    f_motion = open_memmap('{}/{}_motion.npy'.format(out_path, part),dtype='float32',mode='w+',shape=(dataset.N, 3, max_frame, num_joint, max_body))

    f_label = open_memmap('{}/{}_label.npy'.format(out_path, part),dtype='int64',mode='w+',shape=(dataset.N, 1))

    index = 0
    for i, (data, motion, label) in tqdm(enumerate(data_loader)):
        length = label.shape[0]
        if i * batch_size != index:
            print(i, index)
        f_position[index:(index+length), :, :, :, :] = data.numpy()
        f_motion[index:(index+length), :, :, :, :] = motion.numpy()
        f_label[index:(index+length), :] = label.numpy().reshape(-1, 1)
        index += length

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--dataset_path', default='data/')
    parser.add_argument('--out_folder', default='data/f50')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(arg.dataset_path, out_path, benchmark=b, part=p)