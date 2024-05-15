import numpy as np
import torch
from . import tools
import random
from . import augmentations

class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, shear_amplitude=0.5, split='train',temperal_padding_ratio=6,p_interval=0.9,window_size=64,stream='joint'):
        self.data_path = data_path
        self.p_interval=p_interval
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.window_size=window_size
        self.split = split
        self.stream = stream
        self.load_data()
    def load_data(self):
        npz_data = np.load(self.data_path,mmap_mode='r')
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise NotImplementedError('data split only supports train/test')
        # data: N C V T M
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        valid=np.sum(data_numpy.sum(0).sum(-1).sum(-1)!=0)
        data_numpy=augmentations.crop_subsequence(data_numpy,valid,self.p_interval,self.window_size)
        # processing
        
        if self.stream == 'joint':
            pass
        elif self.stream == 'motion':
            motion = torch.zeros_like(data_numpy)
            motion[:, :, :-1, :, :] = data_numpy[:, :, 1:, :, :] - data_numpy[:, :, :-1, :, :]
            data_numpy = motion
        elif self.stream == 'bone':
            Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                    (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
            bone = torch.zeros_like(data_numpy)

            for v1, v2 in Bone:
                bone[:, :, :, v1 - 1, :] = data_numpy[:, :, :, v1 - 1, :] - data_numpy[:, :, :, v2 - 1, :]
            data_numpy = bone
        else:
            raise ValueError
        return data_numpy, label, index


    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        return data_numpy


class Feeder_dual(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, shear_amplitude, temperal_padding_ratio, l_ratio,p_interval=0.9,window_size=64,split='train',stream='joint'):
        self.data_path = data_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.l_ratio = l_ratio
        self.p_interval=p_interval
        self.window_size=window_size
        self.split=split
        self.stream=stream
        self.load_data()

    def load_data(self):
        npz_data = np.load(self.data_path,mmap_mode='r')
        # load data & label
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise NotImplementedError('data split only supports train/test')
        # reshape to (N, 3, 300, 25, 2)
        N,T,_=self.data.shape
        self.data=self.data.reshape((N,T,2,25,3)).transpose(0,4,1,3,2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        valid_numpy=np.sum(data_numpy.sum(0).sum(-1).sum(-1)!=0)

        # processing
        data_numpy_v1 = augmentations.temporal_cropresize(data_numpy, valid_numpy, self.p_interval,self.window_size)
        data_numpy_v2 = augmentations.temporal_cropresize(data_numpy,valid_numpy, self.p_interval,self.window_size)
        data1 = self._augs(data_numpy_v1)
        data2 = self._augs(data_numpy_v2)

        if self.stream == 'joint':
            pass
        elif self.stream == 'motion':
            motion1 = torch.zeros_like(data1)
            motion2 = torch.zeros_like(data2)

            motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
            motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]

            data1 = motion1
            data2 = motion2
        elif self.stream == 'bone':
            Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                    (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

            bone1 = torch.zeros_like(data1)
            bone2 = torch.zeros_like(data2)

            for v1, v2 in Bone:
                bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]

            data1 = bone1
            data2 = bone2
        else:
            raise ValueError

        return [data1, data2], label

    def _aug(self, data_numpy):
        data_numpy=tools.random_rotate(data_numpy)
        
        if self.temperal_padding_ratio > 0:
           
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            flip_prob  = random.random()
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
            if flip_prob < 0.5:
                 data_numpy=data_numpy 
            else:
                 data_numpy = tools.courruption(data_numpy)

        return data_numpy
    
    def _augs(self, data_numpy):
        if random.random() < 0.5:
            data_numpy = augmentations.Rotate(data_numpy)
        if random.random() < 0.5:
            data_numpy = augmentations.Flip(data_numpy)
        if random.random() < 0.5:
            data_numpy = augmentations.Shear(data_numpy)
        if random.random() < 0.5:
            data_numpy = augmentations.spatial_masking(data_numpy)
        if random.random() < 0.5:
            data_numpy = augmentations.temporal_masking(data_numpy)
        return data_numpy


