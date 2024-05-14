import numpy as np
import pickle, torch
from . import tools
import random
from . import augmentations

class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, split='train',temperal_padding_ratio=6,p_interval=0.9,window_size=64, mmap=True):
        self.data_path = data_path
        self.label_path = label_path
        self.p_interval=p_interval
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.window_size=window_size
        
        self.split = split
        self.load_data()
    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            print('===============')
            print(len(self.data))
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
             
        # load data
       # if mmap:
         #   self.data = np.load(self.data_path, mmap_mode='r')
        #else:
        #    self.data = np.load(self.data_path)
        #N,T,_= self.data.shape
       # self.data=self.data.reshape((N,T,2,25,3)).transpose(0,4,1,3,2)
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        valid=np.sum(data_numpy.sum(0).sum(-1).sum(-1)!=0)
        data_numpy=augmentations.crop_subsequence(data_numpy,valid,self.p_interval,self.window_size)
        # processing
        #data = self._aug(data_numpy)
        return data_numpy, label, index


    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        return data_numpy


class Feeder_dual(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path, shear_amplitude, temperal_padding_ratio, l_ratio,p_interval=0.9,window_size=64, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.l_ratio = l_ratio
        self.p_interval=p_interval
        self.window_size=window_size
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
             self.label = np.load(self.label_path)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        N,T,_=self.data.shape
        self.data=self.data.reshape((N,T,2,25,3)).transpose(0,4,1,3,2)
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        valid_numpy=np.sum(data_numpy.sum(0).sum(-1).sum(-1)!=0)
       # data_numpy=tools.valid_crop_resize(data_numpy,valid_numpy,self.p_interval,self.window_size)
        # processing
        data_numpy_v1 = augmentations.temporal_cropresize(data_numpy, valid_numpy, self.p_interval,self.window_size)
        data_numpy_v2 = augmentations.temporal_cropresize(data_numpy,valid_numpy, self.p_interval,self.window_size)
        data1 = self._augs(data_numpy_v1)
        data2 = self._augs(data_numpy_v2)

        return [data1, data2,data_numpy_v1], label

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


