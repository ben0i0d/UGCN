import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from numpy.lib.format import open_memmap

sets = {
    'train', 'val'
}

datasets = {
    'xsub', 'xview'
}


def gen_64f(dataset, set):
    print(dataset, set)
    source = open_memmap('./data/{}/{}_joint.npy'.format(dataset, set),mode='r')
    data = open_memmap('./data/f64/{}/{}_joint.npy'.format(dataset, set),mode='w+',shape=(source.shape[0],3,64,25,2),dtype='float32')
    
    for i in tqdm(range(source.shape[0])):
        index = 0
        for j in range(300):
            if index !=0:
                break
            if np.all(source[i][0][j]==0):
                index = j + 1
        if index <= 64:
            data[i] = source[i, :, :64, :, :]
        else:
            sep = 64 - index%64
            width = index // 64
            for k in range(sep):
                data[i,:,k,:,:] = np.divide(source[i,:,k*width:(k+1)*width,:,:].sum(axis=1),width)
            width = width + 1
            for k in range(sep,64):
                data[i,:,k,:,:] = np.divide(source[i,:,k*width-sep:(k+1)*width-sep,:,:].sum(axis=1),width)
        index =0      
                
processes = []

if not os.path.exists('./data/f64/'):
        os.mkdir('./data/f64/')

for dataset in datasets:
    if not os.path.exists('./data/f64/{}/'.format(dataset)):
        os.mkdir('./data/f64/{}/'.format(dataset))
    for set in sets:
        process = multiprocessing.Process(target=gen_64f, args=(dataset, set))
        processes.append(process)
        process.start()

for process in processes:
    process.join()