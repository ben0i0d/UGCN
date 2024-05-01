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
            if np.all(data[i][0][j]==0):
                index = j
        if index <= 64:
            data[i] = source[i, :, :64, :, :]
        else:
            width = (index + 1) // 64
            for k in range(64):
                data[i,:,k,:,:] = np.divide(data[i,:,k*width:k+width,:,:].sum(axis=1),width)
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