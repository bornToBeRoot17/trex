# -*- encoding: utf-8 -*-

from functools import partial
from multiprocessing.pool import ThreadPool
import multiprocessing

from scipy.ndimage import convolve
import numpy as np
import cv2
import os

dataset = '/nobackup/ppginf/rgcastro/research/dataset2/'
classes = ['ft', 'lu', 'map', 'allreduce']
kernel_dims = [32, 64, 128, 256, 512]
data_dims = [32, 64, 128, 256, 512]

def read_imgs(imgs, files, idx):
    file = files[idx]

    tm = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    tm = cv2.blur(tm,(1,1))

    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            aux = float(tm[i][j]/255)
            if (aux < 0.7):
                tm[i][j] = 1
            else:
                tm[i][j] = 0
    if (np.sum(tm) > tm.shape[0]*tm.shape[1]/2):
        tm = 1 - tm #np.invert(tm)

    #for i in range(tm.shape[0]):
    #    for j in range(tm.shape[1]):
    #        print(tm[i][j], end=' ')
    #        print()
    #    print()
    ##exit(-1)

    imgs[idx] = tm

def get_files(dim, kernel_data):
    idx_img = []
    files   = []
    labels  = []

    if (kernel_data == 'kernel'):
        for i in range(len(classes)):
            files.append(dataset + classes[i]+'_' + str(dim) + '/fig1.png')
            idx_img.append(i)
    else:
        count = 0
        for i in range(len(classes)):
            dir = dataset + classes[i] + '_' + str(dim) + '/'
            for r, d, f in os.walk(dir):
                for file in f:
                    files.append(dir+file)
                    idx_img.append(count)
                    labels.append(classes[i])
                    count += 1

    data = [[] for i in range(len(files))]

    #print('Reading kernels...')
    n_threads = multiprocessing.cpu_count()
    pool = ThreadPool(n_threads)
    result = pool.map(partial(read_imgs, data, files), idx_img)
    pool.close()
    pool.join()

    return data, labels

def convolution(kernels, data, pred, idx):
    x = data[idx]

    w, h  = x.shape[0], x.shape[1]

    kernel_idx = 0
    kernel_values = []
    for kernel in kernels:
        sum = 0
        for i in range(0,w,kernel.shape[0]):
            for j in range(0,h,kernel.shape[1]):
                slice = x[i:i+kernel.shape[0],j:j+kernel.shape[1]]
                conv  = slice * kernel
                sum += np.sum(conv)
            kernel_values.append(sum)

        max_idx = 0
        for i in range(1,len(kernel_values)):
            if (kernel_values[i] > kernel_values[max_idx]):
                max_idx = i

        pred[idx] = classes[max_idx]

def main():
    for k_dim in kernel_dims:
        kernels, _ = get_files(k_dim, 'kernel')
        for d_dim in data_dims:
            if (d_dim < k_dim): continue

            data, label = get_files(d_dim, 'data')

            #for i in range(len(data)):
            #    for j in range(data[i].shape[0]):
            #        for k in range(data[i].shape[1]):
            #            print(data[i][j][k],end=' ')
            #        print()
            #    print()
            #exit(-1)

            idx_data = []
            for i in range(len(data)):
                idx_data.append(i)

            pred = [0 for i in range(len(data))]
            n_threads = multiprocessing.cpu_count()
            pool = ThreadPool(n_threads)
            result = pool.map(partial(convolution, kernels, data, pred), idx_data)
            pool.close()
            pool.join()

            count = 0
            for i in range(len(pred)):
                if (pred[i] == label[i]):
                    count += 1

            print('Kernel dim:', k_dim, '\tData dim:', d_dim, '\tAccuracy:', float(count/len(pred)))

    return 0

if (__name__ == '__main__'):
    main()
