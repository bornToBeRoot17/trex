# -*- encoding: utf-8 -*-

from scipy.ndimage import convolve
import numpy as np
import cv2
import os

dataset = '/nobackup/ppginf/rgcastro/research/dataset2/'
classes = ['ft', 'lu', 'map', 'allreduce']
kernel_dims = [32, 64, 128, 256, 512, 1024]
data_dims = [32, 64, 128, 256, 512, 1024]

def read_x_data(kernel_dim, data_dim):
    kernel_files = []
    for c in classes:
        kernel_files.append(dataset+c+'_'+str(kernel_dim)+'/fig1.png')

    y_data = []
    data_files = []
    for c in classes:
        dir = dataset+c+'_'+str(data_dim)+'/'
        for r, d, f in os.walk(dir):
            #f = f[:int(len(f)/2)]
            f = f[:10]
            for file in f:
                data_files.append(dir+file)
                y_data.append(c)

    #print('Reading data...')
    x_data = []
    for data_file in data_files:
        tm = cv2.imread(data_file, cv2.IMREAD_GRAYSCALE).astype(np.int32)
        #tm = cv2.blur(tm,(5,5))
        for i in range(tm.shape[0]):
            for j in range(tm.shape[1]):
                aux = float(tm[i][j]/255)
                if (aux < 0.7):
                    tm[i][j] = 1
                else:
                    tm[i][j] = 0
                #print(tm[i][j], end=' ')
            #print()
        #print()
        if (np.sum(tm) > tm.shape[0]*tm.shape[1]/2):
            tm = 1 - tm #np.invert(tm)
        #for i in range(tm.shape[0]):
        #    for j in range(tm.shape[1]):
        #        print(tm[i][j], end=' ')
        #    print()
        #print()
        x_data.append(tm)
    #exit(-1)

    #print('Reading kernels...')
    kernels = []
    for kernel_file in kernel_files:
        tm = cv2.imread(kernel_file, cv2.IMREAD_GRAYSCALE).astype(np.int32)
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
            #for j in range(tm.shape[1]):
                #print(tm[i][j], end=' ')
            #print()
        #print()
        #exit(-1)
        kernels.append(tm)
    #exit(-1)

    return kernels, x_data, y_data

def main():
    for k_dim in kernel_dims:
        for d_dim in data_dims:
            if (d_dim < k_dim): continue

            kernels, x_data, y_data = read_x_data(k_dim, d_dim)

            y_pred = []
            for data in x_data:
                w, h  = data.shape[0], data.shape[1]

                kernel_idx = 0
                kernel_values = []
                for kernel in kernels:
                    sum = 0
                    for i in range(0,w,kernel.shape[0]):
                        for j in range(0,h,kernel.shape[1]):
                            slice = data[i:i+kernel.shape[0],j:j+kernel.shape[1]]
                            conv  = slice * kernel
                            sum += np.sum(conv)
                    kernel_values.append(sum)
                idx_result = np.where(kernel_values == np.amax(kernel_values))
                y_pred.append(classes[idx_result[0][0]])

            count = 0
            for i in range(len(y_pred)):
                if (y_pred[i] == y_data[i]):
                    count += 1

            print('Kernel dim:', k_dim, '\tData dim:', d_dim, '\tAccuracy:', float(count/len(y_pred)))

    return 0

if (__name__ == '__main__'):
    main()
