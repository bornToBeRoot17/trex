# -*- encoding: utf-8 -*-


from sys import argv, exit
import os, numpy as np
from fishervector import FisherVectorGMM
import multiprocessing
from acquiring import *

dataset_path = "/home/ppginf/rgcastro/dataset_svg/"
dim_train = [128]
dim_test = [256]

dirs = ["bifu","multc","cyl2d","cylinder3d"]

def main(argv):
    X_train = []
    y_train = []
    X_test  = []
    y_test  = []
    for dir in os.listdir(dataset_path):
        if (not os.path.isdir(dataset_path+dir)): continue

        if (str(dim_train[0]) in dir):
            for img in os.listdir(dataset_path+dir):
                X_train.append(np.asarray(image_parser(dataset_path+dir+'/'+img, dim_train, [128])))
                y_train.append(dir.split('_')[0])
        elif (str(dim_test[0]) in dir):
            for img in os.listdir(dataset_path+dir):
                X_test.append(np.asarray(image_parser(dataset_path+dir+'/'+img, dim_test, [256])))
                y_test.append(dir.split('_')[0])
    X_train = np.asarray(X_train)
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[2],X_train.shape[3],X_train.shape[1]))
    y_train = np.asarray(y_train)
    X_test  = np.asarray(X_test)
    X_test  = X_test.reshape((X_test.shape[0],X_test.shape[2],X_test.shape[3],X_test.shape[1]))
    y_test  = np.asarray(y_test)

    print(X_train.shape,X_test.shape)

    n_cpu = multiprocessing.cpu_count()
    fv_gmm = FisherVectorGMM(n_kernels=n_cpu).fit(X_train[100])

    fv = fv_gmm.predict(X_test[:100])
    print(fv)

    return 0

if (__name__ == "__main__"):
    main(argv)
