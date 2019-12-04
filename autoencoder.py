import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

DATASET = '/nobackup/ppginf/rgcastro/research/dataset2/'
classes = ['ft','lu','map','allreduce']

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
N_TEST_IMG = 5

def read_dataset():
    files_class = []
    for c in classes:
        path = DATASET+c+'_32'
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                files.append(os.path.join(r, file))
        files_class.append(files)

    img_rows, img_cols = 28, 28

    import cv2
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    c = 0
    for files in files_class:
        images = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(img_rows, img_cols))
            images.append(img)

        x_train += images[:210]
        for i in range(210):
            y_train.append(c)

        x_test += images[210:]
        for i in range(90):
            y_test.append(c)
        c += 1

    x_train = np.asarray(x_train)
    x_test  = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    return (x_train, x_test, y_train, y_test)

def main():
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,                                     # this is training data
        transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=DOWNLOAD_MNIST,                        # download it if you don't have it
    )

    print(train_data)

    return 0

if (__name__ == '__main__'):
    main()

