#!/usr/bin/python
# -*- encoding: utf-8 -*-

from sys import argv, exit
from util import *
import numpy as np
from PIL import Image

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils  import plot_model

def read_dataset(apps, dimensions):
    for i in range(1,len(apps)+1):
        app, class_num, lst = parse_classes(get_definitions("Classes","l"+str(i),dimensions))
        apps[app].append([class_num]*len(lst))
        #apps[app].append(lst)

        apps[app].append([[]]*len(lst))

        for j in range(len(lst)):
            img_path = "./dataset/" + apps[app][0] + "tm/tm" + str(lst[j]) + ".png"
            img = Image.open(img_path)
            img.load()
            apps[app][len(apps[app])-1][j] = np.asarray(img, dtype="int32")

        apps[app][len(apps[app])-1] = np.asarray(apps[app][len(apps[app])-1])

    #for app in apps:
        #print np.asarray(apps[app][len(apps[app])-1]).shape

def vgg_block(A_prev, n_filters, n_conv):
    for _ in range(n_conv):
        A_prev = Conv2D(n_filters, (3,3), padding='same', activation='relu')(A_prev)
    A_prev = MaxPooling2D((2,2), strides=(2,2))(A_prev)

    return A_prev

def cnn(apps):
    X_train = []
    y_train = []
    X_test  = []
    y_test  = []

    for app in apps:
        if(apps[app][1] == '128'):
            for i in range(len(apps[app][3])):
                X_train.append(apps[app][3][i])
            for i in range(len(apps[app][2])):
                y_train.append(apps[app][2][i])
        else:
            for i in range(len(apps[app][3])):
                X_test.append(apps[app][3][i])
            for i in range(len(apps[app][2])):
                y_test.append(apps[app][2][i])

    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)

    visible = Input(shape=(128, 128, 3))
    layer = vgg_block(visible, 64, 2)
    layer = vgg_block(layer, 128, 2)
    layer = vgg_block(layer, 256, 4)
    model = Model(inputs=visible, outputs=layer)

def main(argv):
    if (len(argv) != 2):
        print "Usage:", argv[0], "<dimensions>"
        exit(-1)

    dimensions = argv[1]

    apps = get_applications(dimensions)

    read_dataset(apps,dimensions)
    cnn(apps)


if (__name__ == "__main__"):
    main(argv)
