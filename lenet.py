# -*- encoding: utf-8 -*-

from sys import argv, exit

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

from acquiring import *
from util import *


class LeNet:
    @staticmethod
    def __init__(numChannels, imgRows, imgCols, numClasses, activation='relu', weightsPath=None):

        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        if (K.image_data_format() == 'channels_first'):
            inputShape = (numChannels, imgRows, imgCols)

        model.add(Conv2D(20, 5, padding='same', input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(50, 5, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPadding2D(pool_size=(2,2), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        if (not weightsPath == None):
            model.load_weights(weightsPath)

        print('here')

        return model

def read_definitions(dim_imgs):
    imgRows = int(get_definitions('ImageSize','img_train_size',dim_imgs)[1:-1])
    imgCols = imgRows
    numClasses = int(get_definitions('Applications','number',dim_imgs))/2
    numChannels = 1

    return imgRows, imgCols, numClasses, numChannels

def read_dataset():
    img_path=get_definitions("Paths","img_path",dim_imgs)
    

def main():
    if (len(argv) != 2):
        print('Usage:', argv[0], '<dim_imgs>')
        exit(-1)

    dim_imgs = argv[1]

    imgRows, imgCols, numClasses, numChannels = read_definitions(dim_imgs)
    read_dataset()

    model = LeNet(numChannels, imgRows, imgCols, numClasses)

    return 0

if (__name__ == '__main__'):
    main()
