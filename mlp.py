# -*- encoding: utf-8 -*-

from __future__ import print_function
from sys import argv, exit
import numpy as np
import os
import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

classes = ['ft', 'lu', 'map', 'allreduce']
#classes = ['ft', 'lu']
dataset = '/nobackup/ppginf/rgcastro/research/dataset2/'

batch_size = 16
num_classes = len(classes)
epochs = 100

def load_data(train_dim, test_dim):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(len(classes)):
        dir = dataset + classes[i] + '_' + train_dim + '/'
        for r, d, f in os.walk(dir):
            for file in f:
                img = cv2.imread(dir+file, cv2.IMREAD_GRAYSCALE)
                #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (256,256))
                x_train.append(img)
                y_train.append(i)

        dir = dataset + classes[i] + '_' + test_dim + '/'
        for r, d, f in os.walk(dir):
            for file in f:
                img = cv2.imread(dir+file, cv2.IMREAD_GRAYSCALE)
                #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img, (256,256))
                x_test.append(img)
                y_test.append(i)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

def mlp(x_train, x_test, y_train, y_test):

    x_train  = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_test   = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    #print(x_train.shape)
    #print(x_test.shape)
    #exit(-1)

    model = Sequential()
    #model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1]*x_train.shape[1],)))
    #model.add(Dense(512, activation='relu', input_shape=(None,)))
    model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
          optimizer=RMSprop(),
          metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def vgg16(x_train, x_test, y_train, y_test):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.optimizers import Adam
    import numpy as np

    #print(x_train.shape)
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[1], 3)
    #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[1], 3)
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255

    model = Sequential()

    model.add(Conv2D(input_shape=(x_train.shape[1],x_train.shape[2],3), filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    model.summary()

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    checkpoint = ModelCheckpoint('vgg16.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def tcnn(x_train, x_test, y_train, y_test):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    import numpy as np
    np.random.seed(1000)

    model = Sequential()

    # Convolutional
    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(2224*224*3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()

    #opt = Adam(lr=0.001)
    #model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def main(argv):
    if (len(argv) != 3):
        print('Usage:', argv[0], '<train_dim> <test_dim>')
        exit(-1)

    train_dim = argv[1]
    test_dim  = argv[2]

    x_train, x_test, y_train, y_test = load_data(train_dim, test_dim)

    mlp(x_train, x_test, y_train, y_test)
    #vgg16(x_train, x_test, y_train, y_test)
    #tcnn(x_train, x_test, y_train, y_test)

    return 0

if (__name__ == '__main__'):
    main(argv)
