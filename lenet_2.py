import os, sys
from functools import partial
from multiprocessing.pool import ThreadPool
import multiprocessing

from util import *
from acquiring import *
from preprocessing import *
from feature_extraction import *
from relwork import *

import shutil


# 0. List that contains all methods used in this work
methods = ["LBP", "DCTraCS_ULBP", "DCTraCS_RLBP", "Eerman", "Soysal", "Zhang"] # "Fahad"

from sys import argv, exit

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import SGD

from acquiring import *
from util import *


class LeNet:
    model = None

    #@staticmethod
    def __init__(self, numChannels, imgRows, imgCols, numClasses, activation='relu', weightsPath=None):

        self.model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        if (K.image_data_format() == 'channels_first'):
            inputShape = (numChannels, imgRows, imgCols)

        self.model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(128,128,1), padding='same'))
        self.model.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
        self.model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.model.add(Flatten())
        self.model.add(Dense(84, activation='tanh'))
        self.model.add(Dense(numClasses, activation='softmax'))
        #self.model.add(Activation('softmax'))

        if (not weightsPath == None):
            self.model.load_weights(weightsPath)

        self.model.compile(
            loss = 'sparse_categorical_crossentropy',
            optimizer = SGD(lr = 0.01),
            metrics = ['accuracy'])

def split_dims(dim_lst):
    if ('[' in dim_lst): dim_lst = dim_lst.replace('[','')
    if (']' in dim_lst): dim_lst = dim_lst.replace(']','')
    if (',' in dim_lst): dim_lst = dim_lst.split(',')
    else: dim_lst = [dim_lst]

    return dim_lst

# 1. Main function
def main(argv):
    # 1.0 Verify if the call contains all the arguments
    if (len(argv) != 2):
        print("Usage:", argv[0], "<dim_imgs>")
        exit(-1)

    # 1.1 Starting the code
    #    - Store the dimension argument in a variable
    #    - Initialize the image list
    #    - Read hosts from hosts file
    #    - Create the feature extraction paths to save .sck files
    #    - Get the values used in resize step
    #        - If newDim == dimToResize, resize step won't be performed)
    #        - Using ImageSize as new size of training images
    dim_imgs = argv[1]
    X_train = []
    y_train = []
    X_test  = []
    y_test  = []
    #hosts = read_hosts(get_definitions("FileNames","hosts",dim_imgs))
    #sci_path = get_sckit_filepath(dim_imgs, out=1)
    #if (os.path.exists(sci_path)):
    #    shutil.rmtree(sci_path)
    #for method in methods:
    #    os.makedirs(sci_path+"/"+method)
    train_dim    = split_dims(get_definitions("ImageSize","img_train_size",dim_imgs))
    train_resize = split_dims(get_definitions("ImageSize","img_train_resize",dim_imgs))
    test_dim     = split_dims(get_definitions("ImageSize","img_test_size",dim_imgs))
    test_resize  = split_dims(get_definitions("ImageSize","img_test_resize",dim_imgs))

    # 1.2 Get the dataset path
    img_path=get_definitions("Paths","img_path",dim_imgs)
    dic_apps = get_applications(dim_imgs)
    # 1.3 Parse all class lines provided in definitions file
    number_of_lines = int(get_definitions("Classes","number_of_lines",dim_imgs))
    number_of_train_classes = int(get_definitions("Classes","number_of_train_classes",dim_imgs))
    acq_name = get_definitions("Functions","acquiring",dim_imgs)
    #acq_tm=globals()[get_definitions("Functions","acquiring",dim_imgs)]
    acq_tm=globals()[acq_name]

    # >>> Train imgs
    # 1.4 Run over all class lines
    for i in range(1, number_of_train_classes+1):
        # 1.4.1 Split the class line
        #     - App: application name
        #     - clnum: column number
        #     - times: list with the images number that corresponding to the application
        app, clnum, times = parse_classes(get_definitions("Classes","l"+str(i),dim_imgs))
        print("Acquiring: class"+clnum+" app="+app)
        # 1.4.2 Run over all application images
        for t in times:
            # 1.4.2.1 Read the image and store it in TM
            #     - Perform resize, if needed
            #     - Apply filters, if needed
            #     - Invert colors, if needed
            tm_img_path=img_path+"/"+dic_apps[app]+"/fig"+str(t)+".png"
            TMs = acq_tm.__call__(tm_img_path,train_dim,train_resize)

            # 1.4.2.2 Add image, application number and class number in img list
            #     - TM: image
            #     - i: application number
            #     - clnum: class number
            for TM in TMs:
                X_train.append(TM)
                y_train.append(int(clnum)-1)

    # >>> Test imgs
    for i in range(number_of_train_classes+1, number_of_lines+1):
        # 1.4.1 Split the class line
        #     - App: application name
        #     - clnum: column number
        #     - times: list with the images number that corresponding to the application
        app, clnum, times = parse_classes(get_definitions("Classes","l"+str(i),dim_imgs))
        print("Acquiring: class"+clnum+" app="+app)
        # 1.4.2 Run over all application images
        for t in times:
            # 1.4.2.1 Read the image and store it in TM
            #     - Perform resize, if needed
            #     - Apply filters, if needed
            #     - Invert colors, if needed
            tm_img_path=img_path+"/"+dic_apps[app]+"/fig"+str(t)+".png"
            TMs = acq_tm.__call__(tm_img_path,test_dim,test_resize)

            # 1.4.2.2 Add image, application number and class number in img list
            #     - TM: image
            #     - i: application number
            #     - clnum: class number
            for TM in TMs:
                X_test.append(TM)
                y_test.append(int(clnum)-1)

    # 1.5 Start multiprocessing step:
    #     - For each image, feature extraction process will be performed in a single thread
    #     - Create a vector to help in multiprocessing step
    #     - Get the best number of threads (= CPU cores)
    #     - Create n_threads
    #     - Start the threads
    #     - Close the threads
    #class_vect = []
    #for i in range(len(imgs)):
    #    class_vect.append(i)

    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    X_train = X_train.reshape(X_train.shape[0],128,128,1)
    X_test = X_test.reshape(X_test.shape[0],128,128,1)

    lenet = LeNet(1, int(train_dim[0]), int(train_dim[0]), number_of_train_classes)
    print(lenet.model.summary())

    lenet.model.fit(
        X_train,
        y_train,
        batch_size = 128,
        epochs = 20,
        verbose = 1)

    (loss, accuracy) = lenet.model.evaluate(
        X_test,
        y_test,
        batch_size = 128,
        verbose = 1)

    print(accuracy)

if (__name__ == '__main__'):
    main(argv)
