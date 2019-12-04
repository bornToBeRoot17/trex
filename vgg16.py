#!./env3/bin/python3
# -*- encoding:utf-8 -*-

#from keras.models import Sequential
#from keras.layers.core import Flatten, Dense, Dropout
#from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
#from keras.optimizers import SGD
#import cv2, numpy as np

# VGG from keras
from sys import argv, exit
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Flatten, Dense
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import cv2, numpy as np

from util import *
#from acquiring import *


def train_test_generator():
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = datagen.flow_from_directory(
        directory='./train/',
        target_size=(256,256),
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    test_generator = datagen.flow_from_directory(
        directory='./test/',
        target_size=(256,256),
        color_mode='rgb',
        batch_size=1,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    return train_generator, test_generator

def image_parser(img_name, dim):
    im = cv2.imread(img_name).astype(np.float32)
    height, width, channels = im.shape
    if (width != dim):
        im = cv2.resize(im, (dim, dim)).astype(np.float32)

    return im

def VGG_16_keras(num_classes):
    input = Input(shape=(256,256,3), name='img_name')
    model_vgg16_conv  = VGG16(weights='imagenet', include_top=False)
    output_vgg16_conv = model_vgg16_conv(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(4, activation='softmax', name='predictions')(x)
    #x = Dense(1, activation='softmax', name='predictions')(x)

    model = Model(inputs=input, outputs=x)

    for layer in model.layers[:2]:
        layer.trainable = False
    for layer in model.layers[2:]:
        layer.trainable = True

    #print(model.summary())
    #features = model.predict(x)

    return model

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_first'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_first'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_first'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_first'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def main(argv):
    if (len(argv) != 2):
        print("Usage:", argv[0], "<sufix_definitions>")
        exit(-1)

    sufix        = argv[1]
    num_datasets = int(get_definitions("Classes","number_of_lines",sufix))
    num_classes  = int(get_definitions("Classes","number_of_train_classes",sufix))
    train_dims   = split_dims(get_definitions("ImageSize","img_train_size",sufix))
    img_path     = get_definitions("Paths","img_path",sufix)
    dic_apps     = get_applications(sufix)
    acq_tm       = globals()[get_definitions("Functions","acquiring",sufix)]

    aux_train_dims = []
    for i in range(len(train_dims)):
        aux_train_dims.append(int(train_dims[i]))
    train_dim = min(aux_train_dims)

    X_train = []
    y_train = []
    X_test  = []
    y_test  = []

    a = 1
    if (not a):
        for i in range(1, num_classes+1):
            app, clnum, times = parse_classes(get_definitions("Classes","l"+str(i),sufix))
            for t in times:
                tm_img_path = img_path+dic_apps[app]+'tm/tm'+str(t)+'.png'
                X_train.append(acq_tm.__call__(tm_img_path, train_dim))
                y_train.append(int(clnum))

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        for i in range(num_classes+1, num_datasets+1):
            app, clnum, times = parse_classes(get_definitions("Classes","l"+str(i),sufix))
            for t in times:
                tm_img_path = img_path+'/'+dic_apps[app]+'/tm/tm'+str(t)+'.png'
                X_test.append(acq_tm.__call__(tm_img_path, train_dim))
                y_test.append(int(clnum))

        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

    ## Test pretrained model
    #model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy')
    #out = model.predict(im)
    #print(np.argmax(out))

    batch_size = 32
    epochs = 10

    model = VGG_16_keras(num_classes)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    if (not a):
        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test)
        )
        #out = model.predict(im)
        #print(np.argmax(out))
    else:
        train_generator, test_generator = train_test_generator()

        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=test_generator.n//test_generator.batch_size
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=test_generator,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=1
        )

        model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

    return 0

if (__name__ == "__main__"):
    main(argv)
