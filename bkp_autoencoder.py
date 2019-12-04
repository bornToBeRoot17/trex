import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
import os
DATASET = '/nobackup/ppginf/rgcastro/research/dataset2/'
classes = ['ft','lu','map','allreduce']

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

import numpy as np
x_train = np.asarray(x_train)
x_test  = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test  = np.asarray(y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

import numpy as np

#temp = []
#for img in x_train:
#    t = []
#    for row in img:
#        for i in row:
#            t.append(i)
#    temp.append(t)
#x_train = []
#x_train = temp

#x_train = np.array(x_train)

#x_train = x_train.reshape(60000,784)
x_train = x_train.reshape(x_train.shape[0],img_rows*img_cols)

#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#exit(-1)

model = Sequential()
model.add(Dense(784,activation='relu',input_dim=784))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(784,activation='relu'))

model.compile(loss=keras.losses.mean_squared_error,
             optimizer=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0),
             metrics = ['accuracy'])

model.fit(x_train,x_train,verbose=1,epochs=10,batch_size=256)
model.save('C:\\Users\\Rohith\\Documents\\Rohith_Stuff\\Datasets\\auto_en.h5')
#del model

from keras.models import load_model
import cv2

model = load_model('C:\\Users\\Rohith\\Documents\\Rohith_Stuff\\Datasets\\auto_en.h5')

test = x_train[1].reshape(1,784)
y_test = model.predict(test)

inp_img = []
temp = []
for i in range(len(test[0])):
    if((i+1)%28 == 0):
        temp.append(test[0][i])
        inp_img.append(temp)
        temp = []
    else:
        temp.append(test[0][i])
out_img = []
temp = []
for i in range(len(y_test[0])):
    if((i+1)%28 == 0):
        temp.append(y_test[0][i])
        out_img.append(temp)
        temp = []
    else:
        temp.append(y_test[0][i])
        
inp_img = np.array(inp_img)
out_img = np.array(out_img)
        
cv2.imshow('Test Image',inp_img)
cv2.imshow('Output Image',out_img)
cv2.waitKey(0)
