# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import os


dataset = '/nobackup/ppginf/rgcastro/research/dataset2/'
classes = ['ft','lu', 'map', 'allreduce']
train_dims = [32]
test_dims = [512]

def read_imgs(dataset, dims):
    imgs = []
    labels = []

    for dim in dims:
        for i in range(len(classes)):
            dir = dataset + classes[i] + '_' + str(dim) + '/'
            for r, d, f in os.walk(dir):
                for file in f:
                    fname = dir + file
                    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

                    #img = filter(img)

                    imgs.append(img)
                    labels.append(i)

    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    #imgs = imgs.reshape(imgs.shape[0], imgs.shape[1]*imgs.shape[2])

    return imgs, labels

def split_imgs(img, label):

    split_imgs = []
    labels = []

    train_dim = 32

    for i in range(0,img.shape[0],train_dim):
        for j in range(0,img.shape[1],train_dim):
            split_img = img[i:i+train_dim,j:j+train_dim]
            split_imgs.append(split_img)
            labels.append(label)

    return split_imgs, labels

def classify(X_train, y_train, X_test, y_test, num):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.metrics import score

    clf = SVC(kernel='rbf', C=8192.0, gamma=8.0)
    clf.fit(X_train, y_train)

    #y_pred = clf.predict(X_test)
    #score = clf.score(X_test,y_test)
    #print(score)

    final_pred = []
    for start in range(0,X_test.shape[0],num):
        y_pred = [0,0,0,0]
        for i in range(num):
            pred = int(clf.predict(X_test[start+i]))
            y_pred[pred] += 1
            #score = clf.score(X_test[i:i+num],y_test[i:i+num])
        max = 0
        for i in range(1,len(y_pred)):
            if (y_pred[i] > y_pred[max]):
                max = i
        final_pred.append(max)
        #print(score,end=' ')

    acc = 0
    for i in range(final_pred):
        if (final_pred[i] == y_test[i]):
            acc += 1
    print(acc/y_pred.shape[0])

    #start = 0
    #while (start < len(X_test)):
    #    y_pred_img = []
    #    for i in range(num):
    #        y_pred = clf.predict(X_test[start+i])
    #        y_pred_img.append(y_pred)
    #    start += num

def main():
    X_train, y_train = read_imgs(dataset, train_dims)
    X_test, y_test   = read_imgs(dataset, test_dims)

    #print(y_train.shape)
    #exit(-1)

    num = 256

    X_test_split = []
    y_test_split = []
    for i in range(len(X_test)):
        imgs, split_labels = split_imgs(X_test[i], y_test[i])
        X_test_split.append(imgs)
        y_test_split.append(split_labels)
    X_test_split = np.asarray(X_test_split)
    y_test_split = np.asarray(y_test_split)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    X_test_split = X_test_split.reshape(X_test_split.shape[0]*X_test_split.shape[1], X_test_split.shape[2]*X_test_split.shape[3])
    y_test_split = y_test_split.reshape(y_test_split.shape[0]*y_test_split.shape[1],)

    #print(X_test_split.shape, y_test_split.shape)
    #exit(-1)

    classify(X_train, y_train, X_test_split, y_test_split, num)
    #classify(X_train, y_train, X_test, y_test, num)

if (__name__ == "__main__"):
    main()
