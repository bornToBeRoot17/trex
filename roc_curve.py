#!/usr/bin/python
# encoding: utf-8

import glob, scipy, numpy as np #, matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

from util import *

def read_classes(method,dim_images):
    sckit_filepath=get_sckit_filepath(dim_images)
    X = None
    cnt=0
    n_files = len(glob.glob(sckit_filepath+"/"+method+"/class*.sck"))/2
    for i in glob.glob(sckit_filepath+"/"+method+"/class*.sck"):
        if (cnt > n_files): break
        X_train, y_train = load_svmlight_file(i)
        cnt+=1
        if X == None:
            X=X_train
            y=y_train
        else:
            X=scipy.sparse.vstack((X,X_train)) #X+=X_train
            y=np.hstack((y,y_train)) #y+=y_train
    return X, y, cnt

def read_classes_train_test(method,num_train,dim_images):
    sckit_filepath=get_sckit_filepath(dim_images)
    X_train = None
    X_test  = None
    y_train = None
    y_test  = None

    #num_train = int(get_definitions("Classes","number_of_train_classes",dim_imgs))

    count = 0
    for i in glob.glob(sckit_filepath+"/"+method+"/class*.sck"):
        try:
            num_class = int(i[-6]+i[-5])
        except:
            num_class = int(i[-5])

        count += 1
        X, y = load_svmlight_file(i)
        if (num_class > num_train):
            if X_test == None:
                X_test=X
                y_test=y
            else:
                X_test=scipy.sparse.vstack((X_test,X)) #X+=X_train
                y_test=np.hstack((y_test,y)) #y+=y_train
        else:
            if X_train == None:
                X_train=X
                y_train=y
            else:
                X_train=scipy.sparse.vstack((X_train,X)) #X+=X_train
                y_train=np.hstack((y_train,y)) #y+=y_train

    return X_train, y_train, X_test, y_test, count

def show_roc(dim_images):
    method="DCTraCS_ULBP"

    sckit_filepath=get_sckit_filepath(dim_images)

    X, y, i = read_classes(method)
    print("LENX=",i)

    # Binarize the output
    y = label_binarize(y, classes=range(1,i))
    n_classes = y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(get_definitions("Validation","test_size")))

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', C=8192.0, gamma=8.0))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    print(y_score.shape)

    y_pred = classifier.predict(X_test)

    # Compute ROC curve and ROC area for each class (Fig. 6)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    plt.figure()
    colors = cycle(['grey','rosybrown','sienna','gold','r','violet',
               'darkgreen','navy','coral','lawngreen','limegreen',
               'teal','cornflowerblue','aqua'])
    for i, color in zip(range(n_classes), colors):
        if i == 11:
            plt.plot(fpr[i], tpr[i], color=color, linestyle='--', lw=2, label='class{0} ({1:0.3f})'.format(i+1, roc_auc[i]))
        elif i == 4:
            plt.plot(fpr[i], tpr[i], color=color, linestyle=':', lw=2, label='class{0} ({1:0.3f})'.format(i+1, roc_auc[i]))
        else:
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='class{0} ({1:0.3f})'.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.01, 0.21])
    plt.ylim([0.84, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Fig. 6. ROC curve of DCTraCS using ULBP and SVM.")
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig("roc_new.png")




