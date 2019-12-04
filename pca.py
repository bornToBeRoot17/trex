#!/usr/bin/python
# encoding: utf-8

import os, sys
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

from util import *
from acquiring import *
from preprocessing import *
from feature_extraction import *
from relwork import *

import numpy as np
from sklearn import decomposition
#from sklearn.preprocessing import StandardScaler

TM_unscr = [[],[],[],[]]

def feature_extraction(img_path, dic_apps, app, clnum, dim_images, TM_unscr, t):
    dimScaling = int(dic_apps[app][1])
    #create a matrix for storing the TM. Each position holds normalized number of bytes transmitted from TM[line] to TM[col]
    TM=[ [ 0 for i in range(dimScaling) ] for j in range(dimScaling) ]

    # Acquiring (from images)
    acq_tm=globals()[get_definitions("Functions","acquiring",dim_images)]
    tm_img_path=img_path+"/"+dic_apps[app][0]+"/tm/tm"+str(t)+".png"
    acq_tm.__call__(TM, tm_img_path)

    # Preprocessing
    preproc=globals()[get_definitions("Functions","preprocessing",dim_images)]
    TM_unscr[int(clnum)-1].append(np.asarray(preproc.__call__(TM)).flatten())

def compute_classes(dic_apps,img_path,dim_images,i):
    app, clnum, times = parse_classes(get_definitions("Classes","l"+str(i),dim_images))

    #if (app != "bifu512"): continue

    print "Acquiring: class"+clnum+" app="+app
    #nThreads = len(times)
    nThreads = 16
    #if (nThreads > 250 or "512" in app):
    #    nThreads = 50

    pool = ThreadPool(nThreads)
    results = pool.map(partial(feature_extraction,img_path,dic_apps,app,clnum,dim_images,TM_unscr),times)
    #feature_extraction(img_path,dic_apps,app,clnum,dim_images,TM_unscr,times[0])
    pool.close()
    pool.join()

    return TM_unscr

def pca():
    for i in range(len(TM_unscr)):
        X = TM_unscr[i]

        #sc = StandardScaler()
        #X_std = sc.fit_transform(X)

        #pca = decomposition.PCA(n_components=2)
        #X_std_pca = pca.fit_transform(X_std)

        pca = decomposition.PCA(n_components=2)
        pca.fit(X)
        print pca.singular_values_
        print

def main(argv):
    if (len(argv) != 2):
        print("Usage:", argv[0], "<dim_images>")
        exit(-1)

    dim_images = argv[1]

    hosts = read_hosts(get_definitions("FileNames","hosts",dim_images))

    img_path=get_definitions("Paths","img_path",dim_images)
    dic_apps = get_applications(dim_images)

    # Parse all class lines provided in "definitions.ini"
    classVect = []
    nClasses = int(get_definitions("Classes","number_of_lines",dim_images))
    for i in range(nClasses):
        classVect.append(i+1)
    pool = ThreadPool(nClasses)
    result = pool.map(partial(compute_classes,dic_apps,img_path,dim_images),classVect)
    #result = compute_classes(dic_apps,img_path,dim_images,classVect[0])
    pool.close()
    pool.join()

    pca()

if __name__ == "__main__":
    main(sys.argv)

