#!/usr/bin/python

import os, sys
from functools import partial
from multiprocessing import Pool as ThreadPool

from util import *
from acquiring import *
from preprocessing import *
from feature_extraction import *
from relwork import *

import shutil


methods = ["DCTraCS_ULBP", "DCTraCS_RLBP", "Eerman", "Fahad", "Soysal", "Zhang"]

def main(argv):
    if (len(argv) != 2):
        print "Usage:", argv[0], "<dim_imgs>"
        exit(-1)

    dim_imgs = argv[1]
    imgs = []

    sci_path = get_sckit_filepath(dim_imgs, out=1)
    if (os.path.exists(sci_path)):
        shutil.rmtree(sci_path)

    for method in methods:
        os.makedirs(sci_path+"/"+method)

    hosts = read_hosts(get_definitions("FileNames","hosts",dim_imgs))

    img_path=get_definitions("Paths","img_path",dim_imgs)
    dic_apps = get_applications(dim_imgs)
    # Parse all class lines provided in "definitions.ini"
    number_of_lines = int(get_definitions("Classes","number_of_lines",dim_imgs))
    for i in range(1, number_of_lines+1):
        app, clnum, times = parse_classes(get_definitions("Classes","l"+str(i),dim_imgs))
        dim = int(dic_apps[app][1])
        print "Acquiring: class"+clnum+" app="+app
        for t in times:
            #create a matrix for storing the TM. Each position holds normalized number of bytes transmitted from TM[line] to TM[col]
            TM=[ [ 0 for i in range(dim) ] for j in range(dim) ]

            # Acquiring (from images)
            acq_tm=globals()[get_definitions("Functions","acquiring",dim_imgs)]
            tm_img_path=img_path+"/"+dic_apps[app][0]+"/tm/tm"+str(t)+".png"
            acq_tm.__call__(TM, tm_img_path)

            imgs.append([TM,clnum])

    #print len(imgs)
    #return 0

    class_vect = []
    for i in range(len(imgs)):
        class_vect.append(i)

    #n_threads = len(imgs) / 16 + 1
    n_threads = 16
    pool = ThreadPool(n_threads)
    result = pool.map(partial(feature_extraction,imgs,dim_imgs),class_vect)
    pool.close()
    pool.join()

def feature_extraction(imgs,dim_imgs,i):
    TM    = imgs[i][0]
    clnum = imgs[i][1]

    # Preprocessing
    preproc=globals()[get_definitions("Functions","preprocessing",dim_imgs)]
    TM_unscr = preproc.__call__(TM)

    # Feature extraction (all methods)
    # DCTraCS_ULBP:
    write_training_scikit(clnum, ulbp(TM_unscr, True), "DCTraCS_ULBP", "class"+str(clnum)+".sck", dim_imgs)
    # DCTraCS_RLBP
    write_training_scikit(clnum, rlbp(TM_unscr, True), "DCTraCS_RLBP", "class"+str(clnum)+".sck", dim_imgs)
    # Eerman
    write_training_scikit(clnum, eermanFeatures(TM, True), "Eerman", "class"+str(clnum)+".sck", dim_imgs)
    # Fahad
    write_training_scikit(clnum, fahadFeatures(TM, True), "Fahad", "class"+str(clnum)+".sck", dim_imgs)
    # Soysal
    write_training_scikit(clnum, soysalFeatures(TM, True), "Soysal", "class"+str(clnum)+".sck", dim_imgs)
    # Zhang
    write_training_scikit(clnum, zhangFeatures(TM, True), "Zhang", "class"+str(clnum)+".sck", dim_imgs)


if __name__ == "__main__":
    main(sys.argv)

