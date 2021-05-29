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
methods = ["LBP", "DCTraCS_ULBP", "DCTraCS_RLBP", "Eerman", "Soysal", "Zhang", "GLCM"] # "Fahad"

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
    imgs = []
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
                imgs.append([TM,i,clnum,t])

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
                imgs.append([TM,i,clnum,t])

    # 1.5 Start multiprocessing step:
    #     - For each image, feature extraction process will be performed in a single thread
    #     - Create a vector to help in multiprocessing step
    #     - Get the best number of threads (= CPU cores)
    #     - Create n_threads
    #     - Start the threads
    #     - Close the threads
    class_vect = []
    for i in range(len(imgs)):
        class_vect.append(i)
    #for i in range(len(class_vect)):
    #    feature_extraction(imgs,dim_imgs,class_vect[i])
    print('Extracting features...')
    n_threads = multiprocessing.cpu_count()
    pool = ThreadPool(n_threads)
    result = pool.map(partial(feature_extraction,imgs,dim_imgs),class_vect)
    pool.close()
    pool.join()

# 2. Feature extraction process
def bkp_feature_extraction(imgs,dim_imgs,i):
    # 2.1 Split the imgs list into variables
    TM        = imgs[i][0]
    clnum     = imgs[i][1]
    class_img = imgs[i][2]

    # 2.2 Run preprocessing algorithm
    # Preprocessing
    preproc=globals()[get_definitions("Functions","preprocessing",dim_imgs)]
    TM_unscr = preproc.__call__(TM)
    #TM_unscr = TM

    # 2.3 For each method, run feature extraction process and save in the respective directory
    # Feature extraction (all methods)
    # DCTraCS_ULBP:
    write_training_scikit(clnum, ulbp(TM_unscr, True), "DCTraCS_ULBP", "class"+str(clnum)+".sck", class_img, dim_imgs)
    # DCTraCS_RLBP
    write_training_scikit(clnum, rlbp(TM_unscr, True), "DCTraCS_RLBP", "class"+str(clnum)+".sck", class_img, dim_imgs)
    # Eerman
    write_training_scikit(clnum, eermanFeatures(TM, True), "Eerman", "class"+str(clnum)+".sck", class_img, dim_imgs)
    # Fahad
    #write_training_scikit(clnum, fahadFeatures(TM, True), "Fahad", "class"+str(clnum)+".sck", class_img, dim_imgs)
    # Soysal
    write_training_scikit(clnum, soysalFeatures(TM, True), "Soysal", "class"+str(clnum)+".sck", class_img, dim_imgs)
    # Zhang
    write_training_scikit(clnum, zhangFeatures(TM, True), "Zhang", "class"+str(clnum)+".sck", class_img, dim_imgs)

def feature_extraction(imgs,dim_imgs,i):
    #from util2 import write_training_scikit

    # 2.1 Split the imgs list into variables
    TM        = imgs[i][0]
    clnum     = imgs[i][1]
    class_img = imgs[i][2]
    #label     = imgs[i][3]

    # 2.2 Run preprocessing algorithm
    # Preprocessing
    #preproc=globals()[get_definitions("Functions","preprocessing",dim_imgs)]
    #TM_unscr = preproc.__call__(TM)
    TM_unscr = TM

    # 2.3 For each method, run feature extraction process and save in the respective directory
    # Feature extraction (all methods)
    # LBP:
    write_training_scikit(clnum, lbp(TM_unscr, True), "LBP", "class"+str(clnum)+".sck", class_img, dim_imgs)
    ## DCTraCS_ULBP:
    write_training_scikit(clnum, ulbp(TM_unscr, True), "DCTraCS_ULBP", "class"+str(clnum)+".sck", class_img, dim_imgs)
    ## DCTraCS_RLBP
    write_training_scikit(clnum, rlbp(TM_unscr, True), "DCTraCS_RLBP", "class"+str(clnum)+".sck", class_img, dim_imgs)
    ## Eerman
    write_training_scikit(clnum, eermanFeatures(TM, True), "Eerman", "class"+str(clnum)+".sck", class_img, dim_imgs)
    ## Fahad
    #write_training_scikit(clnum, fahadFeatures(TM, True), "Fahad", "class"+str(clnum)+".sck", class_img, dim_imgs)
    ## Soysal
    write_training_scikit(clnum, soysalFeatures(TM, True), "Soysal", "class"+str(clnum)+".sck", class_img, dim_imgs)
    ## Zhang
    write_training_scikit(clnum, zhangFeatures(TM, True), "Zhang", "class"+str(clnum)+".sck", class_img, dim_imgs)
    ## GLCM
    write_training_scikit(clnum, glcmFeatures(TM), "GLCM", "class"+str(clnum)+".sck", class_img, dim_imgs)
    


if __name__ == "__main__":
    main(sys.argv)

