# -*- encoding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import os


#RESULT_DIR = "./results_resize_test/"
RESULT_DIR = "./results/"
methods = ['DCTraCS_ULBP', 'DCTraCS_RLBP', 'Eerman', 'Fahad', 'Soysal']
classifiers = ['svm', 'rf']
dims = [32, 64, 100, 128, 200, 256, 300, 512, 1024, 1500]
metric = "average"

def get_files():
    result_files = []
    all_files = os.listdir(RESULT_DIR)

    for f in all_files:
        if (f[-4:] == ".txt" and f.count('_') == 2):
            result_files.append(f)

    return result_files

def generate_tables(result_files):
    svm_values = {}
    rf_values = {}

    for method in methods:
        svm_values[method] = {}
        rf_values[method] = {}
        for tr in dims:
            if (metric != "worse"):
                svm_values[method][tr] = 0.
                rf_values[method][tr]  = 0.
            else:
                svm_values[method][tr] = 100.
                rf_values[method][tr]  = 100.

    for i in range(len(dims)):
        tr = dims[i]
        fname = "resize_tr"+str(tr)+'_ts'+str(tr)+'.txt'
        with open(RESULT_DIR+fname, 'r') as f:
            strF = f.readlines()
            for method in methods:
                aux_line_method = []
                for line in strF:
                    if (method in line and '%' in line): aux_line_method.append(line)

                for line in aux_line_method:
                    idx = line.find('%') - 6
                    prcnt = line[idx:idx+6]
                    if (prcnt[0] == ' '): prcnt = prcnt[1:]
                    if (prcnt[0] == ':'): prcnt = prcnt[2:]
                    prcnt = float(prcnt)

                    if ('svm' in line):
                        if (metric == "best" and prcnt > svm_values[method][tr][ts]):
                            svm_values[method][tr] = prcnt
                        elif (metric == "worse" and prcnt < svm_values[method][tr][ts]):
                            svm_values[method][tr] = prcnt
                        elif (metric == "average"):
                            svm_values[method][tr] += prcnt
                    elif ('rf' in line):
                        if (metric == "best" and prcnt > rf_values[method][tr][ts]):
                            rf_values[method][tr] = prcnt
                        elif (metric == "worse" and prcnt < rf_values[method][tr][ts]):
                            rf_values[method][tr] = prcnt
                        elif (metric == "average"):
                            rf_values[method][tr] += prcnt

        f.close()

    return rf_values, svm_values

def main():
    result_files = get_files()
    rf_values, svm_values = generate_tables(result_files)

    rf = []
    for method in methods:
        rf_array = []
        for tr in dims:
            rf_array.append(rf_values[method][tr])
        rf.append(rf_array)

    svm = []
    for method in methods:
        svm_array = []
        for tr in dims:
            svm_array.append(svm_values[method][tr])
        svm.append(svm_array)

    heatmap(svm, rf)
    #heatmap(rf, "rf", train_dims, test_dims)

def heatmap(svm, rf):
    sns.set(font_scale=2.)
    #cmap = sns.cubehelix_palette(n_colors=100, dark=0, light=1, reverse=False, as_cmap=True)
    cmap = sns.dark_palette("white", 100, as_cmap=True, reverse=True)

    fig, axs = plt.subplots(2, figsize=(25,12), gridspec_kw={'hspace': 0.2, 'wspace': 2})
    for ax in axs.flat:
        im = ax.imshow(np.random.random((10,10)), vmin=90, vmax=100)

    for i in range(len(classifiers)):
        ax = axs[i]
        if (classifiers[i] == 'svm'):
            str_clf = 'SVM'
            data = svm
        else:
            str_clf = 'Random Forest'
            data = rf

        hm = sns.heatmap(data,
                         ax = ax,
                         #cmap = "seismic",
                         cmap = cmap,
                         cbar=False,
                         #square = True,
                         annot = True,
                         fmt = '.2f',
                         cbar_kws = {"shrink":0.8, "pad": 0.02},
                         annot_kws = {"size": 24},
                         linewidths = .01,
                         linecolor = 'black',
                         vmin = 90,
                         vmax = 100
                         #xticklabels = test_dims,
                         #yticklabels = train_dims
        )


        #fig, (ax) = plt.subplots(figsize=(20,12))

        plt.xlabel('Test dimension', fontsize=25, fontweight='bold')
        plt.ylabel('Feature extractors', fontsize=25, fontweight='bold')
        ax.set_xticklabels(dims)
        ax.set_yticklabels(methods, rotation=45, va='top')
        ax.set_title(str_clf, fontsize=30, fontweight='bold')
        #ax.set_ylabels('Test dimension', fontsize=30, fontweight='bold')
        #ax.set_xlabels(train_dims)

    fig.subplots_adjust(top=0.93)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, pad=0.02)
    #fig.suptitle(method + " with " + str_clf,
    #             fontsize=35,
    #             fontweight='bold'
    #)
    #plt.show()
    fig.savefig("img_results/heatmaps_main_diagonal.pdf", bbox_inches='tight')

    return 0

if (__name__ == "__main__"):
    main()
