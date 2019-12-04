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


RESULT_DIR = "./results_final/"
methods = ['DCTraCS_ULBP', 'DCTraCS_RLBP', 'Eerman', 'Fahad', 'Soysal']
classifiers = ['svm', 'rf']
metric = "best"
# key_word = ["Accuracy", "Recall", "F1-Score"]
key_word = "Accuracy"

def read_results_kernel():
    fname = './results_kernel.txt'

    with open(fname,'r') as f:
        lines = f.readlines()
        f.close()

    kernel_dims = []
    data_dims = []
    accuracy = []

    results = {}

    for i in range(len(lines)):
        line = lines[i].split()
        kernel_dim = int(line[2])
        data_dim = int(line[5])
        accuracy = float(line[7])*100

        if (kernel_dim not in kernel_dims):
            kernel_dims.append(kernel_dim)
            results[kernel_dim] = {}

        if (data_dim not in data_dims):
            data_dims.append(data_dim)

        results[kernel_dim][data_dim] = accuracy

    return results, kernel_dims, data_dims

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
    train_dims = []
    test_dims = []

    for i in range(len(result_files)):
        tr = int(result_files[i].split('_')[1][2:])
        ts = int((result_files[i].split('_')[2][2:]).split('.')[0])
        if (tr not in train_dims): train_dims.append(tr)
        if (ts not in test_dims): test_dims.append(ts)

    for method in methods:
        svm_values[method] = {}
        rf_values[method] = {}
        for tr in train_dims:
            svm_values[method][tr] = {}
            rf_values[method][tr] = {}
            for ts in test_dims:
                if (metric != "worse"):
                    svm_values[method][tr][ts] = 0.
                    rf_values[method][tr][ts]  = 0.
                else:
                    svm_values[method][tr][ts] = 100.
                    rf_values[method][tr][ts]  = 100.

    for i in range(len(result_files)):
        tr = int(result_files[i].split('_')[1][2:])
        ts = int((result_files[i].split('_')[2][2:]).split('.')[0])
        with open(RESULT_DIR+result_files[i], 'r') as f:
            strF = f.readlines()
            for method in methods:
                aux_line_method = []
                for line in strF:
                    if (key_word in line and method in line):
                        aux_line_method.append(line)

                for line in aux_line_method:
                    #idx = line.find('%') - 6
                    #prcnt = line[idx:idx+6]
                    #if (prcnt[0] == ' '): prcnt = prcnt[1:]
                    #if (prcnt[0] == ':'): prcnt = prcnt[2:]
                    #prcnt = float(prcnt)
                    prcnt = float(line.split(':')[1].split('%')[0][1:])

                    if ('svm' in line):
                        if (metric == "best" and prcnt > svm_values[method][tr][ts]):
                            svm_values[method][tr][ts] = prcnt
                        elif (metric == "worse" and prcnt < svm_values[method][tr][ts]):
                            svm_values[method][tr][ts] = prcnt
                        elif (metric == "average"):
                            svm_values[method][tr][ts] += prcnt
                    elif ('rf' in line):
                        if (metric == "best" and prcnt > rf_values[method][tr][ts]):
                            rf_values[method][tr][ts] = prcnt
                        elif (metric == "worse" and prcnt < rf_values[method][tr][ts]):
                            rf_values[method][tr][ts] = prcnt
                        elif (metric == "average"):
                            rf_values[method][tr][ts] += prcnt

        f.close()

    train_dims.sort()
    test_dims.sort()

    return rf_values, svm_values, train_dims, test_dims

def heatmap(data, clf, train_dims, test_dims):
    if (clf == 'svm'):
        str_clf = 'SVM'
    else:
        str_clf = 'Random Forest'

    sns.set(font_scale=2.)
    #cmap = sns.cubehelix_palette(n_colors=100, dark=0, light=1, reverse=False, as_cmap=True)
    cmap = sns.dark_palette("white", 100, as_cmap=True, reverse=True)

    if (clf == 'kernel'):
        rf_array = data

        #fig, (ax) = plt.subplots(figsize=(20,12))
        fig, (ax) = plt.subplots(figsize=(12,6))
        hm = sns.heatmap(rf_array,
                         ax = ax,
                         #cmap = "seismic",
                         cmap = cmap,
                         #square = True,
                         annot = True,
                         fmt = '.2f',
                         cbar_kws = {"shrink":0.8, "pad": 0.02},
                         annot_kws = {"size": 24},
                         linewidths = .01,
                         linecolor = 'black',
                         vmin = 90,
                         vmax = 100,
                         xticklabels = test_dims,
                         yticklabels = train_dims
        )

        #for i in range(1,len(train_dims)):
        #    for j in range(0,i):
        #        ax.add_patch(Rectangle((j, i), 1, 1, fill=True, edgecolor='white', facecolor='white', lw=1))

        for i in range(len(train_dims)):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))

        plt.xlabel('Data dimension', fontsize=30, fontweight='bold')
        plt.ylabel('Kernel dimension', fontsize=30, fontweight='bold')
        #ax.set_ylabels(train_dims)
        #ax.set_xlabels(test_dims)

        fig.subplots_adjust(top=0.93)
        #fig.suptitle(method + " with " + str_clf,
        #             fontsize=35,
        #             fontweight='bold'
        #)
        #plt.show()
        fig.savefig("img_results/heatmap_"+clf+"_"+key_word+".pdf", bbox_inches='tight')
        print("img_results/heatmap_"+clf+"_"+key_word+".pdf")

        return

    for i in range(len(methods)):
        rf_array = data[i]
        method = methods[i]

        #fig, (ax) = plt.subplots(figsize=(20,12))
        fig, (ax) = plt.subplots(figsize=(20,6))
        hm = sns.heatmap(rf_array,
                         ax = ax,
                         #cmap = "seismic",
                         cmap = cmap,
                         #square = True,
                         annot = True,
                         fmt = '.2f',
                         cbar_kws = {"shrink":0.8, "pad": 0.02},
                         annot_kws = {"size": 24},
                         linewidths = .01,
                         linecolor = 'black',
                         vmin = 90,
                         vmax = 100,
                         xticklabels = test_dims,
                         yticklabels = train_dims
        )

        for i in range(len(train_dims)):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))

        plt.xlabel('Test dimension', fontsize=30, fontweight='bold')
        plt.ylabel('Train dimension', fontsize=30, fontweight='bold')
        #ax.set_ylabels(train_dims)
        #ax.set_xlabels(test_dims)

        fig.subplots_adjust(top=0.93)
        #fig.suptitle(method + " with " + str_clf,
        #             fontsize=35,
        #             fontweight='bold'
        #)
        #plt.show()
        fig.savefig("img_results/heatmap_"+method+"_"+clf+"_"+key_word+".pdf", bbox_inches='tight')
        print("img_results/heatmap_"+method+"_"+clf+"_"+key_word+".pdf")

    return 0

def graphic_3d_polygon(values, str_clf, train_dims, test_dims):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    import numpy as np
    import random

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #plt.locator_params(axis='y', nbins=10)
    #plt.locator_params(axis='x', nbins=12)
    train_dims = np.asarray(train_dims)
    test_dims = np.asarray(test_dims)

    #def cc(arg):
    #    return mcolors.to_rgba(arg, alpha=0.6)

    if (str_clf != 'kernel'):
        values = values[1]

    verts = []
    xs = np.arange(0, len(train_dims)+2)
    zs = np.arange(0, len(train_dims)) #[0.0, 1.0, 2.0, 3.0]
    for i in range(len(zs)):
        ys = []
        ys.append(0)
        for j in range(len(values[i])):
            ys.append(values[i][j]) #np.random.rand(len(xs))
        ys.append(0)
        verts.append(list(zip(xs, ys)))

    facecolors = []
    r = 0.5
    g = -0.1
    b = -0.1
    for i in range(10):
        #r += .1
        g += .1
        b += .1
        facecolors.append((r,g,b,0.6))

    poly = PolyCollection(verts, facecolors=facecolors)
    #poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    xticks = ['']
    for i in range(len(test_dims)):
        xticks.append(test_dims[i])
    xticks.append('')

    ax.set_xlabel('Test dimensions')
    ax.set_xlim3d(0, len(train_dims)+2)
    ax.set_xticks(np.arange(0, len(train_dims)+2, step=1))
    ax.set_xticklabels(xticks, rotation=25, va='baseline', ha='center')

    ax.set_ylabel('Train dimensions')
    ax.set_ylim3d(0, len(train_dims))
    ax.set_yticks(np.arange(0, len(train_dims), step=1))
    ax.set_yticklabels(train_dims, rotation=-25, va='baseline', ha='center')

    ax.set_zlabel(key_word + ' (%)')
    ax.set_zlim3d(0, 100)
    ax.set_zticks(np.arange(0, 100, step=10))
    #ax.set_zticklabels(0, 100)

    fig.savefig("img_results/" + key_word + "_polygon.pdf", bbox_inches='tight')
    print("img_results/" + key_word + "_polygon.pdf")

def hex_map(values, str_clf, train_dims, test_dims):
    # Libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import kde

    # Create data: 200 points
    #data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    #x, y = data.T
    data = np.asarray(values[1])
    x = np.arange(10)
    y = np.arange(10)
    #y = np.asarray(test_dims)

    #print(x, data)
    #print(type(data))
    #exit(-1)

    # Create a figure with 6 plot areas
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(21, 5))

    # Everything sarts with a Scatterplot
    #axes[0].set_title('Scatterplot')
    #axes[0].plot(x, y, 'ko')
    # As you can see there is a lot of overplottin here!

    # Thus we can cut the plotting window in several hexbins
    nbins = 10
    axes[1].set_title('Hexbin')
    axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)

    # 2D Histogram
    axes[2].set_title('2D Histogram')
    axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    #k = kde.gaussian_kde(data.T)
    #xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    #zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # plot a density
    #axes[3].set_title('Calculate Gaussian KDE')
    #axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)

    # add shading
    #axes[4].set_title('2D Density with shading')
    #axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

    # contour
    #axes[5].set_title('Contour')
    #axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    #axes[5].contour(xi, yi, zi.reshape(xi.shape) )

    fig.savefig("img_results/" + key_word + "_hex.pdf", bbox_inches='tight')
    print("img_results/" + key_word + "_hex.pdf")

def main():

    results, kernel_dims, data_dims = read_results_kernel()
    results_matrix = []
    for kernel_dim in kernel_dims:
        aux = []
        for data_dim in data_dims:
            try:
                aux.append(results[kernel_dim][data_dim])
            except:
                aux.append(-1)
        results_matrix.append(aux)
    heatmap(results_matrix, "kernel", kernel_dims, data_dims)
    #graphic_3d_polygon(results_matrix,'kernel',kernel_dims,data_dims)
    exit(0)

    result_files = get_files()
    rf_values, svm_values, train_dims, test_dims = generate_tables(result_files)

    rf = []
    for method in methods:
        rf_array = []
        for tr in train_dims:
            aux = []
            for ts in test_dims:
                aux.append(rf_values[method][tr][ts])
            rf_array.append(aux)
        rf.append(rf_array)

    svm = []
    for method in methods:
        svm_array = []
        for tr in train_dims:
            aux = []
            for ts in test_dims:
                aux.append(svm_values[method][tr][ts])
            svm_array.append(aux)
        svm.append(svm_array)

    heatmap(svm, "svm", train_dims, test_dims)
    heatmap(rf, "rf", train_dims, test_dims)

    #idx = methods.index('DCTraCS_RLBP')
    #print(idx)

    #print(key_word)
    #print(svm[idx])

    #graphic_3d_lines(svm, 'svm', train_dims, test_dims)
    graphic_3d_polygon(svm, 'svm', train_dims, test_dims)
    #hex_map(svm, 'svm', train_dims, test_dims)

if (__name__ == "__main__"):
    main()
