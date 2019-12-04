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
#methods = ['DCTraCS_ULBP', 'DCTraCS_RLBP', 'Eerman', 'Fahad', 'Soysal']
methods = ['DCTraCS_RLBP']
#classifiers = ['svm', 'rf']
classifiers = ['svm']
metric = "best"
# key_word = ["Accuracy", "Recall", "F1 Score"]
key_word = "Recall"

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
                    if (key_word in line and method in line and '%' in line):
                        aux_line_method.append(line)

                for line in aux_line_method:
                    idx = line.find('%') - 6
                    prcnt = line[idx:idx+6]
                    if (prcnt[0] == ' '): prcnt = prcnt[1:]
                    if (prcnt[0] == ':'): prcnt = prcnt[2:]
                    prcnt = float(prcnt)

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

def main():
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
    #heatmap(rf, "rf", train_dims, test_dims)

    #idx = methods.index('DCTraCS_RLBP')
    #print(idx)

    #print(key_word)
    #print(svm[idx])

    #graphic_3d_2(svm, 'svm', train_dims, test_dims)

def heatmap(data, clf, train_dims, test_dims):
    if (clf == 'svm'):
        str_clf = 'SVM'
    else:
        str_clf = 'Random Forest'

    sns.set(font_scale=2.)
    #cmap = sns.cubehelix_palette(n_colors=100, dark=0, light=1, reverse=False, as_cmap=True)
    cmap = sns.dark_palette("white", 100, as_cmap=True, reverse=True)

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

    return 0

def graphic_3d():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, .25)
    Y = np.arange(-5, 5, .25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    Gx, Gy = np.gradient(Z) # gradients with respect to x and y
    G = (Gx**2+Gy**2)**.5  # gradient magnitude
    N = G/G.max()  # normalize 0..1
    surf = ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1,
            facecolors=cm.jet(N),
            linewidth=0, antialiased=False, shade=False
    )
    plt.savefig("img_results/test.pdf", bbox_inches='tight')
    #plt.show()

def graphic_3d_2(values, str_clf, train_dims, test_dims):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    cmap = sns.dark_palette("white", 100, as_cmap=True, reverse=True)
    ax = fig.add_subplot(111, projection='3d')

    x = np.asarray(train_dims)
    y = np.asarray(test_dims)
    z = np.asarray(values[0][0])
    #c = np.asarray(values[0][0])
    print(x.shape,y.shape,z.shape) #,c.shape)

    ax.set_xticks(train_dims)
    ax.set_yticks(test_dims)

    img = ax.scatter(x, y, z)
    fig.colorbar(img)
    plt.savefig("img_results/test.pdf", bbox_inches='tight')

if (__name__ == "__main__"):
    main()
