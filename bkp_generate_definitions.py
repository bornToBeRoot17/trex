#!/usr/bin/env python

from sys import argv, exit

def generate_sufix(dims, file_sufix):
    sufix = "tr" + dims["train"][0]
    for j in range(1,len(dims["train"])):
        sufix += "_tr" + dims["train"][j]
    sufix += "_r" + dims["train_resize"][0]
    for j in range(1,len(dims["train_resize"])):
        sufix += "_r" + dims["train_resize"][j]
    sufix += "_ts" + dims["test"][0]
    for j in range(1,len(dims["test"])):
        sufix += "_ts" + dims["test"][j]
    sufix += "_r" + dims["test_resize"][0]
    for j in range(1,len(dims["test_resize"])):
        sufix += "_r" + dims["test_resize"][j]

    sufix += file_sufix

    return sufix

def parse_classes(imgs_lst):
    vls = imgs_lst.strip(" ").split(",")
    lst = []

    for i in range(len(vls)):
        if (len(vls[i].split("-")) == 2):
            for j in range(int(vls[i].split("-")[0]), int(int(vls[i].split("-")[1]) + 1)):
                lst.append(j)
        else:
            lst.append(int(vls[i]))

    return lst

def join_classes(lst):
    res = []

    j = 0
    while (j < len(lst)):
        n = lst[j]
        ok = 0
        while (j+1 < len(lst) and lst[j]+1 == lst[j+1]):
            j += 1
            ok = 1

        if (ok):
            res.append(str(n)+'-'+str(lst[j]))
        else:
            res.append(str(n))
        j += 1

    result = ''
    for j in range(len(res)):
        result += res[j] + ','
    result = result[:-1]

    return result

def count_imgs(train_apps):
    n_max = 999999

    for app in train_apps:
        lst = parse_classes(images[app])
        print(app, len(lst))
        if (len(lst) < n_max):
            n_max = len(lst)

    return n_max

def generate_definitions(dims, sufix, num_imgs, parser, sck_path, classes, dataset):
    with open("definitions.ini",'r') as f:
        lstF = f.readlines()
        f.close()

    # Configuring [Paths]
    sckit_files_path = sck_path

    # Configuring [Applications]
    used_apps = []
    line_used_apps = []
    count = 1

    #for train_dim in dims["train"]:
    #    for app_train in apps[train_dim]:
    #        ok = 0
    #        for test_dim in dims["test"]:
    #            for app_test in apps[test_dim]:
    #                if (app_train == app_test and app_train not in classes):
    #                    classes.append(app_train)
    #                    ok = 1
    #                    break
    #            if (ok): break

    for dim in dims["train"]:
        for app in classes:
            #if (app not in apps[dim]):
            #    continue
            app_name = app + '_' + dim
            used_apps.append(app_name)
            line = "app" + str(count) + '=' + app_name + '=' + app_name + "/\n"
            line_used_apps.append(line)
            count += 1

    train_apps = used_apps[:]
    num_train_apps = len(used_apps)

    for dim in dims["test"]:
        for app in classes:
            #if (app not in apps[dim]):
            #    continue
            app_name = app + '_' + dim
            #if (app_name in used_apps): continue
            used_apps.append(app_name)
            line = "app" + str(count) + '=' + app_name + '=' + app_name + "/\n"
            line_used_apps.append(line)
            count += 1

    #n_max = count_imgs(used_apps)
    #exit(-1)

    # Configuring [Classes]
    line_classes = []
    for j in range(len(used_apps)):
        line = 'l' + str(j+1) + '=' + used_apps[j] + '='
        for k in range(len(classes)):
            if (classes[k] in used_apps[j]):
                line += str(k+1)

        line += '='
        for k in range(num_imgs):
            line += str(k) + ','
        line += str(num_imgs) + '\n'
        #if (used_apps[j] in images_balanced):
        #    line += '=' + images_balanced[used_apps[j]] + '\n'
        #else:
        #    line += '=' + images[used_apps[j]] + '\n'

        line_classes.append(line)

    i = 0
    while("acquiring:" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + parser + '\n'

    while ("sckit_files_path:" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + ' ' + sckit_files_path + '\n'

    while ("img_path:" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + ' ' + dataset + '\n'

    while ("sckit_files_path_out:" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + ' ' + sckit_files_path + '\n'

    while ("img_train_size:" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + " [" + dims["train"][0]
    for j in range(1,len(dims["train"])):
        lstF[i] += ',' + dims["train"][j]
    lstF[i] += "]\n"

    while ("img_test_size:" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + " [" + dims["test"][0]
    for j in range(1,len(dims["test"])):
        lstF[i] += ',' + dims["test"][j]
    lstF[i] += "]\n"

    while ("img_train_resize:" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + " [" + dims["train_resize"][0]
    for j in range(1,len(dims["train_resize"])):
        lstF[i] += ',' + dims["train_resize"][j]
    lstF[i] += "]\n"

    while ("img_test_resize:" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + " [" + dims["test_resize"][0]
    for j in range(1,len(dims["test_resize"])):
        lstF[i] += ',' + dims["test_resize"][j]
    lstF[i] += "]\n"

    while ("number=" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + str(len(used_apps)) + '\n'
    for line in line_used_apps:
        lstF[i] += line

    while ("number_of_lines=" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + str(len(used_apps)) + '\n'
    for j in range(len(line_classes)):
        lstF[i] += line_classes[j]

    while ("number_of_train_classes=" not in lstF[i]): i += 1
    lstF[i] = lstF[i][:-1] + str(num_train_apps) + '\n'

    return lstF

def write_definitions(lstF, dims, sufix):
    file_name = "definitions_" + sufix + ".ini"

    with open(file_name,'w') as f:
        for i in range(len(lstF)):
            f.write(lstF[i])
        f.close()

def update_dims(dims, dim_lst, key):
    if ('[' in dim_lst): dim_lst = dim_lst.replace('[','')
    if (']' in dim_lst): dim_lst = dim_lst.replace(']','')
    if (',' in dim_lst): dim_lst = dim_lst.split(',')
    else: dim_lst = [dim_lst]

    dims[key] = []
    for dim in dim_lst:
        dims[key].append(dim)

    return dims

def main(argv):
    if (len(argv) != 10):
        print("Usage:", argv[0], "<[train_dim]> <[train_resize_dim]> <[test_dim]> <[test_resize_dim]> <num_imgs> <parser> <sck_path> <classes> <dataset>")
        exit(-1)

    train_dim        = argv[1]
    train_resize_dim = argv[2]
    test_dim         = argv[3]
    test_resize_dim  = argv[4]
    num_imgs         = int(argv[5])
    parser           = argv[6]
    sck_path         = argv[7]
    classes          = argv[8].split(',')
    dataset          = argv[9]
    #file_sufix       = argv[10]
    file_sufix       = ''

    dims = {}

    dims = update_dims(dims, train_dim,        "train")
    dims = update_dims(dims, train_resize_dim, "train_resize")
    dims = update_dims(dims, test_dim,         "test")
    dims = update_dims(dims, test_resize_dim,  "test_resize")

    sufix = generate_sufix(dims, file_sufix)

    lstF = generate_definitions(dims, sufix, num_imgs, parser, sck_path, classes, dataset)
    write_definitions(lstF, dims, sufix)

    return 0

if (__name__ == "__main__"):
    main(argv)
