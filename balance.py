from __future__ import print_function
from sys import argv, exit


N_MAX = 208
#N_MAX = 50

def parse_classes(dig):
    sp = dig.split("=")
    l = sp[0]
    app = sp[1]
    class_num = sp[2]
    vls = sp[3].strip(" ").split(",")
    lst = []

    for i in range(len(vls)):
        if (len(vls[i].split("-")) == 2):
            for j in range(int(vls[i].split("-")[0]), int(int(vls[i].split("-")[1]) + 1)):
                lst.append(j)
        else:
            lst.append(int(vls[i]))

    return [l, app, class_num, lst]

def main(argv):
    if (len(argv) != 2):
        print("Usage:", argv[0], "<definitions.ini>")
        exit(-1)

    defFile = argv[1]

    try:
        f = open(defFile,'r')
        lstF = f.readlines()
        f.close()
    except:
        print("Definition file doesn't exists.")
        print("Usage:", argv[0], "<definitions.ini>")
        exit(-1)

    auxF = []
    for i in range(len(lstF)):
        if (lstF[i][0] == 'l'):
            auxF.append(lstF[i])

    for i in range(len(auxF)):
        l, app, class_num, lst = parse_classes(auxF[i])
        #print(len(lst))
        #continue
        lst = lst[:N_MAX]
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

        print(l + '=' + app + '=' + class_num + '=',end="")
        for j in range(len(res)):
            print(res[j] + ',',end="")
        print('\n')


    return 0

if (__name__ == "__main__"):
    main(argv)
