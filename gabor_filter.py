# -*- encoding: utf-8 -*-

import numpy as np
import cv2


def build_filters():
    filters = []
    ksize = 5
    for theta in np.arange(0, np.pi, np.pi/8):
        for lamda in np.arange(0, np.pi, np.pi/9):
            kern = cv2.getGaborKernel((ksize,ksize), 1.0, theta, lamda, 0.50, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)

    print(len(filters))

    return filters

def main():
    filters = build_filters()

    return 0

if (__name__ == '__main__'):
    main()
