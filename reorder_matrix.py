import numpy as np
from PIL import Image
from PIL import ImageFilter
from sys import argv, exit


def image_reorder(img_path):
    img = Image.open(img_path)
    w, h = img.size

    # Create a matrix for storing the TM
    #     - Each position holds normalized number of bytes transmitted from TM[line] to TM[col]
    TM = [ [ 0 for i in range(h) ] for j in range(w) ]

    pix=img.load()
    for i in range(w):
        for j in range(h):
            TM[i][j]=(255-pix[i,j])   # Normal colors

    TM = np.asmatrix(TM)
    TM = TM.sort()

    image = Image.fromarray(TM.astype('uint8'), 'RGB')
    image.save('test.png')


def main(argv):
    if (len(argv) != 2):
        print('Usage', argv[0], '<img.png>')
        exit(-1)

    img_file = argv[1]
    image_reorder(img_file)

    return 0

if (__name__ == '__main__'):
    main(argv)
