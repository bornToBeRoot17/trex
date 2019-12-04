import numpy as np
from PIL import Image
from PIL import ImageFilter
#import cv2 as cv

#from scipy import misc
#import png2svg


N_REM = 5

def remove_no_middle_img(img, w, h):
    for i in range(h):
        start = i - N_REM/2
        end   = i + N_REM/2
        for j in range(0,start):
            img[i,j] = (255,255,255)

        for j in range(end,w):
            img[i,j] = (255,255,255)

    return img

def remove_middle_img(img, w, h):
    for i in range(h):
        for j in range(N_REM):
            idx = i + j
            if (idx > 0 and idx < w):
                img[i,idx] = (255,255,255)
            idx = i - j
            if (idx > 0 and idx < w):
                img[i,idx] = (255,255,255)

    return img

# Converts an image to a TM
def image_parser(img_path,dim,resize_dim):
    img = Image.open(img_path)
    #img = img.filter(ImageFilter.BLUR)
    #img = img.filter(ImageFilter.SMOOTH)
    #img = img.filter(ImageFilter.SMOOTH_MORE)
    w, h = img.size

    TMs = []

    #dim = 256
    #newDim = 64
    #if (w != newDim):
    for new_dim in resize_dim:
        if (w in dim and w != new_dim and new_dim != 0):
            img = img.resize((new_dim,new_dim))
            #img = img.filter(ImageFilter.BLUR)
            #img = img.filter(ImageFilter.SMOOTH)
            #img = img.filter(ImageFilter.SMOOTH_MORE)
            w, h = img.size
            #img = img.resize((newDim,newDim),Image.ANTIALIAS)

    # Create a matrix for storing the TM
    #     - Each position holds normalized number of bytes transmitted from TM[line] to TM[col]
    TM = [ [ 0 for i in range(h) ] for j in range(w) ]

    pix=img.load()
    #pix = remove_middle_img(pix, w, h)
    #pix = remove_no_middle_img(pix, w, h)
    for i in range(w):
        for j in range(h):
            TM[i][j]=(255-pix[i,j])   # Normal colors
            #TM[i][j]=(255-pix[i,j][0])   # Normal colors
            #TM[i][j]=(pix[i,j][0])       # Invert colors

    TMs.append(TM)

    return TMs

def image_parser_pooling(TM, img_path):
    img = Image.open(img_path)

    w, h = img.size

    if (w == 256):
        img = img.resize((64,64))
        w, h = img.size
        pix = img.load()

        stride = 2

        # Using R channel only. Fix it later, maybe
        #newImg = [[[0,0,0]*w]*h]
        newImg = [[0]*(w/stride)]*(h/stride)

#        print(len(newImg), len(newImg[0]))
#        print(newImg)
#        return

        currW = 0
        for i in range(0,w,stride):
            currH = 0
            for j in range(0,h,stride):
                aux = np.zeros(stride*stride, dtype=np.int)

                aux[0] = 255 - pix[i,j][0]
                aux[1] = 255 - pix[i,j+1][0]
                aux[2] = 255 - pix[i+1,j][0]
                aux[3] = 255 - pix[i+1,j+1][0]

                newImg[currW][currH] = np.amax(aux)

                currH += 1
            currW += 1

        TM = newImg
    else:
        newDim = 32

        img = img.resize((newDim,newDim))
        w, h = img.size
        #img = img.resize((newDim,newDim),Image.ANTIALIAS)
        pix = img.load()

        for i in range(w):
            for j in range(h):
                TM[i][j]=(255-pix[i,j][0])

def image_crop(img_path):
    img = Image.open(img_path)
    #img = img.filter(ImageFilter.BLUR)
    #img = ImageOps.invert(img)
    w, h = img.size

    new_dim = 128

    if (w != new_dim):
        left   = (w - new_dim)/2
        top    = (h - new_dim)/2
        right  = (w + new_dim)/2
        bottom = (h + new_dim)/2

        img = img.crop((left, top, right, bottom))
        w, h = img.size

    TM=[ [ 0 for i in range(h) ] for j in range(w) ]

    pix = img.load()
    for i in range(w):
        for j in range(h):
            TM[i][j]=(255-pix[i,j][0])
            #TM[i][j]=(pix[i,j][0])

    return TM
