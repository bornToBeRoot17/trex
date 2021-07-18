import numpy as np
from PIL import Image
from PIL import ImageFilter

from util import *
from preprocessing import *

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

def image_parser_blur(img_path,dim,resize_dim):
    import cv2

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ksize = (10,10)
    img = cv2.blur(img, ksize)
    w, h = img.shape

    TMs = []
    TM  = [ [ 0 for i in range(h) ] for j in range(w) ]
    TM  = np.abs(255-img)
    TMs.append(TM)

    return TMs

# Converts an image to a TM
def image_parser(img_path,dim,resize_dim):
    import cv2

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = img.shape

    TMs = []
    # Create a matrix for storing the TM
    #     - Each position holds normalized number of bytes transmitted from TM[line] to TM[col]
    TM = [ [ 0 for i in range(h) ] for j in range(w) ]

    # for new_dim in resize_dim:
    #     if (w in dim and w != new_dim and new_dim != 0):
    #         img = cv2.resize(img, (new_dim,new_dim))
    #         w, h = img.shape
    #
    #         TM = [ [ 0 for i in range(h) ] for j in range(w) ]

    TM = np.abs(255-img)

    #for i in range(w):
    #    for j in range(h):
    #        TM[i][j]=(255-img[i][j])   # Normal colors
    #        #TM[i][j]=(255-pix[i,j][0])   # Normal colors
    #        #TM[i][j]=(pix[i,j][0])       # Invert colors

    TMs.append(TM)

    return TMs

# Converts an image to a TM
def image_parser_bkp(img_path,dim,resize_dim):
    img = Image.open(img_path)
    #img = img.filter(ImageFilter.BLUR)
    #img = img.filter(ImageFilter.SMOOTH)
    #img = img.filter(ImageFilter.SMOOTH_MORE)
    w, h = img.size

    TMs = []
    # Create a matrix for storing the TM
    #     - Each position holds normalized number of bytes transmitted from TM[line] to TM[col]
    TM = [ [ 0 for i in range(h) ] for j in range(w) ]

    #dim = 256
    #newDim = 64
    #if (resize_dim):
    for new_dim in resize_dim:
        if (w in dim and w != new_dim and new_dim != 0):
            img = img.resize((new_dim,new_dim))
            #img = img.filter(ImageFilter.BLUR)
            #img = img.filter(ImageFilter.SMOOTH)
            #img = img.filter(ImageFilter.SMOOTH_MORE)
            w, h = img.size
            #img = img.resize((newDim,newDim),Image.ANTIALIAS)

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

def image_parser_lenet(img_path,dim,resize_dim):
    import cv2

    TMs = []

    #print(img_path)
    tm = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #.astype(np.int32)
    tm = cv2.resize(tm, (128,128)) #, interpolation=cv2.INTER_NEAREST)

    #for i in range(tm.shape[0]):
    #    for j in range(tm.shape[1]):
    #        aux = float(tm[i][j]/255)
    #        if (aux < 0.7):
    #            tm[i][j] = 1
    #        else:
    #            tm[i][j] = 0

    #if (np.sum(tm) > tm.shape[0]*tm.shape[1]/2):
    #    tm = 1 - tm

    TMs.append(tm)

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

def image_crop(img_path,dim,resize_dim):
    import cv2

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = img.shape

    TMs = []

    new_dim = int(resize_dim[0])
    if (new_dim == 0): new_dim = w

    if (w != new_dim):
        left   = int((w - new_dim)/2)
        top    = int((h - new_dim)/2)
        right  = int((w + new_dim)/2)
        bottom = int((h + new_dim)/2)

        img = img[top:bottom, left:right]
        w, h = img.shape

    TM = [ [ 0 for i in range(h) ] for j in range(w) ]
    TM = np.abs(255-img)

    #pix = img.load()
    #for i in range(h):
    #    for j in range(w):
    #        TM[i][j]=(255-pix[i,j])
    #        #TM[i][j]=(pix[i,j])

    TMs.append(TM)

    return TMs

#def image_crop(img_path,dim,resize_dim):
#    img = Image.open(img_path)
#    #img = img.filter(ImageFilter.BLUR)
#    #img = ImageOps.invert(img)
#    w, h = img.size
#
#    TMs = []
#
#    new_dim = int(resize_dim[0])
#    if (new_dim == 0): new_dim = w
#
#    if (w != new_dim):
#        left   = (w - new_dim)/2
#        top    = (h - new_dim)/2
#        right  = (w + new_dim)/2
#        bottom = (h + new_dim)/2
#
#        img = img.crop((left, top, right, bottom))
#        w, h = img.size
#
#    TM=[ [ 0 for i in range(new_dim) ] for j in range(new_dim) ]
#
#    pix = img.load()
#    for i in range(new_dim):
#        for j in range(new_dim):
#            TM[i][j]=(255-pix[i,j])
#            #TM[i][j]=(pix[i,j])
#
#    TMs.append(TM)
#
#    return TMs


def image_replication(img_path,dim,resize_dim):
    import scipy.misc
    #scipy.misc.imsave('outfile.jpg', image_array)

    img = Image.open(img_path)
    w, h = img.size

    TMs = []

    TM=[ [ 0 for i in range(1024) ] for j in range(1024) ]
    img_orig = [ [ 0 for i in range(h) ] for j in range(w) ]

    pix = img.load()
    for i in range(w):
        for j in range(h):
            img_orig[i][j]=(255-pix[i,j])

    for i in range(1024):
        idx_lin_img = i % 32
        for j in range(1024):
            idx_col_img = j % 32
            TM[i][j] = img_orig[idx_lin_img][idx_col_img]

    #scipy.misc.imsave('test.png', TM)
    #pil_img = Image.fromarray(TM)
    #pil_img.save('test.png')

    TMs.append(TM)
    #exit(-1)

    return TMs

def image_vectorize(img_path, dim, resize_dim):
    import cairosvg

    new_dim = int(resize_dim[0])
    dim = int(dim[0])

    if (new_dim != 0 and dim != new_dim):
        cairosvg.svg2png(url=img_path,write_to='file.png',output_width=new_dim,output_height=new_dim)
    else:
        cairosvg.svg2png(url=img_path,write_to='file.png')

    img = Image.open('file.png')
    w, h = img.size

    TMs = []

    # Create a matrix for storing the TM
    #     - Each position holds normalized number of bytes transmitted from TM[line] to TM[col]
    TM = [ [ 0 for i in range(h) ] for j in range(w) ]

    pix=img.load()
    for i in range(w):
        for j in range(h):
            TM[i][j]=(255-pix[i,j][0])   # Normal colors

    TMs.append(TM)

    return TMs

def image_reorder(img_path,dim,resize_dim):
    img = Image.open(img_path)
    w, h = img.size

    TMs = []

    for new_dim in resize_dim:
        if (w in dim and w != new_dim and new_dim != 0):
            img = img.resize((new_dim,new_dim))
            w, h = img.size

    # Create a matrix for storing the TM
    #     - Each position holds normalized number of bytes transmitted from TM[line] to TM[col]
    TM = [ [ 0 for i in range(h) ] for j in range(w) ]

    pix=img.load()
    for i in range(w):
        for j in range(h):
            TM[i][j]=(255-pix[i,j])   # Normal colors

    for i in range(w):
        TM[i].sort()
        TM[i] = TM[i][::-1]

    #for i in range(w-1):
    #    for j in range(i,w):
    #        if (TM[j][0] > TM[i][0]):
    #            aux = TM[i][0]
    #            TM[i][0] = TM[j][0]
    #            TM[j][0] = aux

    TMs.append(TM)

    return TMs

def image_black(img_path,dim,resize_dim):
    import cv2

    TMs = []

    tm = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)

    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            tm[i][j] = int(float(tm[i][j]/255)/0.7)

    fname = img_path.split('/')[-3]+'_resize_'+img_path.split('/')[-1]
    fname = fname[:-4]+'_2.png'
    cv2.imwrite('./imgs/'+fname, tm)


    if (np.sum(tm) > tm.shape[0]*tm.shape[1]/2):
        tm = 1 - tm

    fname = img_path.split('/')[-3]+'_resize_'+img_path.split('/')[-1]
    fname = fname[:-4]+'_3.png'
    cv2.imwrite('./imgs/'+fname, tm)

    if (resize_dim[0] != 0):
        tm = cv2.resize(tm.astype('int32'), (resize_dim[0],resize_dim[0]))

        fname = img_path.split('/')[-3]+'_resize_'+img_path.split('/')[-1]
        cv2.imwrite('./imgs/'+fname, tm)

    TMs.append(tm)

    return TMs

def image_split(img_path,dim,resize_dim):
    import cv2

    TMs = []

    kernel_dim = 32

    tm = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    if (tm.shape[0] != kernel_dim):
        for i in range(0,tm.shape[0],kernel_dim):
            for j in range(0,tm.shape[1],kernel_dim):
                aux_tm = tm[i:i+kernel_dim,j:j+kernel_dim]
                for k in range(aux_tm.shape[0]):
                    for l in range(aux_tm.shape[1]):
                        aux = float(aux_tm[k][l]/255)
                        if (aux < 0.7):
                            aux_tm[k][l] = 1
                        else:
                            aux_tm[k][l] = 0
                if (np.sum(aux_tm) > aux_tm.shape[0]*aux_tm.shape[1]/2):
                    aux_tm = 1 - aux_tm
                TMs.append(aux_tm)
    else:
        for i in range(tm.shape[0]):
            for j in range(tm.shape[1]):
                aux = float(tm[i][j]/255)
                if (aux < 0.7):
                    tm[i][j] = 1
                else:
                    tm[i][j] = 0

        if (np.sum(tm) > tm.shape[0]*tm.shape[1]/2):
            tm = 1 - tm

        TMs.append(tm)

    return TMs


def image_parser_interpolation(img_path,dim,resize_dim):
    import cv2

    TMs = []
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #img = img.filter(ImageFilter.BLUR)
    #img = img.filter(ImageFilter.SMOOTH)
    #img = img.filter(ImageFilter.SMOOTH_MORE)
    newDim = 100

    TM = [ [ 0 for i in range(newDim) ] for j in range(newDim) ]

    if (dim[0] != newDim):
        img = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        dim_resize = (newDim, newDim)

        img = cv2.resize(img, dim_resize)

        fname = img_path.split('/')[-3]+'_'+img_path.split('/')[-1]
        cv2.imwrite('./imgs/'+fname, img)
        #print(img.shape[0], img.shape[1])

    for i in range(newDim):
        for j in range(newDim):
            TM[i][j]=(255-img[i][j])   # Normal colors
            #TM[i][j]=(255-pix[i,j][0])   # Normal colors
            #TM[i][j]=(pix[i,j][0])       # Invert colors

    TMs.append(TM)

    return TMs

def image_parser_dilation(img_path,dim,resize_dim):
    import cv2

    TMs = []
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #TM = [ [ 0 for i in range(newDim) ] for j in range(newDim) ]


    count_black = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j]=(255-img[i][j]) # Normal colors
            #TM[i][j]=(pix[i,j][0])   # Invert colors

            if (float(img[i][j]/255) > 0.7): count_black += 1

    kernel = np.ones((1,1), np.uint8)
    if (count_black > img.shape[0]*img.shape[1]/2):
        kernel = np.zeros((1,1), np.uint8)

    img = cv2.dilate(img, kernel, iterations=1)
    # fname = img_path.split('/')[-3]+'_dilation_'+img_path.split('/')[-1]
    # cv2.imwrite('./imgs/'+fname, img)

    newDim = 100
    if (dim[0] != newDim):
        dim_resize = (newDim, newDim)

        img = cv2.resize(img, dim_resize)
        # fname = img_path.split('/')[-3]+'_resize_'+img_path.split('/')[-1]
        # cv2.imwrite('./imgs/'+fname, img)

        #print(img.shape[0], img.shape[1])


    TMs.append(img)

    return TMs


def image_read_dilation(img_path,dim,resize_dim):
    import cv2

    #print(img_path)

    TMs = []
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    #TM = [ [ 0 for i in range(newDim) ] for j in range(newDim) ]

    dim_imgs = 'tr128_r0_ts256_r128'
    #dim_imgs = 'tr64_r0_ts256_r64'
    #preproc=globals()[get_definitions("Functions","preprocessing",dim_imgs)]
    #img = preproc.__call__(img)

    count = np.sum(img)

    if (count > img.shape[0]*img.shape[1]*255*0.3):
        img -= 255
        img *= -1

    kernel_size=(2,2)
    kernel = np.ones(kernel_size, np.uint8)

    #img = cv2.dilate(img, kernel, iterations=2)
    #if ('fig10.png' in img_path):
    #    fname = img_path.split('/')[-3]+'_dilation_'+img_path.split('/')[-1]
    #    cv2.imwrite('./imgs/'+fname, img)

    dim    = int(dim[0])
    newDim = int(resize_dim[0])
    if (newDim != 0 and dim != newDim):
        dim_resize = (int(newDim), int(newDim))
        img = cv2.dilate(img, kernel, iterations=2)
        img = cv2.resize(img, dim_resize)
        if ('fig10.png' in img_path):
            fname = img_path.split('/')[-3]+'_resize_'+img_path.split('/')[-1]
            cv2.imwrite('./imgs/'+fname, img)
    else:
        img = cv2.dilate(img, kernel, iterations=1)
        if ('fig10.png' in img_path):
            fname = img_path.split('/')[-3]+'_dilation_'+img_path.split('/')[-1]
            cv2.imwrite('./imgs/'+fname, img)

    #    #print(img.shape[0], img.shape[1])


    #for i in range(img.shape[0]):
    #    for j in range(img.shape[1]):
    #        print(img[i][j],end=' ')
    #    print()

    TMs.append(img)

    return TMs

def gaussian_pyramid(img_path,dim,resize_dim):
    import cv2
    import math

    TMs = []
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    dim_imgs = 'tr128_r0_ts512_r128'
    dim = dim[0]
    newDim = int(resize_dim[0])

    if (newDim != 0 and dim != newDim):
        dim_resize = (int(newDim), int(newDim))
        n_down = math.log(dim/newDim,2)
        for i in ramge(n_down):
            img = cv2.pyrDown(img)
        if ('fig10.png' in img_path):
          fname = img_path.split('/')[-3]+'_resize_'+img_path.split('/')[-1]
          cv2.imwrite('./imgs/'+fname, img)

    TMs.append(img)

    return TMs

def normal_parser(img_path,dim,resize_dim):
    import cv2

    TMs = []
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))

    TMs.append(img)

    return TMs
