import numpy as np
from PIL import Image
from PIL import ImageFilter

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

def image_crop(img_path,dim,resize_dim):
    img = Image.open(img_path)
    #img = img.filter(ImageFilter.BLUR)
    #img = ImageOps.invert(img)
    w, h = img.size

    new_dim = 32

    if (w != new_dim):
        left   = (w - new_dim)/2
        top    = (h - new_dim)/2
        right  = (w + new_dim)/2
        bottom = (h + new_dim)/2

        img = img.crop((left, top, right, bottom))
        w, h = img.size

    TM=[ [ 0 for i in range(new_dim) ] for j in range(new_dim) ]

    pix = img.load()
    for i in range(new_dim):
        for j in range(new_dim):
            TM[i][j]=(255-pix[i,j])
            #TM[i][j]=(pix[i,j])

    return TM


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
            aux = float(tm[i][j]/255)
            if (aux < 0.7):
                tm[i][j] = 1
            else:
                tm[i][j] = 0

    if (np.sum(tm) > tm.shape[0]*tm.shape[1]/2):
        tm = 1 - tm

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
