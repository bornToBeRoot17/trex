#!/usr/bin/env python3
#! -*- encode: utf-8 -*-

from sys import argv, exit
import svgutils as su
import os
import re
import shutil
from cairosvg import svg2png


path = "./dataset/"
dest = "./dataset_scale/"

def get_dirs():
    dirs = []

    for r,d,f in os.walk(path):
        for dir in d:
            if ("_svg" in dir):
                dirs.append(dir)

    return dirs

def scaling(dirs):
    if (os.path.exists(dest)):
        shutil.rmtree(dest)
    os.mkdir(dest)
    for dir in dirs:
        new_dir = dir.split("_")[0] + '_' + dir.split("_")[1]
        if (".unscr" in dir):
            new_dir += ".unscr"

        os.mkdir(dest+new_dir)
        os.mkdir(dest+new_dir+"/tm")
        destPath = dest+new_dir+"/tm/"

        origPath = path + dir + "/tm/"
        for r,d,f in os.walk(origPath):
            for file in f:
                original = su.transform.fromfile(origPath+file)
                original_width = float(re.sub('[^0-9]','', original.width))
                original_height = float(re.sub('[^0-9]','', original.width))

                scale = 480/int(original.width[:3])

                scaled = su.transform.SVGFigure(original_width * scale, original_height * scale,)
                svg = original.getroot()
                svg.scale_xy(scale, scale)
                scaled.append(svg)
                scaled.save(destPath+file)
                svg2png(open(destPath+file, 'rb').read(), write_to=open(destPath+file[:-4]+".png", 'wb'))
                os.remove(destPath+file)
                os.remove(origPath+file)

def main(argv):
    dirs = get_dirs()
    scaling(dirs)


if (__name__ == "__main__"):
    main(argv)
