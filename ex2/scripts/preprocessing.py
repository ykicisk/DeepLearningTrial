#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import argparse as ap
from skimage import io
from skimage.color import rgb2gray
from scipy.misc import imresize
from skimage import transform


def main(src_path, dst_path, resize):
    # make directories
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for gender in ["m", "f"]:
        if not os.path.exists("{}/{}".format(dst_path, gender)):
            os.mkdir("{}/{}".format(dst_path, gender))
    # preprocessing !!
    for line in open("{}/PersonData.txt".format(src_path)):
        sp = line.rstrip().split("\t")
        if len(sp) == 1:
            fname = sp[0]
            fpath = "{}/{}".format(src_path, fname)
            print fpath
            img = io.imread(fpath)
        else:
            gender = "m" if sp[5] == "2" else "f"
            x_leye, y_leye, x_reye, y_reye = map(int, sp[:4])
            x_center = (x_leye + x_reye) / 2
            y_center = (y_leye + y_reye) / 2
            length = x_reye - x_leye
            x = int(x_center - 1.5 * length)
            y = int(y_center - 1.0 * length)
            w = 3 * length
            h = w
            if length < 0 or x < 0 or y < 0 or w < 0:
                continue
            print "cropping:", x, y, w, h
            cropped_image = img[y:y+h, x:x + w]
            gray_image = rgb2gray(cropped_image)
            resized_image = imresize(gray_image, (resize, resize),
                                     interp='bilinear')
            froot, ext = os.path.splitext(fname)
            fpath = "{}/{}/{}_{}_{}_{}_{}.png".format(dst_path, gender, froot,
                                                    x, y, w, h)
            print "->", fpath
            io.imsave(fpath, resized_image)


if __name__ == "__main__":
    description = """preprocessing for The Image of Groups Dataset[1]
* cropping
* grayscale
* resize
[1] http://chenlab.ece.cornell.edu/people/Andy/ImagesOfGroups.html
    """
    parser = ap.ArgumentParser(description=description,
                               formatter_class=ap.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--src', required=True,
                        help='input image directory')
    parser.add_argument('-d', '--dst', required=True,
                        help='output image directory')
    parser.add_argument('--resize', default=32,
                        help="output resized image size")
    args = parser.parse_args()
    main(args.src, args.dst, args.resize)

