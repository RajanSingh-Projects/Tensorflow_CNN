#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 01:52:40 2017

@author: mathcode
"""
import os
import struct
import numpy as np
#os.chdir('/home/mathcode/Tensorflow_Practice/')

def readmnist(dataset = "training", path = "."):
    if dataset == "training":
        fimg = os.path.join(path, "train-images.idx3-ubyte")
        flbl = os.path.join(path, "train-labels.idx1-ubyte")
    elif dataset == "testing":
        fimg = os.path.join(path, "t10k-images.idx3-ubyte")
        flbl = os.path.join(path, "t10k-labels.idx1-ubyte")
    else:
        print "Seriously, are you new to Machine Learning?"

    with open(fimg, "rb") as fimage:
        magic, n_img, nrows, ncols = struct.unpack('>IIII', fimage.read(16))
        images = np.fromfile(fimage, dtype = np.int8).reshape(n_img,
                            nrows, ncols)
    with open(flbl, "rb") as flabel:
        magic, n_img = struct.unpack('>II', flabel.read(8))
        labels = np.fromfile(flabel, dtype = np.int8)
    return images, labels



        
