# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Create Time: 20170519
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter twelve
##########################

import os
import struct
import numpy as np

def load_mnist(path, kind='train'):
    ''' Load MNIST data from path'''
    labels_path = os.path.join(path,
                              '{0}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(path,
                               '{0}-images-idx3-ubyte'.format(kind))

    while open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.unit8)

    while open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.unit8).reshape(len(labels), 784)

    return images, labels
