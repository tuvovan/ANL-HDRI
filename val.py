import os
import cv2
import sys
import glob
import time
import math
import argparse
import numpy as np
import tensorflow as tf 
import generate_HDR_dataset

from HDR import *
from PIL import Image
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Concatenate, Conv2D, Input

def get_test_data(images_path):
    file1 = open(os.path.join(images_path, 'input_exp.txt'), 'r') 
    Lines = file1.readlines() 
    t = [float(k) for k in Lines]

    list_file = sorted(glob.glob(os.path.join(images_path, '*.tif')))
    file1 = open(os.path.join(images_path, 'input_exp.txt'), 'r') 
    Lines = file1.readlines() 
    batch_np = np.zeros([3, 256, 256, 6])
    list_file = sorted(glob.glob(os.path.join(images_path, '*.tif')))
    for j, f in enumerate(list_file):
        ldr = (cv2.imread(f)).astype(np.float32)
        ldr = ldr / 255.0
        ldr = cv2.resize(ldr, (256, 256))

        hdr = ldr**2.2 / (2**t[j])

        X = np.concatenate([ldr, hdr], axis=-1)
        X = np.expand_dims(X, axis=0)
        batch_np[j,:,:,:] = X
    imgs_np = np.expand_dims(batch_np, axis=0)
    return imgs_np


def run(config, model):
    MU = 5000.0
    SDR = get_test_data(config.test_path)

    rs = model.predict(SDR)
    out = rs[0]
    out = tf.math.log(1 + MU * out) / tf.math.log(1 + MU)

    cv2.imwrite(os.path.join(config.test_path, 'hdr.jpg'), np.uint8(out*255))