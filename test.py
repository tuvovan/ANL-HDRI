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
from val import run
from PIL import Image
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Concatenate, Conv2D, Input


def get_test_data(images_path):
    file1 = open(os.path.join(images_path, 'exposure.txt'), 'r') 
    Lines = file1.readlines() 
    batch_np = np.zeros([3, 768, 1024, 6])
    list_file = sorted(glob.glob(os.path.join(images_path, '*.tif')))
    for j, f in enumerate(list_file):
        ldr = (cv2.imread(f, -1)).astype(np.float32)
        ldr = ldr / 65535.0
        ldr = cv2.resize(ldr, (1024, 768))

        hdr = ldr**2.2 / (2**t[j])

        X = np.concatenate([ldr, hdr], axis=-1)
        X = np.expand_dims(X, axis=0)
        batch_np[j,:,:,:] = X
    imgs_np = np.expand_dims(batch_np, axis=0)
    return imgs_np


def run(config, model):
    SDR = get_test_data(config.test_path)
    print(SDR.shape)
    rs = model.predict(SDR)
    out = rs[0]
    tonemap = cv2.createTonemapReinhard()
    out = tonemap.process(out.copy())
    cv2.imwrite(os.path.join(config.test_path, 'hdr.jpg'), np.uint8(out*255))



if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--test_path', type=str, default="Test/EXTRA/001/")
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--weight_test_path', type=str, default= "weights/best.h5")
	parser.add_argument('--filter', type=int, default= 32)
	parser.add_argument('--attention_filter', type=int, default= 64)
	parser.add_argument('--kernel', type=int, default= 3)
	parser.add_argument('--encoder_kernel', type=int, default= 3)
	parser.add_argument('--decoder_kernel', type=int, default= 4)
	parser.add_argument('--triple_pass_filter', type=int, default= 256)

	config = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

	model_x = NHDRRNet(config)
	x = Input(shape=(3, 768, 1024, 6))
	out = model_x.main_model(x)
	model = Model(inputs=x, outputs=out)
	model.load_weights(config.weight_test_path)
	model.summary()

	run(config, model)
