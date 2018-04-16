"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np
import scipy.io

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = 21

IMAGE_DIR = '/media/data/bruppik/deeplab_resnet_test_dataset/images/'
COLOR_MASK_SAVE_DIR = '/media/data/bruppik/deeplab_resnet_test_dataset/color_mask_output/'
MATLAB_SAVE_DIR = '/media/data/bruppik/deeplab_resnet_test_dataset/matlab_files/'
FILELIST = '/media/data/bruppik/deeplab_resnet_test_dataset/filelist.txt'
MODEL_WEIGHTS = '/media/data/bruppik/deeplab_resnet_ckpt/deeplab_resnet.ckpt'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("img_path", type=str,
                        help="Path to the RGB image file.")
    parser.add_argument("save_path", type=str,
                        help="Where to save predicted mask.")
    parser.add_argument("--model_weights", type=str, default=MODEL_WEIGHTS,
                        help="Path to the file with model weights.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def infer_and_save_color_map(img_name):
    """Create the model and start the evaluation process."""
    tf.reset_default_graph()
    
    # Prepare image.
    img = tf.image.decode_png(tf.read_file(IMAGE_DIR + img_name), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, MODEL_WEIGHTS)
    
    # Perform inference.
    preds = sess.run(pred)    
    msk = decode_labels(preds, num_classes=NUM_CLASSES)
    im = Image.fromarray(msk[0])

    # Save the image
    filename, file_extension = os.path.splitext(img_name)
    save_path = COLOR_MASK_SAVE_DIR + filename + '_mask.png'
    if not os.path.exists(COLOR_MASK_SAVE_DIR):
        os.makedirs(COLOR_MASK_SAVE_DIR)
    im.save(save_path)
    
    print('The output file has been saved to {}'.format(save_path))

def infer_absolute_path(img_path, model_weights):
    """Create the model and start the evaluation process."""
    tf.reset_default_graph()
    
    # Prepare image.
    img = tf.image.decode_png(tf.read_file(img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, model_weights)
    
    # Perform inference.
    # preds is in our test case an object of type numpy.ndarray
    # with preds.shape = (1, image_height, image_width, 1)
    # (in our case this was (1, 960, 1280, 1)) 
    preds = sess.run(pred)
    
    # Unpack the first dimension    
    preds = preds[0]
    
    return preds

def infer(img_name, model_weights):
    return infer_absolute_path(IMAGE_DIR + img_name, model_weights)

def infer_and_save_to_matlab_absolute_path(img_absolute_path, labels_absolute_path, model_weights):
    # Predict the labels    
    preds = infer_absolute_path(img_absolute_path, model_weights)
    scipy.io.savemat(labels_absolute_path, mdict={'labels': preds.astype(np.uint16)})
    print('The output file has been saved to {}'.format(labels_absolute_path))

    return

def infer_and_save_to_matlab(img_name, labels_name = 'labels.mat'):
    # Predict the labels    
    preds = infer(img_name)

    if not os.path.exists(MATLAB_SAVE_DIR):
        os.makedirs(MATLAB_SAVE_DIR)
    
    path = MATLAB_SAVE_DIR + labels_name
    scipy.io.savemat(path, mdict={'labels': preds})
    print('The output file has been saved to {}'.format(path))

    return

def infer_and_save_color_map_for_list():
    text_file = open(FILELIST, "r")
    lines = text_file.read().splitlines()
    text_file.close()
    
    for img_name in lines:
        infer_and_save_color_map(img_name)

    return

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # Now have
    # args.img_path and args.save_path
    # Optionally have
    # args.model_weights
    
    
    infer_and_save_to_matlab_absolute_path(args.img_path, args.save_path, args.model_weights)

    return
    
if __name__ == '__main__':
    main()
