"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import yaml

from PIL import Image

import tensorflow as tf
import numpy as np
import scipy.io

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, dense_crf, prepare_label

# Load the configuration file
full_path = os.path.realpath(__file__)
config = yaml.safe_load(open(os.path.dirname(full_path) + '/config.yml'))

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
NUM_CLASSES = config['NUM_CLASSES']

IMAGE_DIR = config['directories']['IMAGE_DIR']
COLOR_MASK_SAVE_DIR = config['directories']['COLOR_MASK_SAVE_DIR']
MATLAB_SAVE_DIR = config['directories']['MATLAB_SAVE_DIR']
MODEL_WEIGHTS = config['RESTORE_FROM']

FILELIST = config['directories']['DATA_FILELIST_PATH']

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
    parser.add_argument("--use_crf", action="store_true",
                        help="Whether to apply a CRF after inference")
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


def infer_and_save_color_map(img_path, model_weights, use_crf):

    # Perform inference.
    preds = infer_absolute_path(img_path, model_weights, use_crf)
    # Have to add an extra dimension in the front because `infer_absolute_path` removed the batch dimension
    preds = np.expand_dims(preds, axis=0)

    msk = decode_labels(preds, num_classes=NUM_CLASSES)
    im = Image.fromarray(msk[0])

    # Save the image
    head, tail = os.path.split(img_path)
    filename, file_extension = os.path.splitext(tail)
    save_path = COLOR_MASK_SAVE_DIR + filename + '_colormask.png'
    if not os.path.exists(COLOR_MASK_SAVE_DIR):
        os.makedirs(COLOR_MASK_SAVE_DIR)
    im.save(save_path)
    
    print('The output file has been saved to {}'.format(save_path))


def infer_and_save_color_map_for_filelist():
    text_file = open(FILELIST, "r")
    lines = text_file.read().splitlines()
    text_file.close()

    for img_name in lines:
        infer_and_save_color_map(IMAGE_DIR + img_name, MODEL_WEIGHTS, True)
    return

def infer_absolute_path(img_path, model_weights, use_crf):
    """Create the model and start the evaluation process."""
    tf.reset_default_graph()
    
    # Prepare image.
    img_orig = tf.image.decode_png(tf.read_file(img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img_orig)
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

    # CRF.
    if use_crf:
        raw_output_up = tf.nn.softmax(raw_output_up)
        raw_output_up = tf.py_func(dense_crf, [raw_output_up, tf.expand_dims(img_orig, dim=0)], tf.float32)
    
    # Get maximum score    
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

def infer_and_save_to_matlab_absolute_path(img_absolute_path, labels_absolute_path, model_weights, use_crf):
    # Predict the labels
    preds = infer_absolute_path(img_absolute_path, model_weights, use_crf=False)
    
    if use_crf == False:    
        scipy.io.savemat(labels_absolute_path, mdict={'labels': preds.astype(np.uint16)})
        print('The output file has been saved to {}'.format(labels_absolute_path))
        return
    
    preds_crf = infer_absolute_path(img_absolute_path, model_weights, use_crf=True)
    scipy.io.savemat(labels_absolute_path, mdict={'labels': preds.astype(np.uint16), 'labels_crf': preds_crf.astype(np.uint16)})
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


def infer_multiple_images(model_weights, use_crf):
    text_file = open(FILELIST, "r")
    lines = text_file.read().splitlines()
    text_file.close()

    img_list = []

    for img_name in lines:
        # Prepare image.
        img_orig = tf.image.decode_png(tf.read_file(IMAGE_DIR + img_name), channels=3)
        # Convert RGB to BGR.
        img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img_orig)
        img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
        # Extract mean.
        img -= IMG_MEAN
        img_list.append(img)

    tf.reset_default_graph()

    # Create network.
    img_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    net = DeepLabResNetModel({'data': img_ph}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])

    # CRF.
    # if use_crf:
    #     raw_output_up = tf.nn.softmax(raw_output_up)
    #     raw_output_up = tf.py_func(dense_crf, [raw_output_up, tf.expand_dims(img_orig, dim=0)], tf.float32)

    # Get maximum score
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
    preds = sess.run(pred, feed_dict={img_ph: img})

    return preds

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # Now have
    # args.img_path and args.save_path
    # Optionally have
    # args.model_weights
    
    
    infer_and_save_to_matlab_absolute_path(args.img_path, args.save_path, args.model_weights, args.use_crf)

    return
    
if __name__ == '__main__':
    main()
