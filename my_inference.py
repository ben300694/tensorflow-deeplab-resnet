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

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

NUM_CLASSES = config['NUM_CLASSES']

IMAGE_DIR = config['directories']['IMAGE_DIR']
COLOR_MASK_SAVE_DIR = config['directories']['COLOR_MASK_SAVE_DIR']
MATLAB_SAVE_DIR = config['directories']['inference']['MATLAB_SAVE_DIRECTORY']
MODEL_WEIGHTS = config['RESTORE_FROM']

FILELIST = config['directories']['lists']['DATA_FILELIST_PATH']

BATCH_SIZE = 6


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


def infer_and_save_colormask(img_path, model_weights, use_crf):
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


def infer_and_save_matlab_for_filelist(filelist=FILELIST,
                                       also_save_colormask=False,
                                       color_mask_save_dir=COLOR_MASK_SAVE_DIR,
                                       matlab_save_dir=MATLAB_SAVE_DIR,
                                       model_weights=MODEL_WEIGHTS,
                                       use_crf=False):
    # Open and parse the filelist
    text_file = open(filelist, "r")
    lines = text_file.read().splitlines()
    text_file.close()

    # Perform inference.
    print('Starting inference')
    preds = infer_multiple_images(filelist, model_weights, use_crf)

    # Get the color values for the labels if the colormask is also saved
    if also_save_colormask == True:
        print('Decoding the labels')
        masks = decode_labels(preds, num_images=preds.shape[0], num_classes=NUM_CLASSES)

    # Prepare the location for saving the colormask
    if use_crf == True:
        crf_used_string = 'with_crf'
    else:
        crf_used_string = ''

    if not os.path.exists(color_mask_save_dir):
        os.makedirs(color_mask_save_dir)
    if not os.path.exists(matlab_save_dir):
        os.makedirs(matlab_save_dir)

    for index, img_name in enumerate(lines):
        # Save the colormask as .png
        if also_save_colormask == True:
            im = Image.fromarray(masks[index])
            # os.path.splitext(img_name)[0] removes the '.png' extension from the image name
            colormask_save_path = color_mask_save_dir + os.path.splitext(img_name)[
                0] + crf_used_string + '_colormask.png'
            im.save(colormask_save_path)
            print('The colormask has been saved to {}'.format(colormask_save_path))

        # Save the mask as MATLAB .mat
        # os.path.splitext(img_name)[0] removes the '.png' extension from the image name
        matlab_labels_save_path = matlab_save_dir + os.path.splitext(img_name)[0] + '.mat'
        if use_crf == False:
            scipy.io.savemat(matlab_labels_save_path, mdict={'labels': preds[index].astype(np.uint16)})
            print('The matlab file has been saved to {}'.format(matlab_labels_save_path))
            continue
        # TODO Save the inference correctly in the case that CRF is used
        # preds_crf = infer_absolute_path(img_absolute_path, model_weights, use_crf=True)
        # scipy.io.savemat(labels_absolute_path,
        #                  mdict={'labels': preds.astype(np.uint16), 'labels_crf': preds_crf.astype(np.uint16)})
        # print('The output file has been saved to {}'.format(labels_absolute_path))

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
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])

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


"""
Assumption: All the images referenced in the file list have the same dimension
"""
def infer_multiple_images(filelist=FILELIST, model_weights=MODEL_WEIGHTS, use_crf=False):
    tf.reset_default_graph()

    # Create queue coordinator.
    # coord = tf.train.Coordinator()

    text_file = open(filelist, "r")
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
        # print('Read image {}'.format(img_name))

    img_tensor = tf.convert_to_tensor(img_list, dtype=tf.float32)

    image_dataset = tf.data.Dataset.from_tensor_slices(img_tensor)
    image_batch = image_dataset.batch(BATCH_SIZE)
    # print(image_dataset)
    # print(image_batch)
    print('Image preprocessing done')

    # create the iterator
    iter = image_batch.make_one_shot_iterator()
    data = iter.get_next()

    # Create network.
    net = DeepLabResNetModel({'data': data}, is_training=False, num_classes=NUM_CLASSES)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img_list[0])[0:2, ])

    # TODO Finish writing this function
    # CRF.
    # if use_crf:
    #     raw_output_up = tf.nn.softmax(raw_output_up)
    #     raw_output_up = tf.py_func(dense_crf, [raw_output_up, tf.expand_dims(img_ph, dim=0)], tf.float32)

    # Get maximum score
    raw_output_up = tf.argmax(raw_output_up, axis=3)
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
    # with preds.shape = (number_of_images, image_height, image_width, 1)
    # (in our case this was (?, 960, 1280, 1))
    batch_results = []
    number_of_batches = len(lines) / BATCH_SIZE
    for i in range(number_of_batches + 1):
        batch_results.append(sess.run(pred))
        print('Done with batch ' + str(i) + ' of ' + str(number_of_batches))

    print('Concatenating batches')
    preds = np.concatenate(batch_results, axis=0)

    return preds


def infer_and_save_to_matlab_absolute_path(img_absolute_path, labels_absolute_path, model_weights, use_crf):
    # Predict the labels
    preds = infer_absolute_path(img_absolute_path, model_weights, use_crf=False)

    if use_crf == False:
        scipy.io.savemat(labels_absolute_path, mdict={'labels': preds.astype(np.uint16)})
        print('The output file has been saved to {}'.format(labels_absolute_path))
        return

    preds_crf = infer_absolute_path(img_absolute_path, model_weights, use_crf=True)
    scipy.io.savemat(labels_absolute_path,
                     mdict={'labels': preds.astype(np.uint16), 'labels_crf': preds_crf.astype(np.uint16)})
    print('The output file has been saved to {}'.format(labels_absolute_path))

    return


def infer_and_save_to_matlab(img_name,
                             labels_name='labels.mat',
                             matlab_save_dir=MATLAB_SAVE_DIR):
    # Predict the labels    
    preds = infer(img_name)

    if not os.path.exists(matlab_save_dir):
        os.makedirs(matlab_save_dir)

    path = matlab_save_dir + labels_name
    scipy.io.savemat(path, mdict={'labels': preds})
    print('The output file has been saved to {}'.format(path))

    return


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # Now have
    # args.img_path and args.save_path
    # Optionally have
    # args.model_weights

    infer_and_save_to_matlab_absolute_path(args.img_path,
                                           args.save_path,
                                           args.model_weights,
                                           args.use_crf)

    return


if __name__ == '__main__':
    main()
