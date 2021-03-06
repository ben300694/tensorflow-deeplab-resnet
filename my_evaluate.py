"""
Evaluation script for the DeepLab-ResNet network on a validation subset
of your dataset.

This script evaluates the model on the validation images
specified in the DATA_VAL_LIST_PATH in the config file.

Do not forget to also change
NUM_VALIDATION_IMAGES
in the config file to the appropriate value!
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import yaml

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label

# Load the configuration file
full_path = os.path.realpath(__file__)
config = yaml.safe_load(open(os.path.dirname(full_path) + '/config.yml'))

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = config['directories']['DATA_DIRECTORY']
DATA_VAL_LIST_PATH = config['directories']['lists']['DATA_VAL_LIST_PATH']
IGNORE_LABEL = config['IGNORE_LABEL']
NUM_CLASSES = config['NUM_CLASSES']
NUM_STEPS = config['NUM_VALIDATION_IMAGES'] # Number of images in the validation set.

RESTORE_FROM = config['RESTORE_FROM']

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-val-list", type=str, default=DATA_VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
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

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    print(args)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_val_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # mIoU
    pred = tf.reshape(pred, [-1,])
    ground_truth = tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.less_equal(ground_truth, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.

    # mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, ground_truth, num_classes=args.num_classes,
    #                                                         weights=weights)
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, ground_truth, num_classes=args.num_classes)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Iterate over training steps.
    for step in range(args.num_steps):
        preds, _ = sess.run([pred, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
