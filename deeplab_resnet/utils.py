from PIL import Image
import numpy as np
import scipy.io
import tensorflow as tf
import os
import yaml

import pydensecrf.densecrf as dcrf

# Load the configuration file
full_path = os.path.realpath(__file__)
# Use os.path.dirname again to get a folder up
config = yaml.safe_load(open(os.path.dirname(os.path.dirname(full_path)) + '/config.yml'))

NUM_CLASSES = config['NUM_CLASSES']
MATLAB_COLORMAP_PATH = config['directories']['MATLAB_COLORMAP_PATH']

# Read in the matlab file containing the color map
colormap_mat = scipy.io.loadmat(MATLAB_COLORMAP_PATH)

# For example use colormap_mat['colorNames'][3]
# to get the names of the classes

# Loads the label colors from the colormap.mat
label_colors = colormap_mat['colorRGBValues']

# The MATLAB colormap.mat starts indexed at 1 and contains the
# labels 'background' = 1, ... , 'drone' =27
# The ignore-label in the MATLAB Label tool is '0'
# and not contained in the colormat.mat
# So we are adding the element [255, 255, 255]
# in the front of label_colors to get a colormap
# including the ignore label.
label_colors = np.insert(label_colors, 0, np.array([255, 255, 255]), axis=0)

# # Color map for the example of labeling drone images
# # is now after prepending the IGNORE_LABEL:
#
# label_colors = [[255, 255, 255],
#                 # 0 = undefined
#                 [0, 0, 0],
#                 # 1 = Background
#                 [0, 0, 182],
#                 # 2 = Pedestrian
#                 [0, 0, 219],
#                 # 3 = Bicyclist
#                 [0, 0, 255],
#                 # 4 =
#                 [0, 36, 255],
#                 # 5 =
#                 [0, 73, 255],
#                 # 6 =
#                 [0, 109, 255],
#                 # 7 =
#                 [0, 146, 255],
#                 # 8 =
#                 [0, 182, 255],
#                 [0, 219, 255],
#                 [0, 255, 255],
#                 [36, 255, 219],
#                 [73, 255, 182],
#                 [109, 255, 146],
#                 [146, 255, 109],
#                 [182, 255, 73],
#                 [219, 255, 36],
#                 [255, 255, 0],
#                 [255, 219, 0],
#                 [255, 182, 0],
#                 [255, 146, 0],
#                 [255, 109, 0],
#                 [255, 73, 0],
#                 [255, 36, 0],
#                 [255, 0, 0],
#                 [219, 0, 0],
#                 [0, 0, 146]]
#                 # 27 = Drone


def decode_labels(mask, num_images=1, num_classes=NUM_CLASSES):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = tuple(label_colors[k])
        outputs[i] = np.array(img)
    return outputs


def prepare_label(input_batch, new_size, num_classes=NUM_CLASSES, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.

    Returns:
      Outputs a tensor of shape [batch_size h w num_classes]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch,
                                                       new_size)  # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3])  # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch


def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs


def dense_crf(probs, img=None, n_iters=10,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term
      (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term
      (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    _, h, w, _ = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, NUM_CLASSES)  # Define DenseCRF model.
    U = -np.log(probs)  # Unary potential.
    U = U.reshape((NUM_CLASSES, -1))  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((NUM_CLASSES, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
