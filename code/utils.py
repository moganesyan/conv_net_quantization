#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorpack.utils.argtools import graph_memoized
import functools
from tensorflow.python.ops import variable_scope


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


import time
from datetime import timedelta
import math




import os
import re
import sys
import tarfile

from six.moves import urllib

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'





_origin_get_variable = tf.get_variable
_object_stack = []


def _new_get_variable(*args, **kwargs):
    v = _origin_get_variable(*args, **kwargs)
    if len(_object_stack) != 0:
        return _object_stack[-1]._fn(v)
    else:
        return v


class TFVariableReplaceHelper(object):

    def __init__(self, fn):
        self._old_get_variable = None
        self._fn = fn

    def __enter__(self):
        global _object_stack
        _object_stack.append(self)
        self._old_get_variable = tf.get_variable
        tf.get_variable = _new_get_variable
        variable_scope.get_variable = _new_get_variable

    def __exit__(self, *args):
        global _object_stack
        _object_stack.pop()
        tf.get_variable = self._old_get_variable
        variable_scope.get_variable = self._old_get_variable

def replace_variable(fn):
    return TFVariableReplaceHelper(fn)


@graph_memoized
def get_dorefa(bitW, bitA, bitG):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def quantize(x, k):
        #n = float(2**k - 1)
        n=tf.cast(2**k-1,tf.float32)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def fw(x):
        
        def quant(x):
            def func1(x):
                with G.gradient_override_map({"Sign": "Identity"}):
                    E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
                    #E=tf.Print(E,[E,x],message="Value of E for 1 bit w is: ",summarize=10)
                    return tf.sign(x / E) * E
            def func2(x):
                #x=tf.Print(x,[x],message="Value of w for >1 bit w",summarize=10)
                x = tf.tanh(x)
                x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
                x_tf=2*quantize(x,bitW)-1
                #x_tf=tf.Print(x_tf,[x_tf],message="Value of quantized w for >1 bitw w",summarize=10)
                return x_tf           
            
            x_f=tf.cond(tf.greater(bitW,tf.constant(1,dtype=tf.float32)),lambda:func2(x),lambda:func1(x))
            return x_f
        
        def unquant(x):
            #x=tf.Print(x,[x],message="weights are : ",summarize=50)
            return tf.identity(x)
              
        x_f=tf.cond(tf.equal(bitW,tf.constant(32,dtype=tf.float32)),lambda:unquant(x),lambda:quant(x))
        
        
        
        
        return x_f
  

    def fa(x):
        
        def quant(x):
            return quantize(x, bitA)
        def unquant(x):
            return tf.identity(x)
        
        x_f=tf.cond(tf.equal(bitA,tf.constant(32,dtype=tf.float32)),lambda:unquant(x),lambda:quant(x))
        return x_f

    @tf.RegisterGradient("FGGrad")
    def grad_fg(op, x):
        #print(x.get_shape())
        #x=tf.Print(x,[x],message='raw grad bp',summarize=50)
        rank = x.get_shape().ndims
        assert rank is not None
        maxx = tf.reduce_max(tf.abs(x), list(range(1, rank)), keep_dims=True)
        x = x / maxx
        n=tf.cast(2**bitG-1,tf.float32)
        x = x * 0.5 + 0.5 + tf.random_uniform(
            tf.shape(x), minval=-0.5 / n, maxval=0.5 / n)
        x = tf.clip_by_value(x, 0.0, 1.0)
        x = quantize(x, bitG) - 0.5
        x_tf=x*maxx*2
        #x_tf=tf.Print(x_tf,[x_tf],message="quantised grad bp",summarize=50)
        x_tf=tf.where(tf.is_nan(x_tf), tf.zeros_like(x_tf), x_tf)
        #x_tf=tf.is_nan(x_tf)
        #x_tf=tf.Print(x_tf,[x_tf],message="Nan is in ",summarize=20)
        #print(tf.is_nan(x_tf).eval())
        return x_tf
#    @tf.RegisterGradient("FGGrad")   
#    def grad_fg(op, x):
#        #print(x.get_shape())
#        x=tf.Print(x,[x],message='raw grad bp',summarize=20)
#        x_tf=tf.identity(x)
#        return x_tf


    def fg(x):
        #x=tf.Print(x,[x],message="raw grad fp",summarize=10)
        def quant(x):
            with G.gradient_override_map({"Identity": "FGGrad"}):
                return tf.identity(x)
        def unquant(x):
            return tf.identity(x)
        
        x_f=tf.cond(tf.equal(bitG,tf.constant(32,dtype=tf.float32)),lambda:unquant(x),lambda:quant(x))
        return x_f        
        
            
    return fw, fa, fg


def plot_images(images, cls_true, img_shape, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        
        if(len(images[i].shape)==3):
            ax.imshow(images[i].transpose((1,2,0)), interpolation="nearest")
        else:
            ax.imshow(images[i],cmap='binary')
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors(data_true,cls_pred, correct,img_shape):

    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data_true[0][incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data_true[1].flatten()
    cls_true=cls_true[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                img_shape=img_shape,
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(data_true,cls_pred,num_classes):

    cls_true = data_true[1]
    cls_true=cls_true.flatten()
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_conv_weights(session, config, weights, input_channel=0):
    bitw=config._bitw
    bita=config._bita
    bitg=config._bitg

    BITW=config.BITW
    BITA=config.BITA
    BITG=config.BITG
    
    BITA=32
    BITW=32
    BITG=32

    w = session.run(weights, feed_dict={bitw:BITW,bita:BITA,bitg:BITG})

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    
    print("The weights are ")
    print(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]
    print(num_filters)

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(int(num_grids), int(num_grids))

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


#code taken from  https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    
    
    
def gvdebug(g, v):
    #g=tf.Print(g,[v.name],message='Gradient Name: ',summarize=10)
    #g = tf.Print(g,[g],'Gradient Value: ',summarize=50)
    #v = tf.Print(v,[v],'V: ')
#    g2 = tf.zeros_like(g, dtype=tf.float32)
#    v2 = tf.zeros_like(v, dtype=tf.float32)
#    g2 = g
#    v2 = v
    #print(g.get_shape())
    return g,v




def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op



def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    
    
    
    
def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels



def distort_images(image,HEIGHT,WIDTH,is_train):
    print(image.shape)
    image=tf.transpose(image, [0,2, 3, 1])
    image = tf.cast(image, tf.float32)
    print(image.shape)
    
    #tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    
    #distorted_image = tf.random_crop(image, [HEIGHT, WIDTH, 3])

    # Randomly flip the image horizontally.
    
    def distort(images):
        
        #tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    
        distorted_image=tf.map_fn(lambda img: tf.image.random_flip_left_right(img),images)
    
        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.map_fn(lambda img: tf.image.random_brightness(img,
                                                     max_delta=63),distorted_image)
        
        
        distorted_image = tf.map_fn(lambda img: tf.image.random_contrast(img,
                                                   lower=0.2, upper=1.8),distorted_image)
    
        return distorted_image
    
    def no_distort(image):
        return image
    
    distorted_image=tf.cond(is_train,lambda: distort(image),lambda:no_distort(image))
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img),distorted_image)
    
    
    
    return float_image