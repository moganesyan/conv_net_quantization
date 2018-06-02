import tensorflow as tf
import numpy as np
import utils
#from utils import replace_variable
#from utils import get_dorefa


##catalog of different models
                

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

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



class cifarnetModel(object):
    def __init__(self,config):      
        
        self.num_trn=config.num_trn
        self.num_val=config.num_val
        self.num_tst=config.num_tst
        
        self.BITW=config.BITW
        self.BITA=config.BITA
        self.BITG=config.BITG
              
        
        #filter sizes
        self.filter_size1=config.filter_size1
        self.filter_size2=config.filter_size2



        
        self.num_filters1=config.num_filters1
        self.num_filters2=config.num_filters2

        
        
        self.fc1_size=config.fc1_size
        self.fc2_size=config.fc2_size
        
        self.img_size=config.img_size
        self.num_classes=config.num_classes
        self.num_channels=config.num_channels
        
        self.img_size_flat=self.img_size*self.img_size*self.num_channels

        
        # placeholder vars
        
        self._input_data=tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE,IMAGE_SIZE,3), name='x_in')
        self._output_data=tf.placeholder(tf.float32, shape=(None, NUM_CLASSES), name='y_true')
        
        self._is_train=tf.placeholder(dtype=tf.bool,shape=(),name="is_train")
        self._bitw=tf.placeholder(dtype=tf.float32,shape=(),name="bitw")
        self._bita=tf.placeholder(dtype=tf.float32,shape=(),name="bita")
        self._bitg=tf.placeholder(dtype=tf.float32,shape=(),name="bitg")
        
        
        self.fw,self.fa,self.fg=utils.get_dorefa(tf.cast(self._bitw,dtype=tf.float32),
                                                 tf.cast(self._bita,dtype=tf.float32),
                                                 tf.cast(self._bitg,dtype=tf.float32))

        
        
        ##
        self.sum_size=config.sum_size
        
        
        self.total_iterations=0
        self.total_trn_error=0
        
        self.train_batch_size=config.train_batch_size
        self.epochs=config.epochs
        self.iters=(self.num_trn/self.train_batch_size)*self.epochs




    def _variable_on_cpu(self,name, scope, shape, initializer):
      """Helper to create a Variable stored on CPU memory.
    
      Args:
        name: name of the variable
        scope: scope of the variable
        shape: list of ints
        initializer: initializer for Variable
    
      Returns:
        Variable Tensor
      """
      with tf.device('/cpu:0'):
        dtype = tf.float32
        
        if "conv1" in scope.name or "softmax_linear" in scope.name:
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

                
        else:
            with utils.replace_variable(lambda x: self.fw(x)):
                var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
                #var=self.fg(var)
                
      return var
    
    
    def _variable_with_weight_decay(self,name, scope, shape, stddev, wd):
      """Helper to create an initialized Variable with weight decay.
    
      Note that the Variable is initialized with a truncated normal distribution.
      A weight decay is added only if one is specified.
    
      Args:
        name: name of the variable
        scope: scope of the varible
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    
      Returns:
        Variable Tensor
      """
      dtype = tf.float32
      var = self._variable_on_cpu(
          name,
          scope,
          shape,
          tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
      if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
      return var

    
          
    def cabs(self,x):
        tmp=tf.abs(x)
        max_=tf.reduce_max(tmp)
        return tf.nn.relu(x/max_,name="act")
  
    
    def get_model(self,x_image,y_true):
        
        
#        x_image=self._input_data
#        #x_image = tf.reshape(x, [-1, self.img_size, self.img_size, self.num_channels])
#
#        y_true=self._output_data
        print("y true dim")
        print(y_true.shape)
        
#        y_true=tf.reshape(y_true,(y_true.shape[0],1))
#        print("reshaped")
#        print(y_true.shape)
#        y_true_cls = tf.argmax(y_true, axis=1)
#        print("y true cls")
#        print(y_true_cls.shape)
        
        
        #print(x_image.shape)
        with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE) as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                scope,
                                                 shape=[5, 5, 3, 64],
                                                 stddev=5e-2,
                                                 wd=None)
            conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases',scope, [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            utils._activation_summary(conv1)
            
            
            

      # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
      # norm1
#        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                        name='norm1')
    
      # conv2
        with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE) as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                scope,
                                                 shape=[5, 5, 64, 64],
                                                 stddev=5e-2,
                                                 wd=None)
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases',scope, [64], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            utils._activation_summary(conv2)
    
      # norm2
#        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                        name='norm2')
      # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
      # local3
        with tf.variable_scope('local3',reuse=tf.AUTO_REUSE) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        
        
            layer_shape=pool2.get_shape()
            num_features=layer_shape[1:4].num_elements()
            
            reshape=tf.reshape(pool2,[-1,num_features])
            #print(reshape.shape)
            #reshape = tf.reshape(pool2, [x_image.get_shape().as_list()[0], -1])
            
            
            dim = reshape.get_shape()[1].value
            #print(dim)
            weights = self._variable_with_weight_decay('weights', scope,shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases',scope, [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            utils._activation_summary(local3)
        
      # local4
        with tf.variable_scope('local4',reuse=tf.AUTO_REUSE) as scope:
            weights = self._variable_with_weight_decay('weights', scope,shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', scope,[192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
            utils._activation_summary(local4)
    
      # linear layer(WX + b),
      # We don't apply softmax here because
      # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
      # and performs the softmax internally for efficiency.
        with tf.variable_scope('softmax_linear',reuse=tf.AUTO_REUSE) as scope:
            weights = self._variable_with_weight_decay('weights',scope, [192, NUM_CLASSES],
                                                  stddev=1/192.0, wd=None)
            biases = self._variable_on_cpu('biases',scope, [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name) #logits
            utils._activation_summary(softmax_linear)
        
        ## 
        print("softmax linear")
        print(softmax_linear.shape)
        y_pred = tf.nn.softmax(softmax_linear)
        print("y pred cls")
        y_pred_cls = tf.argmax(y_pred, axis=1)
        print(y_pred_cls.shape)
#        print(softmax_linear)
#        print(y_true)
#        print(y_true_cls)
#        
        y_true = tf.cast(y_true, tf.int64)
        #y_true=tf.reshape(y_true,(y_true.shape[0],1))

        print("y true")
        print(y_true.shape)
        
        

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax_linear,
                                                                labels=y_true)
        #cross_entropy=tf.clip_by_value(cross_entropy,-10.,10.)


        cost = tf.reduce_mean(cross_entropy)
        
        
        
        
        tf.add_to_collection('losses', cost)
        
        total_cost=tf.add_n(tf.get_collection('losses'), name='total_loss')
                
        correct_prediction = tf.equal(y_pred_cls, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return y_pred_cls,kernel,total_cost,accuracy
    
    
    
         
    def optimize_loss(self, total_loss,global_step):
                
        #tf.summary.scalar('cost', cost)

        # Variables that affect learning rate.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = utils._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.RMSPropOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_step)
        with tf.control_dependencies([apply_gradient_op]):
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
        
        
        
        
        return variables_averages_op  


        
        
        
        
