import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import utils
import models_opt
import time
import importlib

#import tensorflow.python.debug as tf_debug

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

tf.app.flags.DEFINE_string('config', "config.cifarnet", 'config')


def train(config):
       
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    model=models_opt.cifarnetModel(config)

    global_step = tf.train.get_or_create_global_step()
    

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = utils.distorted_inputs()
      print(images.shape)
        
    bitw=model._bitw
    bita=model._bita
    bitg=model._bitg
    
    print("lol")
    
    print(images)
    print(labels)
    print(model.BITW)
    print(model.BITA)
    print(model.BITG)
    
    
    feed_dict_train = {
                  bitw:model.BITW,
                  bita:model.BITA,
                  bitg:model.BITG
                  }
    
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    y_pred_cls,kernel,total_cost,accuracy = model.get_model(images,labels)

    # Calculate loss.

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.optimize_loss(total_cost, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(total_cost)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(total_cost),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op,feed_dict=feed_dict_train)

    


def get_config():
    
    return importlib.import_module(FLAGS.config).Config()



def main(_):
    #download the cifar10 dataset if needed and run optimizer
    utils.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    
    #get configuration file
    config=get_config()
    #get model class

    train(config) 
    


    
if __name__ == "__main__":
    tf.app.run()





