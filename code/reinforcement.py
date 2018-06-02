import tensorflow as tf
import numpy as np
import pandas as pd
import utils

import tensorflow.contrib.slim as slim


class Qnetwork():
    def __init__(self,is_training,config):
        self.num_trn=config.num_trn
        self.num_val=config.num_val
        self.num_tst=config.num_tst
        
        #filter sizes
        self.filter_size1=config.filter_size1
        self.filter_size2=config.filter_size2
        
        self.num_filters1=config.num_filters1
        self.num_filters2=config.num_filters2
        
        
        self.fc_size=config.fc_size
        
        self.img_size=config.img_size
        self.num_classes=config.num_classes
        self.num_channels=config.num_channels
        
        self.img_size_flat=self.img_size*self.img_size*self.num_channels

        
        # placeholder vars
        
        self._input_data=tf.placeholder(tf.float32, shape=(None, self.img_size_flat), name='x_in')
        self._output_data=tf.placeholder(tf.float32, shape=(None, self.num_classes), name='y_true')

        self._is_train=tf.placeholder(dtype=tf.bool,shape=(),name="is_train")
                
        
        ##
        self.sum_size=config.sum_size
        
        
        self.total_iterations=0
        self.total_trn_error=0
        
        self.train_batch_size=config.train_batch_size
        self.epochs=config.epochs
        self.iters=(self.num_trn/self.train_batch_size)*self.epochs
        
        
        
    def get_model(self):
        X_in =  self._input_data
        
        imageIn = tf.reshape(X_in,shape=[-1,self.img_size,self.img_size,self.num_channels])
        conv1 = slim.conv2d( \
            inputs=imageIn,num_outputs=self.num_filters1,kernel_size=[self.filter_size1,self.filter_size1],stride=[1,1,1,1],padding='SAME', biases_initializer=None)
        conv2 = slim.conv2d( \
            inputs=conv1,num_outputs=self.num_filters2,kernel_size=[self.filter_size2,self.filter_size2],stride=[1,2,2,1],padding='SAME', biases_initializer=None)
        conv2_flat=slim.flatten(conv2)
        fc1=slim.fully_connected(inputs=conv2_flat,num_outputs=self.fc_size,biases_initializer=None)
        
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        streamAC,streamVC = tf.split(fc1,2,3)
        
        streamA = slim.flatten(streamAC)
        streamV = slim.flatten(streamVC)
        
        xavier_init = tf.contrib.layers.xavier_initializer()
        AW = tf.Variable(xavier_init([self.fc_size//2,self.num_classes]))
        VW = tf.Variable(xavier_init([self.fc_size//2,1]))
        Advantage = tf.matmul(streamA,AW)
        Value = tf.matmul(streamV,VW)
        
        #Then combine them together to get our final Q-values.
        Qout = Value + tf.subtract(Advantage,tf.reduce_mean(Advantage,axis=1,keep_dims=True))
        predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        actions = tf.placeholder(shape=[None],dtype=tf.int32)
        actions_onehot = tf.one_hot(actions,self.num_classes,dtype=tf.float32)
        
        Q = tf.reduce_sum(tf.multiply(Qout,actions_onehot), axis=1)
        
        td_error = tf.square(targetQ - Q)
        loss = tf.reduce_mean(td_error)
        trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        updateModel = trainer.minimize(loss)