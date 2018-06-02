import tensorflow as tf
import numpy as np
import utils
#from utils import replace_variable
#from utils import get_dorefa


##catalog of different models

class lenetModel(object):
    def __init__(self,is_training,config):      
        
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
        
        
        self.fc_size=config.fc_size
        
        self.img_size=config.img_size
        self.num_classes=config.num_classes
        self.num_channels=config.num_channels
        
        self.img_size_flat=self.img_size*self.img_size*self.num_channels

        
        # placeholder vars
        
        self._input_data=tf.placeholder(tf.float32, shape=(None, self.img_size_flat), name='x_in')
        self._output_data=tf.placeholder(tf.float32, shape=(None, self.num_classes), name='y_true')
        
        self._bitw=tf.placeholder(dtype=tf.float32,shape=(),name="bitw")
        self._bita=tf.placeholder(dtype=tf.float32,shape=(),name="bita")
        self._bitg=tf.placeholder(dtype=tf.float32,shape=(),name="bitg")
        self._is_train=tf.placeholder(dtype=tf.bool,shape=(),name="is_train")
        
        
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



    def new_conv_layer(self,x_in,name,
                       num_input_channels,filter_size,
                       num_filters,bitw,strides):
        
        shape=[filter_size,filter_size,num_input_channels,num_filters]
        
        if "conv1" in name:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
                #weights=self.fg(weights)
                #tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)

                #utils.variable_summaries(weights)
                
        else:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE),utils.replace_variable(lambda x: self.fw(x)):
                weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
                weights=self.fg(weights)
                #tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)

                #utils.variable_summaries(weights)
                
         
#        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
#            shape=[num_filters]
#            bias=tf.get_variable(name="b",initializer=tf.constant_initializer(0.0),shape=shape)
            
        layer=tf.nn.conv2d(input=x_in,filter=weights,
                           padding="SAME",strides=strides)
        #layer=tf.nn.bias_add(layer,bias)
        return layer,weights
    
    
    def new_fc_layer(self,x_in,name,
                     num_inputs,
                     num_outputs,
                     bitw):
        if "fco" not in name:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE),utils.replace_variable(lambda x:self.fw(x)):
                shape=[num_inputs,num_outputs]
                weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
                weights=self.fg(weights)
                #utils.variable_summaries(weights)
               # tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)


        else:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                shape=[num_inputs,num_outputs]
                weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
                #weights=self.fg(weights)
                #utils.variable_summaries(weights)
                #tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)

        
#        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
#            shape=[num_outputs]
#            bias=tf.get_variable(name="b",initializer=tf.constant_initializer(0.0),shape=shape)
                
        layer=tf.matmul(x_in,weights)
        #layer=tf.nn.bias_add(layer,bias)
        return layer,weights
    
    
    def flatten_layer(self,layer):
        
        layer_shape=layer.get_shape()
        num_features=layer_shape[1:4].num_elements()
        layer_flat=tf.reshape(layer,[-1,num_features])
        
        return layer_flat,num_features
    
    
    def cabs(self,x):
        tmp=tf.abs(x)
        max_=tf.reduce_max(tmp)
        return tf.nn.relu(x/max_,name="act")
  
        
    
    def get_model(self):
        
        
        x=self._input_data
        x_image = tf.reshape(x, [-1, self.img_size, self.img_size, self.num_channels])

        y_true=self._output_data
        y_true_cls = tf.argmax(y_true, axis=1)
        
        layer_conv1, weights_conv1 = \
                        self.new_conv_layer(x_in=x_image,name="conv1",
                                       num_input_channels=self.num_channels,
                                       filter_size=self.filter_size1,
                                       num_filters=self.num_filters1,
                                       bitw=self.BITW,
                                       strides=[1,1,1,1])
                        
                        
        layer_conv1 = tf.nn.max_pool(value=layer_conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',name="conv1_pool")
        
        layer_conv1=self.cabs(layer_conv1)
       # layer_conv1=tf.nn.relu(layer_conv1,name="conv1_act")
        layer_conv1=self.fa(layer_conv1)
        
        
        ##
        
        
        layer_conv2, weights_conv2 = \
                        self.new_conv_layer(x_in=layer_conv1,name="conv2",
                                       num_input_channels=self.num_filters1,
                                       filter_size=self.filter_size2,
                                       num_filters=self.num_filters2,
                                       bitw=self.BITW,
                                       strides=[1,2,2,1])
                        
        
        layer_conv2=tf.nn.max_pool(value=layer_conv2,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',name="conv2_pool")
        
        
       #layer_conv2=tf.nn.relu(layer_conv2,name="conv2_act") 
        layer_conv2=self.cabs(layer_conv2)
        layer_conv2=self.fa(layer_conv2)
        
        
        
        ##
        
        
        layer_flat, num_features = self.flatten_layer(layer_conv2)

        layer_fc1,weights_fc1 = self.new_fc_layer(x_in=layer_flat,name="fc1",
                                 num_inputs=num_features,
                                 num_outputs=self.fc_size,
                                 bitw=self.BITW)
        
        

        #layer_fc1=tf.nn.relu(layer_fc1,name="fc1_act")
        #layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="layer fc1",summarize=50)
        
        layer_fc1=self.cabs(layer_fc1)
        
        #layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="layer fc1 after cabs",summarize=50)
        
        layer_fc1=self.fa(layer_fc1)
        
        #layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="layer fc1 after fa",summarize=50)

        ##
        
        layer_fc2,weights_fc2 = self.new_fc_layer(x_in=layer_fc1,name="fco",
                                 num_inputs=self.fc_size,
                                 num_outputs=self.num_classes,
                                 bitw=self.BITW)
        
        ##
        
        
        y_pred = tf.nn.softmax(layer_fc2)
                
        y_pred_cls = tf.argmax(y_pred, axis=1)
        
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                labels=y_true)
        cross_entropy=tf.clip_by_value(cross_entropy,-10.,10.)


        cost = tf.reduce_mean(cross_entropy)

        
        tf.summary.scalar('cost', cost)

        
        
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)
        gvs=optimizer.compute_gradients(cost)
        
        capped_gvs = [utils.gvdebug(grad,var) for grad, var in gvs]
        
        #add gradient histograms
        for grad,var in capped_gvs:
            tf.summary.histogram("{}-grad".format(str(var.name)), grad) 

        
        optimizer = optimizer.apply_gradients(capped_gvs)
       
    
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                


        
        return y_pred_cls,optimizer,accuracy,[weights_conv1,weights_conv2],gvs
        
        
        
            
                
                        
                
        
class cifarnetModel(object):
    def __init__(self,is_training,config):      
        
        self.num_trn=config.num_trn
        self.num_val=config.num_val
        self.num_tst=config.num_tst
        
        self.BITW=config.BITW
        self.BITA=config.BITA
        self.BITG=config.BITG
              
        
        #filter sizes
        self.filter_size1=config.filter_size1
        self.filter_size2=config.filter_size2
        self.filter_size3=config.filter_size3
        self.filter_size4=config.filter_size4
        self.filter_size5=config.filter_size5
        self.filter_size6=config.filter_size6
        self.filter_size7=config.filter_size7
        self.filter_size8=config.filter_size8


        
        self.num_filters1=config.num_filters1
        self.num_filters2=config.num_filters2
        self.num_filters3=config.num_filters3
        self.num_filters4=config.num_filters4
        self.num_filters5=config.num_filters5
        self.num_filters6=config.num_filters6
        self.num_filters7=config.num_filters7
        self.num_filters8=config.num_filters8
        
        
        self.fc1_size=config.fc1_size
        self.fc2_size=config.fc2_size
        
        self.img_size=config.img_size
        self.num_classes=config.num_classes
        self.num_channels=config.num_channels
        
        self.img_size_flat=self.img_size*self.img_size*self.num_channels

        
        # placeholder vars
        
        self._input_data=tf.placeholder(tf.float32, shape=(None, self.num_channels,self.img_size,self.img_size), name='x_in')
        self._output_data=tf.placeholder(tf.float32, shape=(None, self.num_classes), name='y_true')
        
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



    def new_conv_layer(self,x_in,name,
                       num_input_channels,filter_size,
                       num_filters,bitw,strides):
        
        shape=[filter_size,filter_size,num_input_channels,num_filters]
        
        if "conv1" in name:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
                #weights=self.fg(weights)
                #tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)

                #utils.variable_summaries(weights)
                
        else:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE),utils.replace_variable(lambda x: self.fw(x)):
                weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
                #weights=tf.assign(weights,weights)

                weights=self.fg(weights)
                #tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)

                #utils.variable_summaries(weights)
                
         
#        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
#            shape=[num_filters]
#            bias=tf.get_variable(name="b",initializer=tf.constant_initializer(0.0),shape=shape)
        #weights=tf.assign(weights,weights)    
        layer=tf.nn.conv2d(input=x_in,filter=weights,
                           padding="SAME",strides=strides)
        #layer=tf.nn.bias_add(layer,bias)
        return layer,weights
    
    
    def new_fc_layer(self,x_in,name,
                     num_inputs,
                     num_outputs,
                     bitw):
        if "fco" not in name:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE),utils.replace_variable(lambda x:self.fw(x)):
                shape=[num_inputs,num_outputs]
                weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
                #weights=tf.assign(weights,weights)
                weights=self.fg(weights)
                #utils.variable_summaries(weights)
               # tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)


        else:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                shape=[num_inputs,num_outputs]
                weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
                #weights=self.fg(weights)
                #utils.variable_summaries(weights)
                #tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)

        
#        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
#            shape=[num_outputs]
#            bias=tf.get_variable(name="b",initializer=tf.constant_initializer(0.0),shape=shape)
        #weights=tf.assign(weights,weights)        
        layer=tf.matmul(x_in,weights)
        #layer=tf.nn.bias_add(layer,bias)
        return layer,weights
    

       
    def flatten_layer(self,layer):
        
        layer_shape=layer.get_shape()
        num_features=layer_shape[1:4].num_elements()
        layer_flat=tf.reshape(layer,[-1,num_features])
        
        return layer_flat,num_features
    
#    def cabs(self,x):
#        return tf.minimum(1.0, tf.abs(x), name='cabs')
#    def cabs(self,x):
#        return tf.nn.relu(x,name="act") 
        
#    def cabs(self,x):
#        tmp=tf.abs(x)
#        max_=tf.reduce_max(tmp)
#        return x/max_
#    
    
    def cabs(self,x):
        tmp=tf.abs(x)
        max_=tf.reduce_max(tmp)
        return tf.nn.relu(x/max_,name="act")
  
    
    def get_model(self):
        
        
        x=self._input_data
        #x_image = tf.reshape(x, [-1, self.img_size, self.img_size, self.num_channels])
        
        x_image=utils.distort_images(x,self.img_size,self.img_size,self._is_train)

        y_true=self._output_data
        y_true_cls = tf.argmax(y_true, axis=1)
        
        layer_conv1, weights_conv1 = \
                        self.new_conv_layer(x_in=x_image,name="conv1",
                                       num_input_channels=self.num_channels,
                                       filter_size=self.filter_size1,
                                       num_filters=self.num_filters1,
                                       bitw=self.BITW,strides=[1,1,1,1])
        
        #layer_conv1=self.fg(layer_conv1) 
                   
#        tf.where(tf.is_nan(weights_conv1), tf.zeros_like(weights_conv1), weights_conv1)
#        tf.where(tf.is_nan(layer_conv1), tf.zeros_like(layer_conv1), layer_conv1)
                        
#        layer_conv1=tf.Print(layer_conv1,[tf.is_nan(layer_conv1)],message="layer 1 fails here",summarize=50)
#        layer_conv1=tf.Print(layer_conv1,[tf.is_nan(weights_conv1)],message="weights 1 fails here",summarize=50)
                        
                        
        #layer_conv1=tf.nn.dropout(layer_conv1,keep_prob=0.1,name="conv1_drop")
        #layer_conv1=tf.Print(layer_conv1,[weights_conv1],message="Weights conv 1 main",summarize=10)
        layer_conv1 = tf.nn.max_pool(value=layer_conv1,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',name="conv1_pool")
                        
                               
        #layer_conv1=tf.nn.relu(layer_conv1,name="conv1_act")

        layer_conv1=self.cabs(layer_conv1)
        tf.summary.histogram('pre_activations_c1', layer_conv1)

        layer_conv1=self.fa(layer_conv1)
        tf.summary.histogram('post_activations_c1', layer_conv1)
        tf.summary.histogram('weights_c1',weights_conv1)
        
        
#        layer_conv1=tf.Print(layer_conv1,[layer_conv1],message="Layer conv1 out",summarize=20)
#        weights_conv1=tf.Print(weights_conv1,[weights_conv1],message="Weights conv1 out",summarize=20)
        
        ##
        
        
        layer_conv2, weights_conv2 = \
                        self.new_conv_layer(x_in=layer_conv1,name="conv2",
                                       num_input_channels=self.num_filters1,
                                       filter_size=self.filter_size2,
                                       num_filters=self.num_filters2,
                                       bitw=self.BITW,strides=[1,1,1,1])
        #layer_conv2=tf.Print(layer_conv2,[weights_conv2],message="Weights conv 2 main",summarize=10)

        #layer_conv2=self.fg(layer_conv2)
                        
        layer_conv2 = tf.nn.max_pool(value=layer_conv2,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',name="conv2_pool")
        #layer_conv2=tf.nn.dropout(layer_conv2,keep_prob=0.01,name="conv2_drop")
                        
        
        layer_conv2=tf.nn.max_pool(value=layer_conv2,
                                       ksize=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',name="conv2_pool")
        
        
        #layer_conv2=tf.nn.relu(layer_conv2,name="conv2_act")    
        
        layer_conv2=self.cabs(layer_conv2)
        tf.summary.histogram('pre_activations_c2', layer_conv2)

        layer_conv2=self.fa(layer_conv2)
        tf.summary.histogram('post_activations_c2', layer_conv2)
        tf.summary.histogram('weihts_c2',weights_conv2)
        
#        layer_conv2=tf.Print(layer_conv2,[layer_conv2],message="Layer conv2 out",summarize=20)
#        weights_conv2=tf.Print(weights_conv2,[weights_conv2],message="Weights conv2 out",summarize=20)
        
        #layer_conv2 = tf.layers.dropout(inputs=layer_conv2, rate=0.1,
         #                             training=self._is_train == tf.estimator.ModeKeys.TRAIN)
        
        
        ##
        
        
#        layer_conv3, weights_conv3 = \
#                        self.new_conv_layer(x_in=layer_conv2,name="conv3",
#                                       num_input_channels=self.num_filters2,
#                                       filter_size=self.filter_size3,
#                                       num_filters=self.num_filters3,
#                                       bitw=self.BITW,
#                                       strides=[1,2,2,1])
#        #layer_conv3=tf.Print(layer_conv3,[weights_conv3],message="Weights conv 3 main",summarize=10)
#
#
#        #layer_conv3=tf.nn.dropout(layer_conv3,keep_prob=0.01,name="conv3_drop")
#                        
#        #layer_conv3=self.fg(layer_conv3)
#              
#        layer_conv3=tf.nn.max_pool(value=layer_conv3,
#                                       ksize=[1, 2, 2, 1],
#                                       strides=[1, 2, 2, 1],
#                                       padding='SAME',name="conv3_pool")
#        
#        #layer_conv3=tf.nn.relu(layer_conv3,name="conv3_act")
#        layer_conv3=self.cabs(layer_conv3)
#        tf.summary.histogram('pre_activations_c3', layer_conv3)
#
#        layer_conv3=self.fa(layer_conv3)
#        tf.summary.histogram('post_activations_c3', layer_conv3)
#        tf.summary.histogram('weights_c3',weights_conv3)
#        
#        
##        layer_conv3=tf.Print(layer_conv3,[layer_conv3],message="Layer conv3 out",summarize=20)
##        weights_conv3=tf.Print(weights_conv3,[weights_conv3],message="Weights conv3 out",summarize=20)
#        
#        #layer_conv3 = tf.layers.dropout(inputs=layer_conv3, rate=0.1,
#          #                            training=self._is_train == tf.estimator.ModeKeys.TRAIN)
#        
#        
#        ##
#        
#        
#        layer_conv4, weights_conv4 = \
#                        self.new_conv_layer(x_in=layer_conv3,name="conv4",
#                                       num_input_channels=self.num_filters3,
#                                       filter_size=self.filter_size4,
#                                       num_filters=self.num_filters4,
#                                       bitw=self.BITW,
#                                       strides=[1,2,2,1])
#        #layer_conv4=tf.Print(layer_conv4,[weights_conv4],message="Weights conv 4 main",summarize=10)
#
#        #layer_conv4=tf.nn.dropout(layer_conv4,keep_prob=0.01,name="conv4_drop")
#                        
#        #layer_conv4=tf.nn.dropout(layer_conv4,keep_prob=0.5)
#                        
#        #layer_conv4=self.fg(layer_conv4)
#        
#        layer_conv4=tf.nn.max_pool(value=layer_conv4,
#                               ksize=[1, 2, 2, 1],
#                               strides=[1, 2, 2, 1],
#                               padding='SAME',name="conv4_pool")
#        
#        
#        
#        #layer_conv4=tf.nn.relu(layer_conv4,name="conv4_act")    
#        layer_conv4=self.cabs(layer_conv4)
#        tf.summary.histogram('pre_activations_c4', layer_conv4)
#
#        layer_conv4=self.fa(layer_conv4)
#        tf.summary.histogram('post_activations_c4', layer_conv4)
#        tf.summary.histogram('weights_c4',weights_conv4)
        
#        layer_conv4=tf.Print(layer_conv4,[layer_conv4],message="Layer conv4 out",summarize=20)
#        weights_conv4=tf.Print(weights_conv4,[weights_conv4],message="Weights conv4 out",summarize=20)
        
       # layer_conv4 = tf.layers.dropout(inputs=layer_conv4, rate=0.1,
                                  #    training=self._is_train == tf.estimator.ModeKeys.TRAIN)
        
#        ##
#        
#        
#        ##
#        
#        
#        
#        layer_conv5, weights_conv5 = \
#                        self.new_conv_layer(x_in=layer_conv4,name="conv5",
#                                       num_input_channels=self.num_filters4,
#                                       filter_size=self.filter_size5,
#                                       num_filters=self.num_filters5,
#                                       bitw=self.BITW)
#                        
#        layer_conv5=self.fg(layer_conv5)
#        
#        
#        layer_conv5=tf.nn.relu(layer_conv5,name="conv5_act")    
#        layer_conv5=self.fa(layer_conv5)
#        
#        ##
#        
#        
#        layer_conv6, weights_conv6 = \
#                        self.new_conv_layer(x_in=layer_conv5,name="conv6",
#                                       num_input_channels=self.num_filters5,
#                                       filter_size=self.filter_size6,
#                                       num_filters=self.num_filters6,
#                                       bitw=self.BITW)
#                        
#        layer_conv6=self.fg(layer_conv6)
#        
#       
#        layer_conv6=tf.nn.relu(layer_conv6,name="conv6_act")    
#        layer_conv6=self.fa(layer_conv6)
#        
#        
#        ##
#        
#        
#        layer_conv7, weights_conv7 = \
#                        self.new_conv_layer(x_in=layer_conv6,name="conv7",
#                                       num_input_channels=self.num_filters6,
#                                       filter_size=self.filter_size7,
#                                       num_filters=self.num_filters7,
#                                       bitw=self.BITW)
#                        
#        layer_conv7=tf.nn.dropout(layer_conv7,keep_prob=0.5)
#                        
#        layer_conv7=self.fg(layer_conv7)
#              
#        
#        layer_conv7=tf.nn.relu(layer_conv7,name="conv7_act")    
#        layer_conv7=self.fa(layer_conv7)
#        
#        
#        ##
#        
#        
#        
#        layer_conv8, weights_conv8 = \
#                        self.new_conv_layer(x_in=layer_conv7,name="conv8",
#                                       num_input_channels=self.num_filters7,
#                                       filter_size=self.filter_size8,
#                                       num_filters=self.num_filters8,
#                                       bitw=self.BITW)
#                        
#        layer_conv8=self.fg(layer_conv8)
#              
#        layer_conv8 = tf.nn.avg_pool(value=layer_conv8,
#                               ksize=[1, 8, 8, 1],
#                               strides=[1, 2, 2, 1],
#                               padding='SAME',name="conv8_pool")
#        
#        layer_conv8=tf.nn.relu(layer_conv8,name="conv8_act")    
#        layer_conv8=self.fa(layer_conv8)
        
        
        ##
        
        
        layer_flat, num_features = self.flatten_layer(layer_conv2)
        
        
        #layer_flat=tf.Print(layer_flat,[layer_conv4],message="layer conv4",summarize=200)
        #layer_flat=tf.Print(layer_flat,[layer_flat],message="layer flat",summarize=200)


        
        layer_fc1,weights_fc1 = self.new_fc_layer(x_in=layer_flat,name="fc1",
                                 num_inputs=num_features,
                                 num_outputs=self.fc1_size,
                                 bitw=self.BITW)
        
        
        #layer_fc1=tf.Print(layer_fc1,[weights_fc1],message="weights fc1 from main",summarize=10)
        
        
        #layer_fc1=self.fg(layer_fc1)

        #layer_fc1=tf.nn.relu(layer_fc1,name="fc1_act")
        layer_fc1=self.cabs(layer_fc1)
        tf.summary.histogram('preact_layer_fc1',layer_fc1)
        layer_fc1=self.fa(layer_fc1)
        tf.summary.histogram('postact_layer_fc1',layer_fc1)
        tf.summary.histogram('weights_fc1',weights_fc1)
        
        
        
        
        
        layer_fc2,weights_fc2 = self.new_fc_layer(x_in=layer_fc1,name="fc2",
                         num_inputs=self.fc1_size,
                         num_outputs=self.fc2_size,
                         bitw=self.BITW)

        
        #layer_fc1=tf.Print(layer_fc1,[weights_fc1],message="weights fc1 from main",summarize=10)
        
        
        #layer_fc1=self.fg(layer_fc1)

        #layer_fc1=tf.nn.relu(layer_fc1,name="fc1_act")
        layer_fc2=self.cabs(layer_fc2)
        tf.summary.histogram('preact_layer_fc2',layer_fc2)
        layer_fc2=self.fa(layer_fc2)
        tf.summary.histogram('postact_layer_fc2',layer_fc2)
        tf.summary.histogram('weights_fc2',weights_fc2)
        
        
#        layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="Layer fc1 out",summarize=20)
#        weights_fc1=tf.Print(weights_fc1,[weights_fc1],message="Weights fc1 out",summarize=20)
        #layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="layer fc1",summarize=200)

        
#        layer_fc1=tf.cond(pred=self._is_train,true_fn=lambda: tf.nn.dropout(layer_fc1,keep_prob=0.95),
#                          false_fn=lambda:tf.identity(layer_fc1),name="fc1_drop")

        
        #layer_fc1 = tf.layers.dropout(inputs=layer_fc1, rate=0.5,
           #                           training=self._is_train == tf.estimator.ModeKeys.TRAIN)
        #layer_fc1=tf.nn.dropout(layer_fc1,keep_prob=0.2,name="fc1_drop")
        
        
        ##
        
#        layer_fc2,weights_fc2 = self.new_fc_layer(x_in=layer_fc1,name="fc2",
#                         num_inputs=self.fc_size,
#                         num_outputs=128,
#                         bitw=self.BITW)
#        
#        
#        layer_fc2=self.fg(layer_fc2)
#
#        layer_fc2=tf.nn.relu(layer_fc2,name="fc2_act")
#        layer_fc2=self.fa(layer_fc2)

        ##
        
        layer_fc0,weights_fc0 = self.new_fc_layer(x_in=layer_fc2,name="fco",
                                 num_inputs=self.fc2_size,
                                 num_outputs=self.num_classes,
                                 bitw=self.BITW)
        
        ##
        #layer_fc0=self.fg(layer_fc0)
        
        tf.summary.histogram('logits_act',layer_fc0)
        tf.summary.histogram('logits_weights',weights_fc0)
#        
        #layer_fc0=tf.Print(layer_fc0,[weights_fc0],message="weights fc0 out",summarize=20)
#        weights_fc0=tf.Print(weights_fc0,[weights_fc0],message="Weights fc0 out",summarize=20)
        
        y_pred = tf.nn.softmax(layer_fc0)
        
        #y_pred=tf.Print(y_pred,[y_pred],message="y pred output",summarize=20)
        
        y_pred_cls = tf.argmax(y_pred, axis=1)
        #y_pred_cls=tf.Print(y_pred_cls,[y_pred_cls],message="y pred cls output",summarize=20)
        
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc0,
                                                                labels=y_true)
       # cross_entropy=tf.clip_by_value(cross_entropy,-10.,10.)
#        cross_entropy=tf.Print(cross_entropy,[layer_fc0],message="fc0 output",summarize=20)
#        cross_entropy=tf.Print(cross_entropy,[weights_conv1],message="weights_conv0 output",summarize=20)
#        cross_entropy=tf.Print(cross_entropy,[cross_entropy],message="cross-entroy output",summarize=20)


        cost = tf.reduce_mean(cross_entropy)
        #cost=self.fg(cost)
        #cost=tf.Print(cost,[cost],message="cost run")
        
#        cost=tf.Print(cost,[weights_conv1],message="weights_conv0 output",summarize=20)
#        cost=tf.Print(cost,[layer_fc0],message="fc0 output",summarize=20)
#        cost=tf.Print(cost,[cross_entropy],message="cross-entroy output",summarize=20)
#        cost=tf.Print(cost,[cost],message="cost output",summarize=20)
        
#       cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_cls,
#                                                                     logits=layer_fc0) 
        
        #cross_entropy=tf.Print(cross_entropy,[y_true_cls,cross_entropy],message="CE output",summarize=20)

        #cost=tf.reduce_mean(cross_entropy)
        
        #cost=tf.Print(cost,[y_true_cls,cross_entropy,cost],message="cost output",summarize=20)
        
        tf.summary.scalar('cost', cost)

        
        
        #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost,global_step=tf.train.create_global_step())
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)
        gvs=optimizer.compute_gradients(cost)
        
        
        
        #optimizer=optimizer.minimize(cost)
        
        #optimizer=tf.Print(optimizer,[for grad,var in gvs],message="gradients",summarize=20)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        capped_gvs = [utils.gvdebug(grad,var) for grad, var in gvs]
        
        #add gradient histograms
        for grad,var in capped_gvs:
            tf.summary.histogram("{}-grad".format(str(var.name)), grad) 

        
        optimizer = optimizer.apply_gradients(capped_gvs)
       
        
        #optimizer=tf.identity(train_op)
        
        #optimizer=tf.Print(optimizer,[y_true_cls,cross_entropy,cost],message="optimizer stage",summarize=20)
        
#        y_pred=tf.nn.softmax(layer_fc0)
#        
#        #y_pred=tf.Print(y_pred,[y_true_cls,cross_entropy,cost],message="y_pred output",summarize=20)
#        
#        y_pred_cls=tf.argmax(y_pred,axis=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        #correct_prediction=tf.Print(correct_prediction,[],message="accuracy run")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #accuracy=tf.Print(accuracy,[y_true_cls,cross_entropy,cost],message="accuracy stage",summarize=20)
                


        
        return y_pred_cls,optimizer,accuracy,[weights_conv1,weights_conv2],gvs
         
        
        
