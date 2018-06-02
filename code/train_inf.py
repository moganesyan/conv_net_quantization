import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import utils
import models
import time
import importlib

#import tensorflow.python.debug as tf_debug

from tensorflow.examples.tutorials.mnist import input_data
#from sklearn.preprocessing import OneHotEncoder



flags = tf.flags
logging = tf.logging

flags.DEFINE_string('data_path', None, 'data_path')
flags.DEFINE_string('config', None, 'config')
flags.DEFINE_string('mode', None, 'mode') 
flags.DEFINE_string('checkpoint', None, 'checkpoint') 
flags.DEFINE_string('save', None, 'save') 



FLAGS = flags.FLAGS



test_batch_size = 1
total_iterations = 0
total_trn_error=0

def get_batch(data,batch_size,num_trn):
    global total_iterations
    ind=min(total_iterations+batch_size,num_trn)
    x=data[0][total_iterations:ind]
    y=data[1][total_iterations:ind]
    
    total_iterations += batch_size

    
    return x,y


def reshuffle(data):
    x=data[0]
    y=data[1]
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    x_n=x[p]
    y_n=y[p]
    
    
    return tuple([x_n,y_n])


def optimize(config,session, data, optimizer, accuracy,grads,saver,ckpt):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    global total_trn_error
    #global BITW,BITA,BITG
    x=config._input_data
    y_true=config._output_data
    bitw=config._bitw
    bita=config._bita
    bitg=config._bitg
    is_train=config._is_train
    data_=data
      
    # Start-time used for printing time-usage below.
    start_time = time.time()
    num_iterations=int((config.num_trn/config.train_batch_size)*config.epochs)
    
    batch_size=config.train_batch_size
    count=0
    epochs=1
    print("starting epoch {} ...".format(epochs))
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('tmp/', session.graph)



    for i in range(0, num_iterations):

        if config.num_trn-count < config.train_batch_size:
            batch_size=config.num_trn-count
            print("Needed remainder batch size is {}".format(batch_size))
        
        
        if int(count)==config.num_trn:
            epochs+=1
            print("Moving to epoch {} after training {} images".format(epochs,count))
            total_iterations=0
            print("reshuffling data...")
            data_=reshuffle(data=data)
            batch_size=config.train_batch_size
            count=0


        
        x_batch, y_true_batch = get_batch(data=data_,batch_size=batch_size,num_trn=config.num_trn)
        count+=batch_size

        x_batch=np.array(x_batch,dtype=np.float32)
        x_batch=x_batch.reshape(batch_size,config.img_size_flat)
        
        x_batch=x_batch/255
        
        y_true_batch=y_true_batch.flatten()
        
        temp=np.zeros((batch_size,config.num_classes))
        temp[np.arange(batch_size),y_true_batch]=1
        
        y_true_batch=temp

        feed_dict_train = {x: x_batch,
                          y_true: y_true_batch,
                          bitw:32,
                          bita:32,
                          bitg:32,
                          is_train:True
                          }
        summary,_=session.run([merged,optimizer], feed_dict=feed_dict_train)

        acc= session.run(accuracy, feed_dict=feed_dict_train)
        total_trn_error=total_trn_error+acc
            

        if config.train_batch_size!=1:    
            msg = "Total iteration: {0:>6}, Training Accuracy of last batch: {1:>6.1%}"
    
            # Print it.
            #train_writer.add_summary(summary,(i+1))

            print(msg.format(total_iterations, total_trn_error))
            
            #print("count is {}".format(count))
            #print(acc)
            total_trn_error=0
            
        elif config.train_batch_size==1:
            if ((i+1)*config.train_batch_size) % 1000 == 0:
                train_writer.add_summary(summary,(i+1))

                #train_writer.add_summary(summary,(i+1)*config.train_batch_size)
                # Calculate the accuracy on the training-set.
                # Message for printing.
                msg = "Total iteration: {0:>6}, Training Accuracy of last 1000 batch: {1:>6.1%}"
    
                # Print it.
                print(msg.format((i+1)*config.train_batch_size, total_trn_error/1000))
                total_trn_error=0            
            
    # Update the total number of iterations performed.
    

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
    if ckpt==True:
        print("Saving checkpoint for epochs {}".format(config.epochs))
        saver.save(session,"tmp/model_ep{}.ckpt".format(config.epochs))
    

def test(config,session,data,y_pred_cls,show_example_errors=False,
                        show_confusion_matrix=False):
#    global BITW,BITA,BITG
    # Number of images in the test-set.
   # num_test = len(data.test.images)
    x=config._input_data
    y_true=config._output_data
    bitw=config._bitw
    bita=config._bita
    bitg=config._bitg
    num_test=config.num_tst
    is_train=config._is_train
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data[0][i:j]
        images=np.array(images,dtype=np.float32)
        
        images =images.reshape(len(images),config.img_size_flat)
        images=images/255
        
        # Get the associated labels.
        labels = data[1][i:j].flatten()
        
        temp=np.zeros((len(images),config.num_classes))
        temp[np.arange(len(images)),labels]=1
        
        labels=temp
        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels,
                     bitw:3,
                     bita:3,
                     bitg:3,
                     is_train:False
                     }
           
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    #cls_true = data.test.cls
    cls_true=data[1]
    
    cls_true=cls_true.flatten()
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test
    #print(acc)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        utils.plot_example_errors(data_true=data,cls_pred=cls_pred, correct=correct,
                                  img_shape=(config.num_channels,config.img_size,config.img_size))

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        utils.plot_confusion_matrix(data_true=data,cls_pred=cls_pred,num_classes=config.num_classes)


#def test_sweep(config,session,data,y_pred_cls,show_example_errors=False,
#                        show_confusion_matrix=False):
##    global BITW,BITA,BITG
#    # Number of images in the test-set.
#   # num_test = len(data.test.images)
#    x=config._input_data
#    y_true=config._output_data
#    bitw=config._bitw
#    bita=config._bita
#    bitg=config._bitg
#    num_test=config.num_tst
#    is_train=config._is_train
#    # Allocate an array for the predicted classes which
#    # will be calculated in batches and filled into this array.
#    cls_pred = np.zeros(shape=num_test, dtype=np.int)
#    #saver_tst=tf.train.Saver()
#
#    # Now calculate the predicted classes for the batches.
#    # We will just iterate through all the batches.
#    # There might be a more clever and Pythonic way of doing this.
#
#    # The starting index for the next batch is denoted i.
#    i = 0
#    #saver_tst.save(session, "tmp/ckpt_test.ckpt")  
#    cls_true=data[1]
#    
#    cls_true=cls_true.flatten()
#    while i < num_test:
#        # The ending index for the next batch is denoted j.
#        j = min(i + test_batch_size, num_test)
#
#        # Get the images from the test-set between index i and j.
#        images = data[0][i:j]
#        images=np.array(images,dtype=np.float32)
#        
#        images =images.reshape(len(images),config.img_size_flat)
#        images=images/255
#        
#        # Get the associated labels.
#        labels = data[1][i:j].flatten()
#        
#        temp=np.zeros((len(images),config.num_classes))
#        temp[np.arange(len(images)),labels]=1
#        
#        labels=temp
#        chosen_prec=32
#        #print("iteration is:{}".format(i))
#        # Create a feed-dict with these images and labels.
#        for prec in range(3,33):
#            #saver_tst.restore(session, "tmp/ckpt_test.ckpt")
#            feed_dict = {x: images,
#                         y_true: labels,
#                         bitw:prec,
#                         bita:prec,
#                         bitg:prec,
#                         is_train:False
#                         }
#            cls_pred_comp= session.run(y_pred_cls, feed_dict=feed_dict)
#            cls_true_comp=cls_true[i:j]
#            correct_comp=(cls_true_comp==cls_pred_comp)
#            if correct_comp==True:
#                chosen_prec=prec
#                #print(chosen_prec)
#                break
#        print("chosen prec is:{}".format(chosen_prec))
#
#        feed_dict = {x: images,
#                     y_true: labels,
#                     bitw:chosen_prec,
#                     bita:chosen_prec,
#                     bitg:chosen_prec,
#                     is_train:False
#                     }          
#            
#
#        # Calculate the predicted class using TensorFlow.
#        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
#
#        # Set the start-index for the next batch to the
#        # end-index of the current batch.
#        i = j
#
#    # Convenience variable for the true class-numbers of the test-set.
#    #cls_true = data.test.cls
#
#    # Create a boolean array whether each image is correctly classified.
#    correct = (cls_true == cls_pred)
#
#    # Calculate the number of correctly classified images.
#    # When summing a boolean array, False means 0 and True means 1.
#    correct_sum = correct.sum()
#
#    # Classification accuracy is the number of correctly classified
#    # images divided by the total number of images in the test-set.
#    acc = float(correct_sum) / num_test
#    #print(acc)
#
#    # Print the accuracy.
#    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
#    print(msg.format(acc, correct_sum, num_test))
#
#    # Plot some examples of mis-classifications, if desired.
#    if show_example_errors:
#        print("Example errors:")
#        utils.plot_example_errors(data_true=data,cls_pred=cls_pred, correct=correct,
#                                  img_shape=(config.num_channels,config.img_size,config.img_size))
#
#    # Plot the confusion matrix, if desired.
#    if show_confusion_matrix:
#        print("Confusion Matrix:")
#        utils.plot_confusion_matrix(data_true=data,cls_pred=cls_pred,num_classes=config.num_classes)
        
      
        
        
        
def test_sweep(config,session,data,y_pred_cls,show_example_errors=False,
                        show_confusion_matrix=False):
#    global BITW,BITA,BITG
    # Number of images in the test-set.
   # num_test = len(data.test.images)
    x=config._input_data
    y_true=config._output_data
    bitw=config._bitw
    bita=config._bita
    bitg=config._bitg
    num_test=config.num_tst
    is_train=config._is_train
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    #saver_tst=tf.train.Saver()

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0
    #saver_tst.save(session, "tmp/ckpt_test.ckpt")  
    cls_true=data[1]
    
    cls_true=cls_true.flatten()
    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data[0][i:j]
        images=np.array(images,dtype=np.float32)
        
        images =images.reshape(len(images),config.img_size_flat)
        images=images/255
        
        # Get the associated labels.
        labels = data[1][i:j].flatten()
        
        temp=np.zeros((len(images),config.num_classes))
        temp[np.arange(len(images)),labels]=1
        
        labels=temp
        chosen_prec=3
        #print("iteration is:{}".format(i))
        # Create a feed-dict with these images and labels.

        feed_dict = {x: images,
                     y_true: labels,
                     bitw:chosen_prec,
                     bita:chosen_prec,
                     bitg:chosen_prec,
                     is_train:False
                     }
        cls_pred_comp= session.run(y_pred_cls, feed_dict=feed_dict)
        cls_true_comp=cls_true[i:j]
        correct_comp=(cls_true_comp==cls_pred_comp)
        if correct_comp==False:
            chosen_prec=32
            
        feed_dict = {x: images,
             y_true: labels,
             bitw:chosen_prec,
             bita:chosen_prec,
             bitg:chosen_prec,
             is_train:False
             }      
        print("chosen prec is:{}".format(chosen_prec))

        
            

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    #cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test
    #print(acc)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        utils.plot_example_errors(data_true=data,cls_pred=cls_pred, correct=correct,
                                  img_shape=(config.num_channels,config.img_size,config.img_size))

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        utils.plot_confusion_matrix(data_true=data,cls_pred=cls_pred,num_classes=config.num_classes)
        

def get_config():
    
    return importlib.import_module(FLAGS.config).Config()



def main(_):
    
    if "cifar10" in str(FLAGS.config):
        data=tf.keras.datasets.cifar10.load_data()

        config=get_config()
        model=models.cifarnetModel(is_training=True,config=config)


    elif "mnist" in str(FLAGS.config):
        data=tf.keras.datasets.mnist.load_data()
        config=get_config()
        model=models.lenetModel(is_training=True,config=config)
    else:
        raise ValueError("Unspecified dataset")
    
    data_trn=data[0]
    data_tst=data[1]

    
    y_pred_cls,optimizer,accuracy,weights,gvs=model.get_model()


    session = tf.Session()  
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    
    if "yes" in str(FLAGS.checkpoint):
        #depending on epochs restore the correct checkpoint
        if config.epochs==1:
            saver.restore(session,"tmp/model_ep1.ckpt")
        elif config.epochs==5:
            saver.restore(session,"tmp/model_ep5.ckpt")
        elif config.epochs==10:
            saver.restore(session,"tmp/model_ep10.ckpt")
        elif config.epochs==40:
            saver.restore(session,"tmp/model_ep40.ckpt")
        #if we're in normal mode of testing              
        if "normal" in str(FLAGS.mode):    
            test(config=model,session=session,data=data_tst,y_pred_cls=y_pred_cls,
                            show_confusion_matrix=True,show_example_errors=True)
        #if we're in the sweep mode of training            
        elif "sweep" in str(FLAGS.mode):    
            test_sweep(config=model,session=session,data=data_tst,y_pred_cls=y_pred_cls,
                            show_confusion_matrix=True,show_example_errors=True)
            
    elif "no" in str(FLAGS.checkpoint):  
        #if we're not using the checkpoint
        #start checking if we want to make a checkpoint during training
        if "yes" in str(FLAGS.save):
            #make checkpoint
            optimize(config=model,session=session,data=data_trn,optimizer=optimizer,
                     accuracy=accuracy,grads=gvs,saver=saver,ckpt=True)
        elif "no" in str(FLAGS.save):
            #dont make the checkpoint
            optimize(config=model,session=session,data=data_trn,optimizer=optimizer,
                     accuracy=accuracy,grads=gvs,saver=saver,ckpt=False)
            
        if "normal" in str(FLAGS.mode):                           
            test(config=model,session=session,data=data_tst,y_pred_cls=y_pred_cls,
                            show_confusion_matrix=True,show_example_errors=True)
            
        elif "sweep" in str(FLAGS.mode):
            test_sweep(config=model,session=session,data=data_tst,y_pred_cls=y_pred_cls,
                            show_confusion_matrix=True,show_example_errors=True)       


    utils.plot_conv_weights(session=session,config=model,weights=weights[1])
    
if __name__ == "__main__":
    tf.app.run()