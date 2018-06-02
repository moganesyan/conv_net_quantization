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

#import random
#def reshuffle(data):
#    
#    x=data[0]
#    y=data[1]
#    
#    tmp=zip(x,y)
#    random.shuffle(tmp)
#    x_n,y_n=zip(*tmp)
#    
#    shuffled=[list(x_n),list(y_n)]
#    
#    return shuffled



def reshuffle(data):
    x=data[0]
    y=data[1]
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    x_n=x[p]
    y_n=y[p]
    
    
    return tuple([x_n,y_n])



def optimize_sweep(config,session, data, optimizer, accuracy,saver):
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
    num_iterations=int((config.num_trn/1)*config.epochs)
    
    batch_size=1 #force batch size 1 of sweeper
    count=0
    epochs=1
    print("starting epoch {} ...".format(epochs))
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('tmp/', session.graph)
    saver.save(session, "tmp/model.ckpt")    
    
    prec_array=np.empty((num_iterations,config.img_size_flat+1))

    for i in range(0, num_iterations):

        if config.num_trn-count < 1:
            batch_size=config.num_trn-count
            print("Needed remainder batch size is {}".format(batch_size))
        
        
        if int(count)==config.num_trn:
            epochs+=1
            print("Moving to epoch {} after training {} images".format(epochs,count))
            total_iterations=0
            print("reshuffling data...")
            data_=reshuffle(data=data)
            batch_size=1
            count=0


        
        x_batch, y_true_batch = get_batch(data=data_,batch_size=batch_size,num_trn=config.num_trn)
        count+=batch_size
        x_batch=np.array(x_batch,dtype=np.float32)
        x_batch=x_batch.reshape(batch_size,config.img_size_flat)
        
        x_batch=x_batch/255
        
        y_true_batch=y_true_batch.flatten()
        
        #print(y_true_batch)
        temp=np.zeros((batch_size,config.num_classes))
        temp[np.arange(batch_size),y_true_batch]=1
#        print(x_batch)
#
#        print(y_true_batch)
#        print(temp)

        
        y_true_batch=temp
        
        temp_arr=[]
        for prec in range(4,33):
            saver.restore(session, "tmp/model.ckpt")
            #print("Model restored.")
            #print("testing precision: {}".format(prec))
        
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch,
                               bitw:prec,
                               bita:prec,
                               bitg:prec,
                               is_train:True}
        
            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            #print("Fucks up here1")
            #print("Fucks up here2")
            session.run(optimizer, feed_dict=feed_dict_train)
            acc=session.run(accuracy,feed_dict=feed_dict_train)
            temp_arr.append(acc)
        #chosen_prec=temp.index(max(temp))+1
        if max(temp_arr)==0:
            chosen_prec=32
        else:
            chosen_prec=temp_arr.index(max(temp_arr))+4
            
        saver.restore(session, "tmp/model.ckpt")
        #fw,fa,fg=get_dorefa(chosen_prec,chosen_prec,chosen_prec)
        feed_dict_train = {x: x_batch,
                   y_true: y_true_batch,
                   bitw:chosen_prec,
                   bita:chosen_prec,
                   bitg:chosen_prec,
                   is_train:True}
        #BITW,BITA,BITG=[chosen_prec,chosen_prec,chosen_prec]

        # Print status every 100 iterations.
        #if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            
            
        #save line to csv 
#        if i==1:
#            line_to_save=np.append(x_batch,chosen_prec)
#            row_to_save=line_to_save
#        elif i>1:
#            line_to_save=np.append(x_batch,chosen_prec)
#            row_to_save=np.stack(row_to_save,line_to_save,axis=0)
#        #print(line_to_save.shape)
##        np.savetxt(f,np.transpose(line_to_save),delimiter=",") 
#            df=pd.DataFrame(row_to_save)
#            df.to_csv("mnist.csv")
            
        line_to_save=np.append(x_batch,chosen_prec)
        prec_array[i,:]=line_to_save
       # print(df.head)
#        print(prec_array)
#        print(line_to_save.shape)
       # df.to_csv("mnist.csv")    
            
        session.run(optimizer, feed_dict=feed_dict_train)

        summary,acc = session.run([merged,accuracy], feed_dict=feed_dict_train)
        total_trn_error=total_trn_error+acc

        
        # Message for printing.
        #if i%100==0:


        #print("Iteration is: {}".format(i))
        #print("Chosen precision is: {}".format(chosen_prec))
        #print("The contents of the list are ...")
        #print(temp_arr)
        
        if ((i+1)*1) % 1000 == 0:

            # Calculate the accuracy on the training-set.
            # Message for printing.
            train_writer.add_summary(summary,(i+1)*config.train_batch_size)

            msg = "Total iteration: {0:>6}, Training Accuracy of last 1000 batch: {1:>6.1%}"
        
            # Print it.
            print(msg.format((i+1)*1, total_trn_error/1000))
            total_trn_error=0    
            
        saver.save(session, "tmp/model.ckpt")
        
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    df=pd.DataFrame(prec_array)
    df.to_csv("cifar10_01.csv")
        
        
        

def optimize(config,session, data, optimizer, accuracy,grads):
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

    data_new_x=[]
    data_new_y=[]


    for i in range(0, num_iterations):

        if config.num_trn-count < config.train_batch_size:
            batch_size=config.num_trn-count
            print("Needed remainder batch size is {}".format(batch_size))
        
        
        if int(count)==config.num_trn:
            epochs+=1
            print("Moving to epoch {} after training {} images".format(epochs,count))
            total_iterations=0
            print("reshuffling data...")
            
            data_=tuple((np.concatenate(data_new_x,axis=0),np.concatenate(data_new_y,axis=0)))
            
            df_tmp=pd.DataFrame(np.concatenate(data_new_y,axis=0))
            df_tmp.to_csv("precs_{}.csv".format(epochs))
            data_new_x=[]
            data_new_y=[]
            
            
            
            data_=reshuffle(data=data_)
            batch_size=config.train_batch_size
            count=0


        
        x_batch, y_true_batch = get_batch(data=data_,batch_size=batch_size,num_trn=config.num_trn)
        count+=batch_size
#        print(1)
#        print(x_batch.shape)
#        print(y_true_batch.shape)
#        print("iteration is {}".format(i))
#        print(config.train_batch_size)
#        print(len(x_batch))
#        print(len(y_true_batch))
#        print("Total iterations are {}".format(total_iterations))
#        print("y true batch")
#        print(y_true_batch)
#        print("x batch")
#        print(x_batch)
        #print(temp)
        #print(len(x_batch))
        x_batch_proc=np.array(x_batch,dtype=np.float32)
        x_batch_proc=x_batch_proc.reshape(batch_size,config.img_size_flat)
        
        x_batch_proc=x_batch_proc/255
        #y_true_batch=y_true_batch.flatten()
        #print(y_true_batch)

        #print(y_true_batch)
        temp=np.zeros((batch_size,config.num_classes))
        temp[np.arange(batch_size),y_true_batch[:,0]]=1
        #print(temp)
#        print(x_batch)
#
#        print(y_true_batch)
#        print(temp)

        y_true_batch_precs=y_true_batch[:,1]
        y_true_batch_onehot=temp

        #print(y_true_batch_precs[0])
#        print(y_true_batch)
        
        #print(y_true_batch_precs[0])
        feed_dict_train = {x: x_batch_proc,
                          y_true: y_true_batch_onehot,
                          bitw:y_true_batch_precs[0],
                          bita:y_true_batch_precs[0],
                          bitg:y_true_batch_precs[0],
                          is_train:True
                          }
        summary,_=session.run([merged,optimizer], feed_dict=feed_dict_train)

        acc= session.run(accuracy, feed_dict=feed_dict_train)
        #print(acc)
        #print(y_true_batch_precs)
        #print(y_true_batch)

        if acc==1:
            y_true_batch[:,1][0]=3
        #print(y_true_batch)
#        print(2)
#        print(x_batch.shape)
#        print(y_true_batch.shape)

        data_new_x.append(x_batch)
        data_new_y.append(y_true_batch)
        
#        lol_x=np.concatenate(data_new_x,axis=0)
#        lol_y=np.concatenate(data_new_y,axis=0)
        
#        print(3)
#        print(lol_x.shape)
#        print(lol_y.shape)
        
        #print(len(data_new))
        #print(np.array(y_true_batch))
#        print("checker is"
#        print(checker)
        total_trn_error=total_trn_error+acc

#        if i % 200 == 0:
#            # Calculate the accuracy on the training-set.
#            # Message for printing.
#            msg = "Total trn error so far: {0:>6}, Training Accuracy: {1:>6.1%}"
#
#            # Print it.
#            print(msg.format(i + 1, total_trn_error/(i+1)))
#        count+=1
        
        
#        if ((i+1)*config.train_batch_size) % 1000 == 0:
#
#            # Calculate the accuracy on the training-set.
#            # Message for printing.
#            msg = "Total iteration: {0:>6}, Training Accuracy of last 1000 batch: {1:>6.1%}"
#
#            # Print it.
#            print(msg.format((i+1)*config.train_batch_size, total_trn_error/1000))
#            total_trn_error=0
        
#        for gv in grads:
#            print(str(gv[1].name))
#            
#            g_temp=np.array(session.run(gv[0],feed_dict=feed_dict_train))
#            g_thresh=g_temp>10
#            g_nans=np.isnan(g_temp)
#            print("max value is {}".format(np.max(g_temp)))
#            print("Threshold")
#            print(g_temp[g_thresh==True])
#            print("is_nan")
#            print(g_temp[g_nans==True])
        
        
           # print(session.run(gv[0],feed_dict=feed_dict_train))
            #print(gv[0])
            #print(str(session.run(gv[0],feed_dict=feed_dict_train)) + " - " + gv[1].name)
            
        #train_writer.add_summary(summary,(i+1))

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
            
        #count+=batch_size
            


    # Update the total number of iterations performed.
    

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
    



def optimize_dynamic(config,session, data, optimizer, accuracy,grads):
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
#        print("iteration is {}".format(i))
#        print(config.train_batch_size)
#        print(len(x_batch))
#        print(len(y_true_batch))
#        print("Total iterations are {}".format(total_iterations))
#        print("y true batch")
#        print(y_true_batch)
#        print("x batch")
#        print(x_batch)
        #print(temp)
        #print(len(x_batch))
        x_batch=np.array(x_batch,dtype=np.float32)
        x_batch=x_batch.reshape(batch_size,config.img_size_flat)
        
        x_batch=x_batch/255
        
        y_true_batch=y_true_batch.flatten()
        
        #print(y_true_batch)
        temp=np.zeros((batch_size,config.num_classes))
        temp[np.arange(batch_size),y_true_batch]=1
#        print(x_batch)
#
#        print(y_true_batch)
#        print(temp)

        
        y_true_batch=temp

        feed_dict_train = {x: x_batch,
                          y_true: y_true_batch,
                          bitw:config.BITW,
                          bita:config.BITA,
                          bitg:config.BITG,
                          is_train:True
                          }
        summary,_=session.run([merged,optimizer], feed_dict=feed_dict_train)

        acc= session.run(accuracy, feed_dict=feed_dict_train)
#        print("checker is"
#        print(checker)
        total_trn_error=total_trn_error+acc

#        if i % 200 == 0:
#            # Calculate the accuracy on the training-set.
#            # Message for printing.
#            msg = "Total trn error so far: {0:>6}, Training Accuracy: {1:>6.1%}"
#
#            # Print it.
#            print(msg.format(i + 1, total_trn_error/(i+1)))
#        count+=1
        
        
#        if ((i+1)*config.train_batch_size) % 1000 == 0:
#
#            # Calculate the accuracy on the training-set.
#            # Message for printing.
#            msg = "Total iteration: {0:>6}, Training Accuracy of last 1000 batch: {1:>6.1%}"
#
#            # Print it.
#            print(msg.format((i+1)*config.train_batch_size, total_trn_error/1000))
#            total_trn_error=0
        
#        for gv in grads:
#            print(str(gv[1].name))
#            
#            g_temp=np.array(session.run(gv[0],feed_dict=feed_dict_train))
#            g_thresh=g_temp>10
#            g_nans=np.isnan(g_temp)
#            print("max value is {}".format(np.max(g_temp)))
#            print("Threshold")
#            print(g_temp[g_thresh==True])
#            print("is_nan")
#            print(g_temp[g_nans==True])
        
        
           # print(session.run(gv[0],feed_dict=feed_dict_train))
            #print(gv[0])
            #print(str(session.run(gv[0],feed_dict=feed_dict_train)) + " - " + gv[1].name)
            
        train_writer.add_summary(summary,(i+1))

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
                
                #train_writer.add_summary(summary,(i+1)*config.train_batch_size)
                # Calculate the accuracy on the training-set.
                # Message for printing.
                msg = "Total iteration: {0:>6}, Training Accuracy of last 1000 batch: {1:>6.1%}"
    
                # Print it.
                print(msg.format((i+1)*config.train_batch_size, total_trn_error/1000))
                total_trn_error=0            
            
        #count+=batch_size
            


    # Update the total number of iterations performed.
    

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    





def print_test_accuracy(config,session,data,y_pred_cls,show_example_errors=False,
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
        
        #print(len(images))
        
        images =images.reshape(len(images),config.img_size_flat)
        images=images/255
        
        

        # Get the associated labels.
        labels = data[1][i:j].flatten()
        
        temp=np.zeros((len(images),config.num_classes))
        temp[np.arange(len(images)),labels]=1
        
        
#        print(labels)
#        print(temp)

        
        labels=temp
        
        #print(labels)


        # Create a feed-dict with these images and labels.
        if "normal" in str(FLAGS.mode):
            feed_dict = {x: images,
                         y_true: labels,
                         bitw:32,
                         bita:32,
                         bitg:32,
                         is_train:False
                         }
        elif "sweep" in str(FLAGS.mode):
            feed_dict = {x: images,
                         y_true: labels,
                         bitw:32,
                         bita:32,
                         bitg:32,
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


def get_config():
    
    return importlib.import_module(FLAGS.config).Config()



def main(_):
    
#    data = input_data.read_data_sets('data/MNIST/', one_hot=True)
#    data.test.cls = np.argmax(data.test.labels, axis=1)
#
#    
#    print("Size of:")
#    print("- Training-set:\t\t{}".format(len(data.train.labels)))
#    print("- Test-set:\t\t{}".format(len(data.test.labels)))
#    print("- Validation-set:\t{}".format(len(data.validation.labels))) 
    #data=tf.keras.datasets.mnist.load_data()
    if "cifar10" in str(FLAGS.config):
        data=tf.keras.datasets.cifar10.load_data()
        data_trn=data[0]
        data_tst=data[1]
        
        
        tmp_y=data_trn[1]
        tmp_x=data_trn[0]
        tmp_y_aug=32*np.ones_like(tmp_y)
        new_y=np.concatenate([tmp_y,tmp_y_aug],axis=1)
        
        data_trn=tuple((tmp_x,new_y))
        config=get_config()
        model=models.cifarnetModel(is_training=True,config=config)


    elif "mnist" in str(FLAGS.config):
        data=tf.keras.datasets.mnist.load_data()
        data_trn=data[0]
        data_tst=data[1]
        
        
        tmp_y=data_trn[1].reshape(60000,1)
        tmp_x=data_trn[0]
        tmp_y_aug=32*np.ones_like(tmp_y)
        #print(tmp_y.shape)
        #print(tmp_y_aug.shape)
        new_y=np.concatenate([tmp_y,tmp_y_aug],axis=1)
        #print(new_y.shape)
        
        data_trn=tuple((tmp_x,new_y))
        config=get_config()
        model=models.lenetModel(is_training=True,config=config)
    else:
        raise ValueError("Unspecified dataset")
    

    #print(str(FLAGS.config))
    
    #config=get_config()

    #data.test.cls = np.argmax(data.test.labels, axis=1)
    
    #model=models.lenetModel(is_training=True,config=config)
    
    #iters=(config.num_trn/config.train_batch_size)*config.epochs
    
    y_pred_cls,optimizer,accuracy,weights,gvs=model.get_model()

    #saver = tf.train.Saver()
    #CREATE SESSION
    session = tf.Session()
#    session = tf_debug.LocalCLIDebugWrapperSession(session)
#    session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
#    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
    

    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    if "normal" in str(FLAGS.mode):
        optimize(config=model,session=session,data=data_trn,optimizer=optimizer,
                  accuracy=accuracy,grads=gvs)
        
    elif "sweep" in str(FLAGS.mode):    
        optimize_sweep(config=model,session=session,data=data_trn,optimizer=optimizer,
                 accuracy=accuracy,saver=saver)
        
    print_test_accuracy(config=model,session=session,data=data_tst,y_pred_cls=y_pred_cls,
                        show_confusion_matrix=True,show_example_errors=True)

    utils.plot_conv_weights(session=session,config=model,weights=weights[1])
    
if __name__ == "__main__":
    tf.app.run()





