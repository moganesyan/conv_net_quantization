import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from dorefa2 import get_dorefa
import dorefa


tf.__version__
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Convolutional Layer 3.
filter_size3 = 5          # Convolution filters are 5 x 5 pixels.
num_filters3 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# IMPORT MNIST DATASET
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

num_trn=len(data.train.labels)
num_tst=len(data.test.labels)
num_val=len(data.validation.labels)

#comvert labels from one-hot to values
data.test.cls = np.argmax(data.test.labels, axis=1)

#DATA DIMENSIONS
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

#Helper functions
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

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
    
#Plots few images    
    
## Get the first images from the test-set.
#images = data.test.images[0:9]
#
## Get the true classes for those images.
#cls_true = data.test.cls[0:9]
#
## Plot the images and labels using our helper-function above.
#plot_images(images=images, cls_true=cls_true)
    
#BUILDING GRAPH    
    
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(x_in, name,             # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,    # Number of filters.
                   bitw,      #bitwidth of weights 
                   use_pooling=True
                   ):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    global fw
    # Create new weights aka. filters with the given shape.
#    W_init=tf.truncated_normal(shape,stddev=0.05)
#    weights=tf.get_variable('W',initializer=W_init)
#    
#    B_init=tf.constant(0.05,shape=[num_filters])
#    biases=tf.get_variable('b',initializer=B_init)
    
    #weights = new_weights(shape=shape)
    

    # Create new biases, one for each filter.
    #biases = new_biases(length=num_filters)
    
#    with tf.name_scope(name):
    
#    with tf.variable_scope(name,reuse=tf.AUTO_REUSE),dorefa.replace_variable(
#                lambda x: fw(x)):
    if "conv1" in name:
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
             weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())

    else:
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE),dorefa.replace_variable(
                    lambda x: fw(x)):
            #weights=tf.Variable(tf.truncated_normal(shape, stddev=0.05),name="w")
        
            #biases=tf.Variable(tf.constant(0.05, shape=[num_filters]),name="b")
            weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
           

    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        biases=tf.get_variable(name="b",shape=[num_filters],initializer=tf.constant_initializer())


    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
#    if "conv1" not in weights.name:
#        weights_quant=tf.map_fn(fw,weights)
#    else:
#        weights_quant=weights
        
    #weights_reassign=tf.assign(weights,weights_quant,use_locking=False)
    
    #dud=weights_reassign+biases
    
    #weights_print=tf.Print(weights_reassign,[weights_reassign],message="The input to conv2d is")
    
    
    #weights_print=tf.Print(weights,[weights],message="The weights of {} are".format(weights.name))
    layer = tf.nn.conv2d(input=x_in,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    #layer += biases

    # Use pooling to down-sample the image resolution?
#    if use_pooling:
#        # This is 2x2 max-pooling, which means that we
#        # consider 2x2 windows and select the largest value
#        # in each window. Then we move 2 pixels to the next window.
#        layer = tf.nn.max_pool(value=layer,
#                               ksize=[1, 2, 2, 1],
#                               strides=[1, 2, 2, 1],
#                               padding='SAME')
#
#    # Rectified Linear Unit (ReLU).
#    # It calculates max(x, 0) for each input pixel x.
#    # This adds some non-linearity to the formula and allows us
#    # to learn more complicated functions.
#    ret = tf.nn.relu(layer,name=name)
#    ret.variables.w=weights
#    ret.variables.b=biases
#    ret.variables.W=weights
#    ret.variables.b=biases

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features



def new_fc_layer(x_in,name,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 bitw, #bitwidth of layer
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    global fw
    # Create new weights and biases.
#    weights = new_weights(shape=[num_inputs, num_outputs])
#    biases = new_biases(length=num_outputs)
    if "fco" not in name:
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE),dorefa.replace_variable(
                    lambda x: fw(x)):
            #weights=tf.Variable(tf.truncated_normal([num_inputs,num_outputs], stddev=0.05),name="w")
        
            #biases=tf.Variable(tf.constant(0.05, shape=[num_outputs]),name="b")  
            shape=[num_inputs,num_outputs]
            #weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
            weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
    
    #        weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
            
    else:
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            shape=[num_inputs,num_outputs]
            #weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())
            weights=tf.get_variable(name="w",shape=shape,initializer=tf.glorot_normal_initializer())

            
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        biases=tf.get_variable(name="b",shape=[num_outputs],initializer=tf.constant_initializer())
#    print(weights.name)
#    print(biases.name)
    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
#    if "fco" in weights.name:
#        weights_quant=tf.map_fn(fw,weights)
#    else:
#        weights_quant=weights
    #weights_reassign=tf.assign(weights,weights_quant,use_locking=False)
    #weights_print=tf.Print(weights,[weights],message="The weights of {} are".format(weights.name))

    #layer = tf.matmul(input, weights) + biases
    layer=tf.matmul(x_in,weights)
    #layer = tf.matmul(input, weights_reassign) + biases

#    # Use ReLU?
#    if use_relu:
#        layer = tf.nn.relu(layer)

    return layer


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    global total_trn_error
    global BITW,BITA,BITG
      
    # Start-time used for printing time-usage below.
    start_time = time.time()
    #saver.save(session, "/tmp/model.ckpt")

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        #print("Gets batch")
        feed_dict_train = {x: x_batch,
                          y_true: y_true_batch,
                          bitw:BITW,
                          bita:BITA,
                          bitg:BITG,
                          phase_train:True}
        #print("passed")
        #print("Model saved in path: %s" % save_path)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
#        temp=[]
#        for prec in range(1,9):
#            saver.restore(session, "/tmp/model.ckpt")
#            #print("Model restored.")
#            #print("testing precision: {}".format(prec))
#        
#            feed_dict_train = {x: x_batch,
#                               y_true: y_true_batch}
#        
#            # Run the optimizer using this batch of training data.
#            # TensorFlow assigns the variables in feed_dict_train
#            # to the placeholder variables and then runs the optimizer.
#            #print("Fucks up here1")
#            #fw,fa,fg=get_dorefa(prec,prec,prec)
#            BITW,BITA,BITG=[prec,prec,prec]
#            #print("Fucks up here2")
#            session.run(optimizer, feed_dict=feed_dict_train)
#            acc=session.run(accuracy,feed_dict=feed_dict_train)
#            temp.append(acc)
#        #chosen_prec=temp.index(max(temp))+1
#        if max(temp)==0:
#            chosen_prec=32
#        else:
#            chosen_prec=temp.index(max(temp))+1
#            
        
        #fw,fa,fg=get_dorefa(chosen_prec,chosen_prec,chosen_prec)
        #chosen_prec=32
        #BITW,BITA,BITG=[chosen_prec,chosen_prec,chosen_prec]

        # Print status every 100 iterations.
        #if i % 100 == 0:
            # Calculate the accuracy on the training-set.
        session.run(optimizer, feed_dict=feed_dict_train)

        acc = session.run(accuracy, feed_dict=feed_dict_train)
        total_trn_error=total_trn_error+acc

        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            # Message for printing.
            msg = "Total trn error so far: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, total_trn_error/(i+1)))


    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
    
    
def plot_example_errors(cls_pred, correct):

    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):

    cls_true = data.test.cls
    
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

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    global BITW,BITA,BITG
    # Number of images in the test-set.
    num_test = len(data.test.images)

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
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels,
                     bitw:BITW,
                     bita:BITA,
                     bitg:BITG,
                     phase_train:False}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

def cabs(x):
    return tf.minimum(1.0, tf.abs(x), name='cabs')
##var definitions
    

def batch_norm_conv(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_fc(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed



total_iterations = 0
total_trn_error=0

BITW,BITA,BITG=[1,2,2]
sum_size=20

bitw,bita,bitg=[tf.placeholder(dtype=tf.float32,shape=(),name='bitw'),
                tf.placeholder(dtype=tf.float32,shape=(),name='bita'),
                tf.placeholder(dtype=tf.float32,shape=(),name='bitg')]

phase_train=tf.placeholder(dtype=tf.bool,name="phase_train") # training or not training

train_batch_size = 1      

epochs=0.1

iters=(num_trn/train_batch_size)*epochs

fw,fa,fg=get_dorefa(bitw,bita,bitg,phase_train)
print("Fails here1")





#placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

#x_image=tf.Print(x_image,[x_image],message="Image is:",summarize=200)

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

#create first conv layer





layer_conv1, weights_conv1 = \
    new_conv_layer(x_in=x_image,name="conv1",
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   bitw=bitw,
                   use_pooling=True)
    
    
#quantize first layer 
    
print("Passed creation of layer 1") 
#layer_conv1=fg(layer_conv1)
#layer_conv1=fa(cabs(layer_conv1))
#layer_conv1=fg(layer_conv1)
#layer_conv1=tf.Print(layer_conv1,[layer_conv1],message="conv1",summarize=sum_size)

layer_conv1 = tf.nn.max_pool(value=layer_conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')



#layer_conv1=tf.Print(layer_conv1,[layer_conv1],message="after maxpool conv1",summarize=sum_size)
#layer_conv1=fg(layer_conv1)
#layer_conv1=tf.map_fn(fa,layer_conv1,name="conv1_activated")


#layer_conv1=batch_norm_conv(layer_conv1,num_filters1,phase_train)
#layer_conv1=tf.Print(layer_conv1,[layer_conv1],message="after batchnorm conv1",summarize=5)

#layer_conv1=tf.nn.relu(layer_conv1,name="conv1_act")
layer_conv1=cabs(layer_conv1)
#layer_conv1=tf.Print(layer_conv1,[layer_conv1],message="after cabs conv1",summarize=sum_size)

layer_conv1=fa(layer_conv1)
#layer_conv1=tf.Print(layer_conv1,[layer_conv1],message="after act conv1",summarize=sum_size)

#layer_conv1_print=tf.Print(layer_conv1,[layer_conv1],message="The values of conv1",summarize=100)

print("Passed cabs1")    
#layer_conv1=fa(layer_conv1)
#layer_conv1=tf.py_func(fa,[tf.py_func(cabs,[layer_conv1],tf.float32)],tf.float32)

print("Passed activation of layer 1 ")

#create second conv layer
layer_conv2, weights_conv2 = \
    new_conv_layer(x_in=layer_conv1,name="conv2",
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   bitw=bitw,
                   use_pooling=True)

#layer_conv2=fw(layer_conv2)
    
    
#layer_conv2_grad=fg(layer_conv2)
#layer_conv2_act=fa(cabs(layer_conv2))
#layer_conv2=fg(layer_conv2)
#layer_conv2=tf.Print(layer_conv2,[layer_conv2],message="conv2",summarize=sum_size)

layer_conv2=fg(layer_conv2)
#layer_conv2=tf.Print(layer_conv2,[layer_conv2],message="after fg conv2",summarize=sum_size)



layer_conv2=tf.nn.max_pool(value=layer_conv2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

#layer_conv2=tf.Print(layer_conv2,[layer_conv2],message="after maxpool conv2",summarize=sum_size)

#layer_conv2=tf.map_fn(fa,layer_conv2,name="conv2_activated")


#layer_conv2=batch_norm_conv(layer_conv2,num_filters2,phase_train)
#layer_conv2=tf.Print(layer_conv2,[layer_conv2],message="after batchnorm conv2",summarize=5)


#layer_conv2=tf.nn.relu(layer_conv2,name="conv2_act")    

layer_conv2=cabs(layer_conv2)
#layer_conv2=tf.Print(layer_conv2,[layer_conv2],message="after cabs conv2",summarize=sum_size)

layer_conv2=fa(layer_conv2)
#layer_conv2=tf.Print(layer_conv2,[layer_conv2],message="after act conv2",summarize=sum_size)

    
#layer_conv2_print_grad = tf.Print(layer_conv2_act, [layer_conv2_grad], message="Contents of grad quantised tensor are: ")   
#layer_conv2_print_act = tf.Print(layer_conv2_print_grad, [layer_conv2_act], message="Contents of activated quantised tensor are: ")   

#    _act
#layer_conv2=tf.py_func(fw,[layer_conv2],tf.float32)
#layer_conv2=tf.py_func(fg,[layer_conv2],tf.float32)    
#layer_conv2=tf.py_func(fa,[tf.py_func(cabs,[layer_conv2],tf.float32)],tf.float32)
    
#print("Passed activation of layer 2 ")

    
layer_flat, num_features = flatten_layer(layer_conv2)
#print("Passed flattening ")


##FC layer 1
#layer_fc1 = new_fc_layer(input=layer_flat,name="fc1",
#                         num_inputs=num_features,
#                         num_outputs=fc_size,
#                         bitw=bitw,
#                         use_relu=True)
#
##layer_fc1=fw(layer_fc1)
#
#
##layer_fc1=fg(layer_fc1)
##layer_fc1=fg(layer_fc1)
#layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="fc1",summarize=5)
#
#
#layer_fc1=fg(layer_fc1)
#layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="after fg fc1",summarize=5)
#
###layer_fc1=tf.map_fn(fa,layer_fc1,name="fc1_activated")
##layer_fc1=batch_norm_fc(layer_fc1,1,phase_train)
##layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="after batchnorm fc1",summarize=5)
#
#
#layer_fc1=tf.nn.relu(layer_fc1,name="fc1_act")
##layer_fc1=cabs(layer_fc1)
#layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="after cabs fc1",summarize=5)
#
#layer_fc1=fa(layer_fc1)
#layer_fc1=tf.Print(layer_fc1,[layer_fc1],message="after act fc1",summarize=5)
#
#
##layer_fc1=tf.py_func(fw,[layer_fc1],tf.float32)
##layer_fc1=tf.py_func(fg,[layer_fc1],tf.float32)    
##layer_fc1=tf.py_func(cabs,[layer_fc1],tf.float32)   
#
##print("Passed activation of fc1 ")


#Output FC layer
layer_fc2 = new_fc_layer(x_in=layer_flat,name="fco",
                         num_inputs=num_features,
                         num_outputs=num_classes,
                         bitw=bitw,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

#CREATE SESSION
session = tf.Session()
session.run(tf.global_variables_initializer())





# Counter for total number of iterations performed so far.


optimize(num_iterations=int(iters)) # We already performed 1 iteration above.

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)


def plot_conv_weights(weights, input_channel=0):
    global BITW,BITA,BITG
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights, feed_dict={bitw:BITW,bita:BITA,bitg:BITG,phase_train:False})

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
#print("connv 1 weights")
#print(weights_conv1)
#print(session.run(weights_conv1) )   
#
print("conv2 weights")
#print(weights_conv1)
#print(session.run(weights_conv2, feed_dict={bitw:32,bita:32,bitg:32}))




#print("output tensor of conv2 is")
#print(session.run(layer_conv2))
#plot_conv_weights(weights=weights_conv2)
#print("plotted one")

plot_conv_weights(weights=weights_conv2)

s