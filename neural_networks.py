import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.regularizers import l1_l2

##
## simple_classifier
##-----------------------------------------------------------------------------------------------------
##
## Python function that constructs a simple autoencoder neural network for image classification.
##
## INPUT: l2_reg -> value of the L2 norm coefficient for kernel regularization.  If value <= 0.0001,
##                  no kernel regularization will be employed.
##
##        image_width, image_height -> width and height of the input images given in pixels
##
##        num_gpus -> # of GPUs on which the neural network will be run.  If value > 1, 
##                    a data-parallel operation is performed (eg network instance copied to each GPU)
##
##        max_filters -> maximum number of filters to be used in the encoding section 
##
##        max_hidden_nodes -> maximum number of hidden nodes in a single layer in the perceptron section
##
## OUTPUT: model -> fully formed neural network mapped over 1 or more GPUs
##

def simple_classifier( l2_reg, image_width, image_height, num_gpus, max_filters, max_hidden_nodes ):
    
    # check if L2 regularization has been requested by the user
    if l2_reg > 0.0001:
       conv_kernel_reg = l1_l2(l1=0.0, l2=l2_reg)
    else:
       conv_kernel_reg = None

    # set the input layer based on the image dimensions (width and height) and 3 input channels
    # (each channel corresponds to an observed atmospheric pressure level)
    inputs = layers.Input(shape = (image_width, image_height, 3))

    # construct the encoding section of the neural network using CNNs and max pooling
    
    if max_filters > 256:
       max_filters = 256
    if max_filters < 32:
       max_filters = 32

    num_filters = max_filters 
    for n in range(3):
        net = BatchNormalization(axis=3)(inputs)
        net = Conv2D( np.int32(num_filters), 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
        net = MaxPooling2D( 2 )(net)

        num_filters = num_filters / 2 
        if num_filters < 16:
           break

    #net = BatchNormalization(axis=3)(inputs)
    #net = Conv2D( 32, 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    #net = MaxPooling2D( 2 )(net)

    #net = BatchNormalization(axis=3)(inputs)
    #net = Conv2D( 64, 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    #net = MaxPooling2D( 2 )(net)

    # construct the decoding part of the neural network using transpose CNNs 

    num_filters = max_filters / 2
    for n in range(3):
        net = BatchNormalization(axis=3)(inputs)
        net = Conv2DTranspose( np.int32(num_filters), 5, padding='same', activation='relu', kernel_regularizer=conv_kernel_reg)(net)

        num_filters = num_filters / 2
        if num_filters < 8:
           break

    # add a multi-layer perceptron section for the image classification implementation
    net = Flatten()(net)
   
    if max_hidden_nodes < 64:
       max_hidden_nodes = 64
    if max_hidden_nodes > 256:
       max_hidden_nodes = 256

    num_nodes = np.int32( max_hidden_nodes )
    for n in range(2):
        net = Dense( num_nodes, activation='relu' )(net)
        net = Dropout(0.1)(net)
        num_nodes = num_nodes / 2

    net = Dense( 4, activation='softmax' )(net)

    # check if a multi-gpu model needs to be created
    if ( num_gpus <= 1 ):
       model = models.Model(inputs=inputs, outputs=net)
    else:
       with tf.device("/cpu:0"):
            model = models.Model(inputs=inputs, outputs=net)
       model = multi_gpu_model( model, gpus=num_gpus )

    model.summary()
    return model

