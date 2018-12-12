import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Cropping2D
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.regularizers import l1_l2

def GetModel( num_gpus, inputs, net ):
    # check if a multi-gpu model needs to be created
    if ( num_gpus <= 1 ):
       model = models.Model(inputs=inputs, outputs=net)
    else:
       with tf.device("/cpu:0"):
            model = models.Model(inputs=inputs, outputs=net)
       model = multi_gpu_model( model, gpus=num_gpus )
    return model

##
## encoder_decoder
##
## Python function that defines a basic encoder-decoder neural network designs
##

def encoder( filters, kernel_sizes, strides, conv_kernel_reg, inputs ):
    net = inputs
    for n in range(len(filters)):
        net = BatchNormalization(axis=3)(net)
        net = Conv2D(filters[n], kernel_sizes[n], strides[n], activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    return net

def decoder( filters, kernel_sizes, strides, conv_kernel_reg, inputs ):
    net = inputs
    for n in range(len(filters)):
        net = BatchNormalization(axis=3)(net)
        net = Conv2DTranspose(filters[n], kernel_sizes[n], strides[n], padding='same', activation='relu', kernel_regularizer=conv_kernel_reg)(net)
    return net

def encoder_decoder( l2_reg, image_width, image_height, num_channels, num_gpus ):

    # check if L2 regularization has been requested by the user
    if l2_reg > 0.0001:
       conv_kernel_reg = l1_l2(l1=0.0, l2=l2_reg)
    else:
       conv_kernel_reg = None

    # set the input layer
    inputs = layers.Input(shape = (image_width, image_height, num_channels))

    # construct the encoding section
    filters = [64,128,256,256]
    kernel_sizes = [5,3,3,5]
    strides = [2,2,2,3]
    net = encoder( filters, kernel_sizes, strides, conv_kernel_reg, inputs )

    # construct the decoder section
    filters = [128,128,64,1]
    strides = [3,2,2,2]
    net = decoder( filters, kernel_sizes, strides, conv_kernel_reg, net )

    return GetModel( num_gpus, inputs, net )


def encoder_decoder2( l2_reg, image_width, image_height, num_channels, num_gpus ):

    # check if L2 regularization has been requested by the user
    if l2_reg > 0.0001:
       conv_kernel_reg = l1_l2(l1=0.0, l2=l2_reg)
    else:
       conv_kernel_reg = None

    # set the input layer
    inputs = layers.Input(shape = (image_width, image_height, num_channels))

    # construct the encoding section
    filters = [16,32,64,128,256]
    kernel_sizes = [4,6,8,10,12]
    strides = [[2,1],2,[2,1],2,2]
    net = encoder( filters, kernel_sizes, strides, conv_kernel_reg, inputs )

    # construct the decoder section
    filters = [128,64,32,16,1]
    net = decoder( filters, kernel_sizes, strides, conv_kernel_reg, net )
    
    return GetModel( num_gpus, inputs, net )


##
## unet 
##
## Python function that creates a basic unet neural network 
##

#def RightBlock( conv, num_filter, kernel_size, inputs ):
#    net = Conv2DTranspose( num_filter, 2, 2, padding="same" )(inputs) 
#    print( "upsampled dimensions:", net.shape )
#    print( "conv dimensions:", conv.shape )
#    net = layers.concatenate([net, conv], axis=3)
#    bn = BatchNormalization(axis=3)(net)
#    conv = Conv2D(num_filter, 3, activation='relu', padding='same')(bn)
#    bn = BatchNormalization(axis=3)(conv)
#    return Conv2D(num_filter, 3, activation='relu', padding='same')(conv)
#    bn12 = BatchNormalization(axis=3)(conv6)

def unet( image_width, image_height, num_channels, num_gpus ):

    # set the input layer
    inputs = layers.Input(shape = (image_width, image_height, num_channels))

    # construct the encoding section
    net = BatchNormalization(axis=3)( inputs )
    net = Conv2D( 64, 2, padding='same', activation='relu' )( net )
    print("Conv_1 output dimensions: ", net.shape)
    conv = Conv2D( 64, 2, padding='same', activation='relu' )( net )
    print("Conv_2 output dimensions: ", conv.shape)
    net = MaxPooling2D(pool_size=2)( conv )
    print("MaxPool output dimensions: ", net.shape)

    net = BatchNormalization(axis=3)( net )
    net = Conv2D( 128, 2, activation='relu', padding='same' )( net )
    print("Conv_3 output dimensions: ", net.shape)
    net = Conv2D( 128, 2, activation='relu', padding='same' )( net )
    print("Conv_4 output dimensions: ", net.shape)

    net = BatchNormalization(axis=3)( net )
    dconv = Conv2DTranspose( 64, 2, strides=2 )( net )
    print("dConv_1 output dimensions: ", dconv.shape)
    #conv = Cropping2D( cropping=2 )( conv )
    net = layers.concatenate([conv, dconv], axis=3)
    net = Conv2D( 64, 2, activation='relu', padding='same' )( net )
    print("Conv_5 output dimensions: ", net.shape)
    net = Conv2D( 64, 2, activation='relu', padding='same' )( net )
    print("Conv_6 output dimensions: ", net.shape)
    net = Conv2D( 1, 1, activation='relu', padding='same' )( net )
    print("Conv_7 output dimensions: ", net.shape)

    return GetModel( num_gpus, inputs, net )

