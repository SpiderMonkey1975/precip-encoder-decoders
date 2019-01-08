import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, applications
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import concatenate, Dropout, UpSampling2D, Dense, Flatten
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
## classifier
##
## Python function that constructs a simple image classifier neural network
##

def simple_classifier( l2_reg, image_width, image_height, num_gpus ):
    
    # check if L2 regularization has been requested by the user
    if l2_reg > 0.0001:
       conv_kernel_reg = l1_l2(l1=0.0, l2=l2_reg)
    else:
       conv_kernel_reg = None

    # set the input layer
    inputs = layers.Input(shape = (image_width, image_height, 3))

    # construct the CNN section
    net = BatchNormalization(axis=3)(inputs)
    net = Conv2D( 16, 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    net = Conv2D( 32, 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    net = MaxPooling2D( 2 )(net)
    net = BatchNormalization(axis=3)(net)
    net = Conv2D( 64, 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    net = Conv2D( 128, 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    net = MaxPooling2D( 2 )(net)
    net = BatchNormalization(axis=3)(net)
    net = Dropout(0.1)(net)
    net = Conv2D( 256, 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    net = Conv2D( 512, 5, activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    net = MaxPooling2D( 2 )(net)

    net = BatchNormalization(axis=3)(net)
    net = Conv2DTranspose( 256, 5, padding='same', activation='relu', kernel_regularizer=conv_kernel_reg)(net)
    net = BatchNormalization(axis=3)(net)
    net = Conv2DTranspose( 128, 5, padding='same', activation='relu', kernel_regularizer=conv_kernel_reg)(net)
    net = BatchNormalization(axis=3)(net)
    net = Conv2DTranspose( 64, 5, padding='same', activation='relu', kernel_regularizer=conv_kernel_reg)(net)
    net = BatchNormalization(axis=3)(net)
    net = Conv2DTranspose( 32, 5, padding='same', activation='relu', kernel_regularizer=conv_kernel_reg)(net)

    # add multi-layer perceptron section
    net = Flatten()(net)
    net = Dense( 256, activation='relu' )(net)
    net = Dropout(0.25)(net)
    net = Dense( 128, activation='relu' )(net)
    net = Dropout(0.25)(net)
    net = Dense( 4, activation='softmax' )(net)

    return GetModel( num_gpus, inputs, net )


def vgg_classifier( image_width, image_height, num_gpus ):
    vgg_model = applications.VGG16( weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3) )

    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    net = layer_dict['block2_pool'].output

    net = Flatten()(net)
    net = Dense( 256, activation='relu' )(net)
    net = Dropout(0.3)(net)
    net = Dense( 64, activation='relu' )(net)
    net = Dropout(0.3)(net)
    net = Dense( 4, activation='softmax' )(net)

    model = GetModel( num_gpus, vgg_model.input, net )
    for layer in model.layers[:12]:
        layer.trainable = False

    return model

##
## encoder_decoder
##
## Python function that defines a basic encoder-decoder neural network designs
##

def encoder( filters, kernel_sizes, strides, conv_kernel_reg, inputs ):
    net = inputs
#    net = BatchNormalization(axis=3)(net)
    for n in range(len(filters)):
        net = BatchNormalization(axis=3)(net)
        net = Conv2D(filters[n], kernel_sizes[n], strides[n], activation='sigmoid', padding='same', kernel_regularizer=conv_kernel_reg)(net)
    return net

def decoder( filters, kernel_sizes, strides, conv_kernel_reg, inputs ):
    net = inputs
#    net = BatchNormalization(axis=3)(net)
    for n in range(len(filters)):
        net = BatchNormalization(axis=3)(net)
        net = Conv2DTranspose(filters[n], kernel_sizes[n], strides[n], padding='same', activation='relu', kernel_regularizer=conv_kernel_reg)(net)
    return net

def encoder_decoder( l2_reg, image_width, image_height, num_gpus ):

    # check if L2 regularization has been requested by the user
    if l2_reg > 0.0001:
       conv_kernel_reg = l1_l2(l1=0.0, l2=l2_reg)
    else:
       conv_kernel_reg = None

    # set the input layer
    inputs = layers.Input(shape = (image_width, image_height, 3))

    # construct the encoding section
    filters = [64,128,256,512]
    kernel_sizes = [5,3,3,5]
    strides = [2,2,2,3]
    net = encoder( filters, kernel_sizes, strides, conv_kernel_reg, inputs )

    # construct the decoder section
    filters = [256,128,64,4]
    strides = [3,2,2,2]
    net = decoder( filters, kernel_sizes, strides, conv_kernel_reg, net )

    return GetModel( num_gpus, inputs, net )

##
## unet 
##
## Python function that creates a basic unet neural network 
##

def unet_encoding_layer( num_filters, inputs ):
    conv = Conv2D( num_filters, 4, activation='relu', padding='same' )( inputs )
    conv = Conv2D( num_filters, 4, activation='relu', padding='same' )( conv )
    pool = MaxPooling2D( pool_size=2 )( conv )
    return conv, pool

def unet_decoding_layer( num_filters, conv1, conv2 ):
    up = Conv2D( num_filters, 2, activation='relu', padding='same' )(UpSampling2D(size=2)(conv1))
    merge = concatenate( [conv2,up], axis=3 )
    conv = Conv2D( num_filters, 4, activation='relu', padding='same' )(merge)
    return Conv2D( num_filters, 4, activation='relu', padding='same' )(conv)

def unet( image_width, image_height, num_gpus ):

    # define input layer
    inputs = layers.Input(shape = (image_width, image_height, 3))
    
    # build the encoding section of the neural net
    conv1, pool = unet_encoding_layer( 8, inputs )
    conv2, pool = unet_encoding_layer( 16, pool )
    conv3, pool = unet_encoding_layer( 32, pool )
 
    # bottom section of the U-Net design
    net = Conv2D(64, 4, activation = 'relu', padding = 'same')(pool)
    net = Conv2D(64, 4, activation = 'relu', padding = 'same')(net)

    # build the decoding section of the neural net
#    net = unet_decoding_layer( 32, net, conv3 )
#    net = unet_decoding_layer( 16, net, conv2 )
#    net = unet_decoding_layer( 8, net, conv1 )
    
#    conv6 = unet_decoding_layer( 512, drop5, conv4 )
#    conv7 = unet_decoding_layer( 256, conv6, conv3 )
#    conv8 = unet_decoding_layer( 128, conv7, conv2 )
#    conv9 = unet_decoding_layer(  64, conv8, conv1 )
#    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    # finish by adding some dense layers for final classification
    net = Flatten()(net)
    net = Dense( 64, activation = 'relu' )(net)
    net = Dense(  2, activation = 'relu' )(net)

#    drop5 = Dropout(0.4)(conv5)
#    net = Flatten()(conv10)
#    net = Dense( 64, activation = 'relu' )(net)
#    net = Dense( 2, activation = 'relu' )(net)

    return GetModel( num_gpus, inputs, net )
   
