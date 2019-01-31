import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout
from tensorflow.keras.layers import Dense, Flatten, UpSampling2D, concatenate, Dropout

def add_perceptron( net, num_nodes ):
    net = Flatten()(net)
    net = Dense( 4*num_nodes, activation='relu' )(net)
    net = Dense( num_nodes, activation='relu' )(net)
    return Dense( 4, activation='softmax' )(net)

##
## *layer_unet 
##-----------------------------------------------------------------------------------------------------
##
## Python function that constructs an autoencoder neural network for image classification lossely based
## on the U-Net design.
##
## INPUT: input_layer -> 3D tensor containing the feature input data for the neural network to process
##
## OUTPUT: net -> fully formed neural network
##

def shallow_unet( input_layer, num_filters, num_hidden_nodes ):

    # construct the contracting path
    cnv1 = Conv2D( num_filters, 3, activation='relu', padding='same' )(input_layer)
    net = MaxPooling2D( 2 )(cnv1)

    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)

    net = Conv2D( 4, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net, num_hidden_nodes )

def deep_unet( input_layer, num_filters, num_hidden_nodes ):

    # construct the contracting path

    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(input_layer)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    cnv1 = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)

    net = Conv2D( 4, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net, num_hidden_nodes )

def unet_1_layer( input_layer, num_filters, num_hidden_nodes ):

    # construct the contracting path

    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(input_layer)
    cnv1 = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)
    net = BatchNormalization(axis=3)( net )

    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = BatchNormalization(axis=3)( net )

    net = Conv2D( 4, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net, num_hidden_nodes )

def unet_2_layer( input_layer, num_filters, num_hidden_nodes ):

    # construct the contracting path 

    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(input_layer)
    cnv1 = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    cnv2 = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv2)

    net = Conv2D( 4*num_filters, 3, activation='relu', padding='same' )(net)

    # construct the expansive path  

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv2], axis=3 )
    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = BatchNormalization(axis=3)( net )

    net = Conv2D( 4, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net, num_hidden_nodes )
   
def unet_3_layer( input_layer, num_filters, num_hidden_nodes ):

    dropout_fraction = 0.4

    # construct the contracting path

    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(input_layer)
    net = Dropout(dropout_fraction)(net)
    cnv1 = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = Dropout(dropout_fraction)(net)
    cnv2 = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv2)

    net = Conv2D( 4*num_filters, 3, activation='relu', padding='same' )(net)
    net = Dropout(dropout_fraction)(net)
    cnv3 = Conv2D( 4*num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv3)

    net = Conv2D( 8*num_filters, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv3], axis=3 )
    net = Conv2D( 4*num_filters, 3, activation='relu', padding='same' )(net)
    net = Dropout(dropout_fraction)(net)
    net = Conv2D( 4*num_filters, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv2], axis=3 )
    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = Dropout(dropout_fraction)(net)
    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Dropout(dropout_fraction)(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)

    net = Conv2D( 4, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net, num_hidden_nodes )

