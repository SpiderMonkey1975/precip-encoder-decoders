import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout
from tensorflow.keras.layers import Dense, Flatten, UpSampling2D, concatenate

def add_perceptron( net ):
    net = Flatten()(net)
    net = Dense( 128, activation='relu' )(net)
    net = Dropout(0.2)(net)
    net = Dense( 32, activation='relu' )(net)
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

def 1layer_unet( input_layer ):

    # construct the contracting path

    net = BatchNormalization(axis=3)( input_layer )
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    cnv1 = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)

    net = Conv2D( 2, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net )
#    net = Flatten()(net)
#
#    net = Dense( 128, activation='relu' )(net)
#    net = Dropout(0.2)(net)
#    net = Dense( 32, activation='relu' )(net)
#    net = Dense( 4, activation='softmax' )(net)
#
#    return net

def 2layer_unet( input_layer ):

    # construct the contracting path 

    net = BatchNormalization(axis=3)( input_layer )
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    cnv1 = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = BatchNormalization(axis=3)( net )
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    cnv2 = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv2)

    net = Conv2D( 128, 3, activation='relu', padding='same' )(net)

    # construct the expansive path  

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv2], axis=3 )
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)

    net = Conv2D( 2, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net )
   
def 3layer_unet( input_layer ):

    # construct the contracting path

    net = BatchNormalization(axis=3)( input_layer )
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    cnv1 = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = BatchNormalization(axis=3)( net )
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    cnv2 = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv2)

    net = BatchNormalization(axis=3)( net )
    net = Conv2D( 128, 3, activation='relu', padding='same' )(net)
    cnv3 = Conv2D( 128, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv3)

    net = Conv2D( 256, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv3], axis=3 )
    net = Conv2D( 128, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 128, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv2], axis=3 )
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)

    net = Conv2D( 2, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net )

def full_unet( input_layer ):

    # construct the contracting path

    net = BatchNormalization(axis=3)( input_layer )
    net = Conv2D( 16, 3, activation='relu', padding='same' )(net)
    cnv1 = Conv2D( 16, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = BatchNormalization(axis=3)( net )
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    cnv2 = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv2)

    net = BatchNormalization(axis=3)( net )
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    cnv3 = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv3)

    net = BatchNormalization(axis=3)( net )
    net = Conv2D( 128, 3, activation='relu', padding='same' )(net)
    cnv4 = Conv2D( 128, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv4)

    net = Conv2D( 256, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv4], axis=3 )
    net = Conv2D( 128, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 128, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv3], axis=3 )
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 64, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv2], axis=3 )
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 32, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( 16, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 16, 3, activation='relu', padding='same' )(net)

    net = Conv2D( 2, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return add_perceptron( net )
