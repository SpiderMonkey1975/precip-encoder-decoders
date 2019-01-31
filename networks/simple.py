import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Dense, Flatten, UpSampling2D

##
## encoder_decoder 
##-----------------------------------------------------------------------------------------------------
##
## Python function that constructs a simple autoencoder neural network for image classification.
##
## INPUT: input_layer -> 3D tensor containing the feature input data for the neural network to process
##
##        max_filters -> maximum number of filters to be used in the encoding section 
##
##        max_hidden_nodes -> maximum number of hidden nodes in a single layer in the perceptron section
##
## OUTPUT: net -> fully formed neural network
##

def encoder_decoder( input_layer, num_filters, num_nodes ):

#    net = BatchNormalization(axis=3)( input_layer )
    net = input_layer
    for n in range(3):
        net = Conv2D( np.int32(num_filters), 5, activation='relu', padding='same' )(net)
        net = MaxPooling2D( 2 )(net)
        num_filters *= 2

    # construct the decoding part of the neural network using transpose CNNs 

#    net = BatchNormalization(axis=3)(net)
    for n in range(2):
        net = Conv2DTranspose( np.int32(n), 5, padding='same', activation='relu' )(net)
        num_filters *= 0.5 

    # add a multi-layer perceptron section for the image classification implementation
   
    net = Flatten()(net)

    net = Dense( num_nodes, activation='relu' )(net)
    net = Dense( num_nodes/2, activation='relu' )(net)
    net = Dense( 4, activation='softmax' )(net)
    return net

def encoder_classifier( input_layer, num_filters ):

    net = input_layer
   
    for i in range(5): 
        net = Conv2D( num_filters, 5, activation='relu', padding='same' )(net)
        net = MaxPooling2D( 2 )(net)

    for i in range(5): 
        net = UpSampling2D( 2 )(net)
        net = Conv2D( num_filters, 5, activation='relu', padding='same' )(net)

    net = Flatten()(net)

    net = Dense( 1024, activation='relu' )(net)
    net = Dense( 64, activation='relu' )(net)
    net = Dense( 4, activation='softmax' )(net)
    return net

def classifier( input_layer, num_nodes, num_bins ):
    
    dropout_ratio = 0.4

    net = Flatten()(input_layer)
    net = Dense( 8*num_nodes, activation='relu' )(net)
    net = Dropout(dropout_ratio)(net)
    net = Dense( 4*num_nodes, activation='relu' )(net)
    net = Dropout(dropout_ratio)(net)
    net = Dense( 2*num_nodes, activation='relu' )(net)
    net = Dropout(dropout_ratio)(net)
    net = Dense( num_nodes, activation='relu' )(net)
    return Dense( num_bins, activation='softmax' )(net)

