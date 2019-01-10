import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Dense, Flatten

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

def encoder_decoder( input_layer ):

    # construct the encoding section of the neural network using CNNs and max pooling
    num_filters = [128,64 ]
 
    net = BatchNormalization(axis=3)( input_layer )
    for n in num_filters:
        net = Conv2D( np.int32(n), 5, activation='relu', padding='same' )(net)
        net = MaxPooling2D( 2 )(net)

    # construct the decoding part of the neural network using transpose CNNs 

    num_filters = [64,32] 
    net = BatchNormalization(axis=3)(net)
    for n in num_filters:
        net = Conv2DTranspose( np.int32(n), 5, padding='same', activation='relu' )(net)

    # add a multi-layer perceptron section for the image classification implementation
   
    net = Flatten()(net)

    num_nodes = np.int32( max_hidden_nodes )
    net = Dense( 64, activation='relu' )(net)
    net = Dropout(0.2)(net)
    net = Dense( 32, activation='relu' )(net)
    net = Dense( 4, activation='softmax' )(net)
    return net

