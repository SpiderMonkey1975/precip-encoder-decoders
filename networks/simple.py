import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.regularizers import l1, l2

##
## classifier 
##-----------------------------------------------------------------------------------------------------
##
## Python function that constructs a simple  neural network for image classification.
##
## INPUT: input_layer -> 3D tensor containing the feature input data for the neural network to process
##
##        num_nodes -> number of hidden nodes in the last perceptron layer 
##
##        num_bins -> number of image classifications used in the output layer
##
##        dropout_ratio -> ratio of layer connections dropped to limit overfitting
##
## OUTPUT: net -> fully formed neural network
##

def classifier( input_layer, num_nodes, num_bins, dropout_ratio, reg_constant, num_layers ):
    
    net = Flatten()(input_layer)

    factor = 8
    for n in range(num_layers-1):
        net = Dense( factor*num_nodes, activation='relu', kernel_regularizer=l2(reg_constant) )(net)
        net = Dropout(dropout_ratio)(net)
        factor = factor / 2
        if factor<1:
           factor = 1
#    net = Dense( 4*num_nodes, activation='relu', kernel_regularizer=l2(reg_constant) )(net)
 #   net = Dropout(dropout_ratio)(net)
#    net = Dense( num_nodes, activation='relu', kernel_regularizer=l2(reg_constant) )(net)
    return Dense( num_bins, activation='softmax' )(net)

