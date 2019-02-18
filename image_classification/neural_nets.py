
from tensorflow.keras.layers import Dropout, Dense, Flatten

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

def classifier( input_layer, num_nodes, num_bins, dropout_ratio, num_layers ):

    net = Flatten()(input_layer)

    factor = 8
    for n in range(num_layers-1):
        net = Dense( factor*num_nodes, activation='relu' )(net)
        net = Dropout(dropout_ratio)(net)
        factor = factor / 2
        if factor<1:
           factor = 1
    return Dense( num_bins, activation='softmax' )(net)
