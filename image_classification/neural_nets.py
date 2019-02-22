
from tensorflow.keras.layers import Dropout, Dense, Flatten, BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate

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

def unet_1_layer( input_layer, num_filters, num_classes, num_nodes ):

    # construct the contracting path

    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(input_layer)
    cnv1 = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)
#    net = BatchNormalization(axis=3)( net )

    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
#    net = BatchNormalization(axis=3)( net )

    net = Conv2D( 4, 1, activation='relu', padding='same' )(net)

    # add a multi-layer perceptron section for the image classification implementation
    return classifier( net, num_nodes, num_classes, 0.3, 3 )
