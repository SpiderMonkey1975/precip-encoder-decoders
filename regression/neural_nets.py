from tensorflow.keras import layers
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, UpSampling2D, concatenate

##
## unet_1_layer 
##
## Python function that constructs an autoencoder neural network for image classification lossely based
## on the U-Net design.
##
## INPUT: input_layer -> 3D tensor containing the feature input data for the neural network to process
##        num_filters -> # of filters used in convolutions of the top layer of the neural net
##
## OUTPUT: net -> fully formed neural network
##

def unet_1_layer( input_layer, num_filters ):

    # construct the contracting path

    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(input_layer)
    cnv1 = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)

    return Conv2D( 1, 1 )(net)


def unet_2_layer( input_layer, num_filters ):

    # construct the contracting path

    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(input_layer)
    cnv1 = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv1)

    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    cnv2 = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = MaxPooling2D( 2 )(cnv2)

    # at bottom

    net = Conv2D( 4*num_filters, 3, activation='relu', padding='same' )(net)

    # construct the expansive path

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv2], axis=3 )
    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( 2*num_filters, 3, activation='relu', padding='same' )(net)

    net = UpSampling2D( 2 )(net)
    net = concatenate( [net,cnv1], axis=3 )
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)
    net = Conv2D( num_filters, 3, activation='relu', padding='same' )(net)

    return Conv2D( 1, 1 )(net)
