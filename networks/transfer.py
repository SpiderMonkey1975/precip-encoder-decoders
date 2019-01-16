import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.keras import applications
from tensorflow.keras.applications import resnet50, vgg16 


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

def transfer_learning( model_choice, input_layer, num_hidden_nodes ):
    
    # setup the base model based on the ResNet-50 architecture

    if model_choice == 1:
       base_model = ResNet50( weights='imagenet', include_top=False, 
                              input_tensor=input_layer, input_shape=(240,360,3) )
       for layer in base_model.layers[:25]:
           layer.trainable = False
    else: 
       base_model = applications.VGG19( weights='imagenet', include_top=False, 
                           input_tensor=input_layer, input_shape=(240,360,3) )
       for layer in base_model.layers[:5]:
           layer.trainable = False
    net = base_model.output

    # add our own customized classifer layer(s)

    net = Flatten()(net)
    net = Dense( 2*num_hidden_nodes, activation='relu' )(net)
    net = Dropout(0.2)(net)
    net = Dense( num_hidden_nodes, activation='relu' )(net)
    return Dense( 4, activation='softmax' )(net)

