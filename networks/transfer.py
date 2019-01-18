import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.keras import applications

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
    

    if model_choice == 0:

     # setup the base model based on the ResNet-50 architecture (168 layers)
       base_model = applications.ResNet50( weights='imagenet', include_top=False, 
                              input_tensor=input_layer, input_shape=(240,360,3) )
       num_constant_layers = 68

    elif model_choice == 1:

     # setup the base model based on the Xception V1 architecture (126 layers)
       base_model = applications.Xception( weights='imagenet', include_top=False, 
                              input_tensor=input_layer, input_shape=(240,360,3),
                              pooling='max' )
       num_constant_layers = 100

    elif model_choice == 2:

     # setup the base model based on the Inception V3 architecture (159 layers)
       base_model = applications.inception_v3( weights='imagenet', include_top=False, 
                              input_tensor=input_layer, input_shape=(240,360,3),
                              pooling='max' )
       num_constant_layers = 129

    else: 

     # setup the base model based on the VGG-19 architecture (26 layers)
       base_model = applications.VGG19( weights='imagenet', include_top=False, 
                           input_tensor=input_layer, input_shape=(240,360,3) )
       num_constant_layers = 13

   # for layer in base_model.layers[:num_constant_layers]:
   #     layer.trainable = False
   # for layer in base_model.layers[num_constant_layers:]:
    for layer in base_model.layers:
        layer.trainable = True
    net = base_model.output

    # add our own customized classifer layer(s)

    net = Flatten()(net)
    net = Dense( 2*num_hidden_nodes, activation='relu' )(net)
    net = Dropout(0.3)(net)
    net = Dense( num_hidden_nodes, activation='relu' )(net)
    return Dense( 4, activation='softmax' )(net)

