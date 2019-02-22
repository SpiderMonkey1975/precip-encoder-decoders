import numpy as np
import tensorflow as tf
import sys, argparse

from datetime import datetime

from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model

from neural_nets import unet_1_layer, unet_2_layer


print(" ")
print(" ")
print("*===================================================================================================*")
print("*                                         RAINFALL CLASSIFIER                                       *")
print("*===================================================================================================*")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=250, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=250, help="set batch size per GPU")
parser.add_argument('-l', '--levels', type=str, default="800", help="atmospheric pressure level used for input data. Valid values are 500, 800, 1000 and all")
args = parser.parse_args()

print(" ")
print(" ")
print("       Model Settings:")
print("         * using ERA5-Australia specific input at the %s hPa level" % (args.levels))
print("         * model will run for a maximum of %2d epochs" % (args.epochs))
print("         * a batch size of %2d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of 0.0001 will be used")
print("         * dropout ratio is 0.3")

##
## Set dimensions of input data 
##

image_width = 240
image_height = 360
num_channels = 3 

##
## Contruct the neural network
##

num_cnn_filters = 16

input_layer = layers.Input(shape = (image_width, image_height, num_channels))
#net = unet_1_layer( input_layer, int(num_cnn_filters) )
net = unet_2_layer( input_layer, int(num_cnn_filters) )

with tf.device("/cpu:0"):
     model = models.Model( inputs=input_layer, outputs=net )
model = multi_gpu_model( model, gpus=4 )

##
## Set the optimizer to be used and compile model
##

opt = Adam( lr=0.0001 )
model.compile( loss='mae', optimizer=opt, metrics=['mae'] )

## 
## Load the training input data from disk
##

filename = "../input_data/training/" + args.levels + "hPa/era5_all_vars_au.npy"
x_train = np.load( filename )

filename = "../input_data/training/au_labels.npy"
y_train = np.expand_dims( np.load(filename), axis=3 )
y_train = y_train[ :, :image_width, :image_height ]


##
## Define two callbacks to be applied during the model training
##

filename = "../model_backups/model_weights_3vars_" + str(args.levels) + "hPa_regression.h5"
checkpoint = ModelCheckpoint( filename, 
                              monitor='val_loss', 
                              save_best_only=True, 
                              mode='min' )

earlystop = EarlyStopping( min_delta=0.01,
                           patience=10,
                           mode='min' )

my_callbacks = [checkpoint, earlystop]

##
## Train model.  Only output information for the validation steps only.
##

print(" ")
print(" ")
print("                                       Model Training Output")
print("*---------------------------------------------------------------------------------------------------*")

t1 = datetime.now()
history = model.fit( x=x_train, y=y_train, 
                     batch_size=4*args.batch_size, 
                     epochs=args.epochs, 
                     verbose=2,
                     shuffle=False, 
                     callbacks=my_callbacks,
                     validation_split=0.16 )
training_time = datetime.now() - t1
print(" ")
print("       Training time was", training_time)
