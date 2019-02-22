import numpy as np
import tensorflow as tf
import sys, argparse

from datetime import datetime

from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import neural_nets
from neural_nets import classifier, unet_1_layer


print(" ")
print(" ")
print("*===================================================================================================*")
print("*                                         RAINFALL CLASSIFIER                                       *")
print("*===================================================================================================*")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=250, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.0001, help="set intial learning rate for optimizer")
parser.add_argument('-v', '--variable', type=str, default='rh', help="set variable to be used for training. Valid values are z, t, rh")
parser.add_argument('-n', '--num_nodes', type=int, default=50, help="number of hidden nodes in last layer of classifier")
parser.add_argument('-y', '--bins', type=int, default=6, help="number of bins")
parser.add_argument('-d', '--dropout', type=float, default=0.3, help="set dropout fraction")
parser.add_argument('-z', '--layers', type=int, default=3, help="set number of layers in classifier")
args = parser.parse_args()

print(" ")
print(" ")
print("       Model Settings:")
print("         * using ERA5-Australia specific %s input over all 3 pressure levels over %d bins" % (args.variable,args.bins))
print("         * model will run for %2d epochs" % (args.epochs))
print("         * a batch size of %2d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))
print("         * dropout ratio is %f" % (args.dropout))
print("         * using 3-layer classifier with %d, %d and %d numbers of hidden nodes" % (8*args.num_nodes,4*args.num_nodes,args.num_nodes))

##
## Set some important parameters
##

image_width = 240
image_height = 360
num_channels = 3

##
## Construct the neural network 
##

input_layer = layers.Input(shape = (image_width, image_height, num_channels))
if args.layers>0:
   net = classifier( input_layer, args.num_nodes, args.bins, args.dropout, args.layers )
else:
   net = unet_1_layer( input_layer, int(32), args.bins )

model = models.Model(inputs=input_layer, outputs=net)

##
## Set the appropriate optimizer and loss function 
##

opt = Adam(lr=args.learn_rate)
model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## 
## Load the training input data from disk
##

inputfile = "../input_data/training/" + args.variable + "_era5_au_" + str(args.bins) + "bins.npy"
x_train = np.load( inputfile )

inputfile = "../input_data/training/au_labels_" + str(args.bins) + "bins.npy"
y_train = np.load( inputfile )

num_images = np.amin( [x_train.shape[0],y_train.shape[0]] )

x_train = x_train[ :num_images, :image_width, :image_height, : ]
y_train = y_train[ :num_images, : ]

##
## Train model.  Only output information for the validation steps only.
##

print(" ")
print(" ")
print("                                       Model Training Output")
print("*---------------------------------------------------------------------------------------------------*")

filename = "../model_backups/" + str(args.bins) + "bins/model_weights_" + args.variable + "_all_levels.h5"
checkpoint = ModelCheckpoint( filename, monitor='val_acc', verbose=0, save_best_only=True, mode='max' )

earlystop = EarlyStopping( min_delta=0.00001,
                           patience=10,
                           mode='min' )

my_callbacks = [checkpoint, earlystop]

t1 = datetime.now()
history = model.fit( x=x_train, y=y_train, 
                     batch_size=args.batch_size, 
                     epochs=args.epochs, 
                     verbose=2,
                     shuffle=False,
                     callbacks=my_callbacks, 
                     validation_split=0.16 )
training_time = datetime.now() - t1
print(" ")
print("       Training time was", training_time)

sys.exit(0)

##
## Save the model and weights to hard disk
##

#filename = "../model_backups/model_weights_" + args.variable + "_all_levels.h5"
#model.save_weights( "model_weights.h5" )

#filename = "../model_backups/model_architecture_" + args.variable + "_all_levels.json"
#with open( filename ) as f:
#     f.write( model.to.json() )

## 
## Load the test input data from disk
##

inputfile = "../input_data/test/" + args.variable + "_era5_au_" + str(args.bins) + "bins.npy"
x_test = np.load( inputfile )

inputfile = "../input_data/test/au_labels_" + str(args.bins) + "bins.npy"
y_test = np.load( inputfile )

num_images = np.amin( [x_test.shape[0],y_test.shape[0]] )

x_test = x_test[ :num_images, :image_width, :image_height, : ]
y_test = y_test[ :num_images, : ]

##
## End by sending test data through the trained model
##

print(" ")
print(" ")
print("                                        Model Evaluation")
print("*---------------------------------------------------------------------------------------------------*")

score = model.evaluate( x=x_test, y=y_test, 
                        batch_size=args.batch_size,
                        verbose=0 )
print("       Test on %s samples, Accuracy of %4.3f"  % (num_images,score[1]))

##
## Compare predictions for the first 10 test samples with known labels 
##

print(" ")
print(" ")
print("                                        Model Prediction Test")
print("*---------------------------------------------------------------------------------------------------*")

filename = "predictions_" + args.variable + "all_levels.npy"
predictions = np.argmax( model.predict( x_test ), axis=1 )
np.save( filename, predictions )

