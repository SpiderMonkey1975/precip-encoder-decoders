import numpy as np
import tensorflow as tf
import sys, argparse, neural_networks

from datetime import datetime

from neural_networks import simple_classifier, unet, vgg_classifier, encoder_decoder 

from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

print(" ")
print(" ")
print("*===========================================================================================================*")
print("*                                             RAINFALL CLASSIFIER                                           *")
print("*===========================================================================================================*")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_gpus', type=int, default=4, help="number of GPUs to be used")
parser.add_argument('-e', '--epochs', type=int, default=25, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.0001, help="set intial learning rate for optimizer")
parser.add_argument('-v', '--variable', type=str, default='z', help="set variable to be used for training. Valid values are z, t, rh")
parser.add_argument('-r', '--l2_reg', type=float, default=0.00001, help="set L2 regularization parameter")
parser.add_argument('-d', '--data', type=str, default='native', help="dataset type: native, au")
args = parser.parse_args()

print(" ")
print(" ")
print("       Starting %s run on %1d GPUS using %s precision" % (K.backend(),args.num_gpus,K.floatx()))
print(" ")
print("       Model Settings:")
print("         * using ERA5-%s %s input" % (args.data,args.variable))
print("         * model will run for %2d epochs" % (args.epochs))
print("         * a batch size of %2d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))

##
## Set some important dimension data
##

image_width = 241
image_height = 361

##
## Contruct the neural network 
##

model = simple_classifier( args.l2_reg, image_width, image_height, args.num_gpus )

##
## Set the appropriate optimizer and loss function 
##

opt = Adam(lr=args.learn_rate)
model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## 
## Load the training input data from disk
##

inputfile = "input_data/training/" + args.variable + "_era5_" + args.data + "_CLASSIFICATION.npy"
x_train = np.load( inputfile )

inputfile = "input_data/training/" + args.data + "_one_hot_encoding.npy"
y_train = np.load( inputfile )

num_images = np.amin( [x_train.shape[0],y_train.shape[0]] )

x_train = x_train[ :num_images, :, :, : ]
y_train = y_train[ :num_images, : ]

##
## Train model.  Only output information for the validation steps only.
##

print(" ")
print(" ")
print("                                           Model Training Output")
print("*-----------------------------------------------------------------------------------------------------------*")

t1 = datetime.now()
history = model.fit( x=x_train, y=y_train, 
                     batch_size=args.batch_size*args.num_gpus, 
                     epochs=args.epochs, 
                     verbose=2,
                     shuffle=True, 
                     validation_split=0.16 )
training_time = datetime.now() - t1
print(" ")
print("       Training time was", training_time)
sys.exit()

## 
## Load the test input data from disk
##

inputfile = "input_data/" + args.variable + "_era5_" + args.data + "_NWHC.npy"
x_test = np.load( inputfile )

inputfile = "input_data/era5_labels_" + args.data + ".npy"
y_test = np.load( inputfile )

##
## End by sending test data through the trained model
##

print(" ")
print(" ")
print("                                            Model Prediction Test")
print("*-----------------------------------------------------------------------------------------------------------*")

t1 = datetime.now()
score = model.evaluate( x=x_test, y=y_test, 
                        batch_size=args.batch_size*args.num_gpus,
                        verbose=0 )
prediction_time = datetime.now() - t1
print("       Test on %s samples, Accuracy of %4.3f"  % (num_test_images,score[1]))
print("       Prediction time was", prediction_time)
print(" ")

