import numpy as np
import tensorflow as tf
import argparse, neural_networks

from datetime import datetime

from neural_networks import encoder_decoder, unet

from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

print(" ")
print(" ")
print("*===========================================================================================================*")
print("*                               RAINFALL AUTO-ENCODER / DECODER PREDICTOR                                   *")
print("*===========================================================================================================*")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_gpus', type=int, default=4, help="number of GPUs to be used")
parser.add_argument('-e', '--epochs', type=int, default=5, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=64, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set intial learning rate for optimizer")
parser.add_argument('-r', '--l2_reg', type=float, default=0.00002, help="set L2 regularization parameter")
parser.add_argument('-m', '--min_change', type=float, default=0.0001, help="minimum change in validation MAE to continue run")
parser.add_argument('-f', '--train_fraction', type=float, default=0.7, help="fraction of input images used for training")
parser.add_argument('-d', '--dataset', type=str, default='native', help="datset type: native, au")
parser.add_argument('-n', '--network', type=str, default='encoder', help="set neural network design. Valid values are: unet, encoder")
args = parser.parse_args()

##
## Set the path to the raw datafiles and some important dimension data
##

features_input_datafile = "datasets/z_era5_" + args.dataset + "_NWHC.npy"
labels_input_datafile = "datasets/tp_era5_" + args.dataset + ".npy"
num_input_images = 30000
num_test_images = 640 
image_width = 240
image_height = 360
channels = [1,2]
label_factor = 6000


print(" ")
print(" ")
print("       Starting %s run on %1d GPUS using %s precision" % (K.backend(),args.num_gpus,K.floatx()))
print(" ")
print("       Model Settings:")
print("         * employing %s neural network design" % (args.network))
print("         * using ERA5-%s input" % (args.dataset))
print("         * model will run for a maximum of %2d epochs" % (args.epochs))
print("         * a batch size of %2d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))
if args.l2_reg > 0.0001:
   print("         * L2 regularization is enabled with L2 =", args.l2_reg)

##
## Contruct the neural network
##

if args.network == 'unet':
   model = unet( image_width, image_height, len(channels), args.num_gpus )
else:
   model = encoder_decoder( args.l2_reg, image_width, image_height, len(channels), args.num_gpus )


##
## Set the optimizer to be used and compile model
##

opt = Adam(lr=args.learn_rate)
model.compile(loss='mae', optimizer=opt, metrics=['mae'])

## 
## Load the raw input data from disk
##

x = np.load( features_input_datafile )
y = np.load( labels_input_datafile )[:, :, :, None]

##
## Divide the input data into input and test sets
##

n = num_input_images + num_test_images + 1
x_test = x[num_input_images:n, 0:image_width, 0:image_height, channels]
y_test = label_factor*y[num_input_images:n, 0:image_width, 0:image_height]

x = x[1:num_input_images+1, 0:image_width, 0:image_height, channels]
y = label_factor*y[1:num_input_images+1, 0:image_width, 0:image_height]

##
## Set a callback to monitor the observed validation mean square error. End
## the training early if it falls below an user-supplied value.
##

if args.min_change>0.0:
   earlyStop = EarlyStopping( monitor='val_mean_absolute_error',
                              min_delta=args.min_change,
                              patience=4,
                              mode='min' )
   my_callbacks = [earlyStop]
else:
   my_callbacks = []

##
## Train model.  Only output information for the validation steps only.
##

print(" ")
print(" ")

t1 = datetime.now()
history = model.fit( x, y, 
                     batch_size=args.batch_size*args.num_gpus, 
                     epochs=args.epochs, 
                     verbose=2,
                     shuffle=True, 
                     validation_split=1.0-args.train_fraction,
                     callbacks=my_callbacks )
training_time = datetime.now() - t1
print(" ")
print("       Training time was", training_time)

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
print("       Test on %s samples, MAE of %4.3f"  % (num_test_images,score[1]))
print("       Prediction time was", prediction_time)
print(" ")

