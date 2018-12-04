import numpy as np
import tensorflow as tf
import argparse

from datetime import datetime

from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

print(" ")
print(" ")
print("*===========================================================================================================*")
print("*                               RAINFALL AUTO-ENCODER / DECODER PREDICTOR                                   *")
print("*===========================================================================================================*")

##
## Set the number of images used for training, verification and testing
##

num_training_images = 20000
num_verification_images = 10000
num_testing_images = 640 

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_gpus', type=int, default=1, help="number of GPUs to be used")
parser.add_argument('-e', '--epochs', type=int, default=1, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=512, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set intial learning rate for optimizer")
parser.add_argument('-r', '--l2_reg', type=float, default=0.00002, help="set L2 regularization parameter")
parser.add_argument('-m', '--min_change', type=float, default=1.0, help="minimum change in loss that ends run")
args = parser.parse_args()

print(" ")
print(" ")
print("       Starting %s run on %1d GPUS using %s precision" % (K.backend(),args.num_gpus,K.floatx()))
print(" ")
print("       Model Settings:")
print("         * model will run for a maximum of %2d epochs" % (args.epochs))
print("         * a batch size of %3d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))
if args.l2_reg > 0.0001:
   print("         * L2 regularization is enabled with L2 =", args.l2_reg)

##
## Contruct the neural network
##

if args.l2_reg > 0.0001:
   conv_kernel_reg = l1_l2(l1=0.0, l2=args.l2_reg)
else:
   conv_kernel_reg = None

# input data
inputs = layers.Input(shape = (240, 360, 2))

# encoding section
bn0 = BatchNormalization(axis=3)(inputs)
conv1 = layers.Conv2D(64, (5, 5), strides=(2,2), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn0)
bn1 = BatchNormalization(axis=3)(conv1)
conv2 = layers.Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn1)
bn2 = BatchNormalization(axis=3)(conv2)
conv3 = layers.Conv2D(256, (3, 3), strides=(2,2), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn2)
bn3 = BatchNormalization(axis=3)(conv3)
conv4 = layers.Conv2D(256, (5, 5), strides=(3,3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn3)

# decoder section
bn4 = BatchNormalization(axis=3)(conv4)
dconv1 = layers.Conv2DTranspose( 128, (5,5), strides=(3,3), padding='same', activation='relu', kernel_regularizer=conv_kernel_reg )(bn4)
bn5 = BatchNormalization(axis=3)(dconv1)
dconv2 = layers.Conv2DTranspose( 128, (3,3), strides=(2,2), padding='same', activation='relu', kernel_regularizer=conv_kernel_reg )(bn5)
bn6 = BatchNormalization(axis=3)(dconv2)
dconv3 = layers.Conv2DTranspose( 64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_regularizer=conv_kernel_reg )(bn6)
bn7 = BatchNormalization(axis=3)(dconv3)
dconv4 = layers.Conv2DTranspose( 1, (5,5), strides=(2,2), padding='same', activation='relu', kernel_regularizer=conv_kernel_reg )(bn7)

##
## Create separate data-parallel instances of the neural net on each GPU
##

if ( args.num_gpus <= 1 ):
   model = models.Model(inputs=inputs, outputs=dconv4)
else:
   with tf.device("/cpu:0"):
        model = models.Model(inputs=inputs, outputs=dconv4)
   model = multi_gpu_model( model, gpus=args.num_gpus )

##
## Set the optimizer to be used and compile model
##

opt = Adam(lr=args.learn_rate)
model.compile(loss='mae', optimizer=opt, metrics=['mae'])

## 
## Load the raw input data from disk
##

x = np.load("datasets/z_era5_native_NWHC.npy")
y = 1008*np.load("datasets/tp_era5_native.npy")[:, :, :, None]

##
## Divide the input data into training, verification and testing sets
##

n = num_training_images + num_verification_images
n2 = num_training_images + num_verification_images + num_testing_images

x_train = x[1:num_training_images+1, 0:240, 0:360, [1,2]]
y_train = y[1:num_training_images+1, 0:240, 0:360]

x_verify = x[num_training_images+1:n+1, 0:240, 0:360, [1,2]]
y_verify = y[num_training_images+1:n+1, 0:240, 0:360]

x_test = x[n:n2, 0:240, 0:360, [1,2]]
y_test = y[n:n2, 0:240, 0:360]

print(" ")
print("       Input Description:")
print("         * images dimensions are %3d by %d" % (x_train.shape[1],x_train.shape[2]))
print("         * %5d training images used" % (x_train.shape[0]))
print("         * %5d validation images used" % (x_verify.shape[0]))
print("         * %4d testing images used" % (x_test.shape[0]))
print(" ")

##
## Set a callback to monitor the observed validation mean square error. End
## the training early if it falls below an user-supplied value.
##

if args.min_change>0.0
   earlyStop = EarlyStopping( monitor='loss',
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
print("                                           Model Training Output")
print("*-----------------------------------------------------------------------------------------------------------*")

t1 = datetime.now()
history = model.fit( x_train, y_train, 
                     batch_size=args.batch_size*args.num_gpus, 
                     epochs=args.epochs, 
                     verbose=2, 
                     validation_data=(x_verify, y_verify),
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
print("       Test on %s samples, MAE of %4.3f"  % (num_testing_images,score[1]))
print("       Prediction time was", prediction_time)
print(" ")

