import argparse
import itertools
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, backend
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model


print(" ")
print(" ")
print("*===========================================================================================================*")
print("*                                       CHANNEL CORRELATION RUNS                                            *")
print("*===========================================================================================================*")

##
## Set the number of images used for training, verification and testing
##

num_training_images = 40000
num_verification_images = 13000
num_testing_images = 1000


##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_gpus', type=int, default=1, help="number of GPUs to be used")
parser.add_argument('-e', '--epochs', type=int, default=1, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=128, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set intial learning rate for optimizer")
parser.add_argument('-c', '--channels', type=int, default=1, help="set number of channels to test")
args = parser.parse_args()


print(" ")
print(" ")
print("       Starting %s run on %1d GPUS using %s precision" % (backend.backend(),args.num_gpus,backend.floatx()))
print(" ")
print("       Model Settings:")
print("         * %1d channel correlation run will be performed" % (args.channels))
print("         * model will run for %2d epochs" % (args.epochs))
print("         * a batch size of %3d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))
print(" ")


##
## Contruct the neural network
##

inputs = layers.Input(shape = (80, 120, args.channels))

bn0 = BatchNormalization(axis=3)(inputs)
conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding='same')(bn0)
bn1 = BatchNormalization(axis=3)(conv1)
conv2 = layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(bn1)
bn2 = BatchNormalization(axis=3)(conv2)
conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(bn2)
bn3 = BatchNormalization(axis=3)(conv3)

conv4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(bn3)
bn4 = BatchNormalization(axis=3)(conv4)
conv5 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(bn4)
bn5 = BatchNormalization(axis=3)(conv5)
conv6 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), activation='relu', padding='same')(bn5)


##
## Create separate data-parallel instances of the neural net on each GPU
##

if ( args.num_gpus <= 1 ):
   model = models.Model(inputs=inputs, outputs=conv6)
else:
   with tf.device("/cpu:0"):
        model = models.Model(inputs=inputs, outputs=conv6)
   model = multi_gpu_model( model, gpus=args.num_gpus )

model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001), metrics=['mae'])


## 
## Load the input data sets
##

x = np.load("datasets/10zlevels.npy")
y = 1000*np.expand_dims(np.load("datasets/full_tp_1980_2016.npy"), axis=3)


##
## Divide the input dataset into training, verification and testing sets
##

n = num_training_images + num_verification_images
n2 = n + num_testing_images + 1

y_train = y[:num_training_images, :]
y_verify = y[num_training_images+1:n+1:, :]
y_test = y[n:n2, :]

min_score = 1.0
channels = range(6)
if args.channels==1:

##
## Iterate through all 10 channels in the input datasets looking for correlation 
##

   for i in channels:
     if i<9:
       x_train = x[:num_training_images, :, :, i:i+1]
       x_verify = x[num_training_images+1:n+1, :, :, i:i+1]
       x_test = x[n:n2, :, :, i:i+1]

       history = model.fit( x_train, y_train, 
                            batch_size=args.batch_size*args.num_gpus, 
                            epochs=args.epochs, 
                            verbose=0, 
                            validation_data=(x_verify, y_verify) )
       score = model.evaluate( x=x_test, y=y_test, 
                               batch_size=args.batch_size*args.num_gpus, 
                               verbose=0 )
       print("       channel correlation ", i, i+1, "   evaluation MSE: ", score[1])
       if score[1]<min_score:
          min_score = score[1]
          corr_str = "Channels %1d - %1d" % (i,i+1)

#
elif args.channels==2:
   
##
## Iterate through all channel combinations in the input datasets looking for correlation
##

   for i,j in itertools.combinations(channels,2):
       x_train = x[:num_training_images, :, :, [i,j]]
       x_verify = x[num_training_images+1:n+1, :, :, [i,j]]
       x_test = x[n:n2, :, :, [i,j]]

       history = model.fit( x_train, y_train, 
                            batch_size=args.batch_size*args.num_gpus, 
                            epochs=args.epochs, 
                            verbose=0, 
                            validation_data=(x_verify, y_verify) )
       score = model.evaluate( x=x_test, y=y_test, 
                               batch_size=args.batch_size*args.num_gpus, 
                               verbose=0 )
       print("       channel correlation ", i, j, "   evaluation MSE: ", score[1])
       if score[1]<min_score:
          min_score = score[1]
          corr_str = "Channels %1d - %1d" % (i,j)

elif args.channels==3:

##
## Iterate through all channel combinations in the input datasets looking for correlation
##

   for i,j,k in itertools.combinations(channels,3):
       x_train = x[:num_training_images, :, :, [i,j,k]]
       x_verify = x[num_training_images+1:n+1, :, :, [i,j,k]]
       x_test = x[n:n2, :, :, [i,j]]

       history = model.fit( x_train, y_train, 
                            batch_size=args.batch_size*args.num_gpus, 
                            epochs=args.epochs, 
                            verbose=0, 
                            validation_data=(x_verify, y_verify) )
       score = model.evaluate( x=x_test, y=y_test, 
                               batch_size=args.batch_size*args.num_gpus, 
                               verbose=0 )
       print("       channel correlation ", i, j, k, "   evaluation MSE: ", score[1])
       if score[1]<min_score:
          min_score = score[1]
          corr_str = "Channels %1d - %1d - %1d" % (i,j,k)

print(" ")
print("      minimum MSE of %f achieved with %s" % (min_score,corr_str))
print(" ")
