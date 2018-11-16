import numpy as np
import tensorflow as tf
import argparse

from datetime import datetime

from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras import backend as K

print(" ")
print(" ")
print("*===========================================================================================================*")
print("*                               RAINFALL AUTO-ENCODER / DECODER PREDICTOR                                   *")
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
parser.add_argument('-b', '--batch_size', type=int, default=512, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set intial learning rate for optimizer")
parser.add_argument('-r', '--l2_reg', type=float, default=0.00022, help="set L2 regularization parameter")
parser.add_argument('-m', '--min_change', type=float, default=0.01, help="minimum change in MSE that ends run")
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

concat_axis = 3
inputs = layers.Input(shape = (80, 120, 3))

if args.l2_reg > 0.0001:
   conv_kernel_reg = l1_l2(l1=0.0, l2=args.l2_reg)
else:
   conv_kernel_reg = None

bn0 = BatchNormalization(axis=3)(inputs)
conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn0)
bn1 = BatchNormalization(axis=3)(conv1)
conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn1)
bn2 = BatchNormalization(axis=3)(conv1)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)

conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(pool1)
bn3 = BatchNormalization(axis=3)(conv2)
conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn3)
bn4 = BatchNormalization(axis=3)(conv2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(pool2)
bn5 = BatchNormalization(axis=3)(conv3)
conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn5)
bn6 = BatchNormalization(axis=3)(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6)

conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(pool3)
bn7 = BatchNormalization(axis=3)(conv4)
conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn7)
bn8 = BatchNormalization(axis=3)(conv4)
pool4 = layers.MaxPooling2D(pool_size=(2, 3))(bn8)

conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(pool4)
bn9 = BatchNormalization(axis=3)(conv5)
conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn9)
bn10 = BatchNormalization(axis=3)(conv5)

up_conv5 = layers.UpSampling2D(size=(2, 3))(bn10)
up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)
conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(up6)
bn11 = BatchNormalization(axis=3)(conv6)
conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn11)
bn12 = BatchNormalization(axis=3)(conv6)

up_conv6 = layers.UpSampling2D(size=(2, 2))(bn12)
up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(up7)
bn13 = BatchNormalization(axis=3)(conv7)
conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn13)
bn14 = BatchNormalization(axis=3)(conv7)

up_conv7 = layers.UpSampling2D(size=(2, 2))(bn14)
up8 = layers.concatenate([up_conv7, conv2], axis=concat_axis)
conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(up8)
bn15 = BatchNormalization(axis=3)(conv8)
conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn15)
bn16 = BatchNormalization(axis=3)(conv8)

up_conv8 = layers.UpSampling2D(size=(2, 2))(bn16)
up9 = layers.concatenate([up_conv8, conv1], axis=concat_axis)
conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(up9)
bn17 = BatchNormalization(axis=3)(conv9)
conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=conv_kernel_reg)(bn17)
bn18 = BatchNormalization(axis=3)(conv9)

conv10 = layers.Conv2D(1, (1, 1), kernel_regularizer=conv_kernel_reg)(bn18)


##
## Create separate data-parallel instances of the neural net on each GPU
##

if ( args.num_gpus <= 1 ):
   model = models.Model(inputs=inputs, outputs=conv10)
else:
   with tf.device("/cpu:0"):
        model = models.Model(inputs=inputs, outputs=conv10)
   model = multi_gpu_model( model, gpus=args.num_gpus )


##
## Set the optimizer to be used and compile model
##

opt = Adam(lr=args.learn_rate)
model.compile(loss='mae', optimizer=opt, metrics=['mse'])


##
## Define a subroutine that defines the learning rate for the optimizer.  We linearly
## scale the learning rate by number of epochs performed.
##
## Provided by Alessandro Rigazzi, Cray
##

def set_learning_rate( epoch ):
  total_epochs = args.epochs
  initial_lr = args.learn_rate
  if epoch<args.warmup_epochs:
    if epoch < 1:
      epoch = 1
    return initial_lr/args.warmup_epochs * epoch
  else:
    epochs_to_end = total_epochs-epoch
    if epochs_to_end < 1:
      epochs_to_end = 1
    return initial_lr/(total_epochs-args.warmup_epochs)*epochs_to_end


## 
## Load the input data set and then randomly shuffle the order of the input images
##

x = np.load("datasets/10zlevels.npy")
y = 1000*np.expand_dims(np.load("datasets/full_tp_1980_2016.npy"), axis=3)

#print( "total # of input images: ", x.shape[0] )
idxs = np.arange(x.shape[0])
np.random.seed(0)
np.random.shuffle(idxs)

x = x[idxs, :, :, :]
y = y[idxs, :]


##
## Divide the input dataset into training, verification  and testing sets
##

x_train = x[:num_training_images, :, :, [0,2,6]]
y_train = y[:num_training_images, :]

n = num_training_images + num_verification_images
x_verify = x[num_training_images+1:n+1, :, :, [0,2,6]]
y_verify = y[num_training_images+1:n+1, :]

n2 = n + num_testing_images + 1
x_test = x[n:n2, :, :, [0,2,6]]
y_test = y[n:n2, :]


##
## Set all (if any) callbacks we want implemented during training and validation
##

lrate = LearningRateScheduler(set_learning_rate)

earlyStop = EarlyStopping( monitor='val_mean_squared_error',
                           min_delta=args.min_change,
                           patience=4,
                           mode='min' )

my_callbacks = [earlyStop]


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
print("       Test on %s samples, MSE of %4.3f"  % (num_testing_images,score[1]))
print("       Prediction time was", prediction_time)
print(" ")
