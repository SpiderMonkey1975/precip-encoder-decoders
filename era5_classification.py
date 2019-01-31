import numpy as np
import tensorflow as tf
import sys, argparse

from datetime import datetime

from networks import simple
from networks.simple import classifier

from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

print(" ")
print(" ")
print("*===================================================================================================*")
print("*                                         RAINFALL CLASSIFIER                                       *")
print("*===================================================================================================*")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=50, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=250, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.0001, help="set intial learning rate for optimizer")
parser.add_argument('-v', '--variable', type=str, default='rh', help="set variable to be used for training. Valid values are z, t, rh")
parser.add_argument('-n', '--num_nodes', type=int, default=200, help="number of hidden nodes in last layer of classifier")
parser.add_argument('-y', '--bins', type=int, default=4, help="number of bins")
parser.add_argument('-x', '--levels', type=int, default=800, help="atmospheric pressure level used for input data. Valid values are 500, 800, 1000")
args = parser.parse_args()

print(" ")
print(" ")
print("       Model Settings:")
print("         * using ERA5-Australia specific %s input at %d hPa level over %d bins" % (args.variable,args.levels,args.bins))
print("         * model will run for %2d epochs" % (args.epochs))
print("         * a batch size of %2d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))

##
## Construct the neural network 
##

image_width = 240
image_height = 360
input_layer = layers.Input(shape = (image_width, image_height, 1))

print(" ")
print("       Network Settings:")
print("         * using 3-layer classifier with %d, %d and %d numbers of hidden nodes" % (8*args.num_nodes,4*args.num_nodes,args.num_nodes))

net = classifier( input_layer, args.num_nodes, args.bins )

with tf.device("/cpu:0"):
     model = models.Model(inputs=input_layer, outputs=net)
model = multi_gpu_model( model, gpus=4)

##
## Set the appropriate optimizer and loss function 
##

opt = Adam(lr=args.learn_rate)
model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## 
## Load the training input data from disk
##

inputfile = "input_data/training/" + str(args.levels) + "hPa/" + args.variable + "_era5_au_" + str(args.bins) + "bins.npy"
x_train = np.load( inputfile )
x_train = np.expand_dims( x_train, axis=3 )

inputfile = "input_data/training/au_labels_" + str(args.bins) + "bins.npy"
y_train = np.load( inputfile )

num_images = np.amin( [x_train.shape[0],y_train.shape[0]] )

x_train = x_train[ :num_images, :image_width, :image_height, : ]
y_train = y_train[ :num_images, : ]

##
## Define the callbacks to be used in the model training
##

def step_decay(epoch):
    if epoch<11:
       return args.learn_rate
    elif epoch >= 11 and epoch < 30:
       return args.learn_rate/2.0
    else:
       return args.learn_rate/5.0

lrate = LearningRateScheduler(step_decay)
earlystopper = EarlyStopping( monitor='val_acc', patience=25 )
#checkpointer = ModelCheckpoint(filepath='checkpoints/bestmodel_' + args.variable + "_" + str(args.bins) + "bins.hdf5", save_best_only=True)

##
## Train model.  Only output information for the validation steps only.
##

print(" ")
print(" ")
print("                                       Model Training Output")
print("*---------------------------------------------------------------------------------------------------*")

t1 = datetime.now()
history = model.fit( x=x_train, y=y_train, 
                     batch_size=args.batch_size*4, 
                     epochs=args.epochs, 
                     verbose=2,
                     shuffle=True, 
                     validation_split=0.25,
                     callbacks=[lrate,earlystopper] )
                   #  callbacks=[lrate,earlystopper,checkpointer] )
training_time = datetime.now() - t1
print(" ")
print("       Training time was", training_time)

## 
## Load the test input data from disk
##

inputfile = "input_data/test/" + str(args.levels) + "hPa/" + args.variable + "_era5_au_" + str(args.bins) + "bins.npy"
x_test = np.load( inputfile )
x_test = np.expand_dims( x_test, axis=3 )

inputfile = "input_data/test/au_labels_" + str(args.bins) + "bins.npy"
y_test = np.load( inputfile )

num_images = np.amin( [x_test.shape[0],y_test.shape[0]] )

x_test = x_test[ :num_images, :image_width, :image_height, : ]
y_test = y_test[ :num_images, : ]

##
## End by sending test data through the trained model
##

print(" ")
print(" ")
print("                                        Model Prediction Test")
print("*---------------------------------------------------------------------------------------------------*")

t1 = datetime.now()
score = model.evaluate( x=x_test, y=y_test, 
                        batch_size=args.batch_size*4,
                        verbose=0 )
prediction_time = datetime.now() - t1
print("       Test on %s samples, Accuracy of %4.3f"  % (num_images,score[1]))
print("       Prediction time was", prediction_time)
print(" ")

