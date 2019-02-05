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
parser.add_argument('-e', '--epochs', type=int, default=100, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=100, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.0001, help="set intial learning rate for optimizer")
parser.add_argument('-v', '--variable', type=str, default='rh', help="set variable to be used for training. Valid values are z, t, rh")
parser.add_argument('-n', '--num_nodes', type=int, default=50, help="number of hidden nodes in last layer of classifier")
parser.add_argument('-y', '--bins', type=int, default=6, help="number of bins")
parser.add_argument('-x', '--levels', type=int, default=800, help="atmospheric pressure level used for input data. Valid values are 500, 800, 1000")
parser.add_argument('-r', '--reg_constant', type=float, default=0.05, help="set L2 regularization constant")
parser.add_argument('-d', '--dropout', type=float, default=0.3, help="set dropout fraction")
parser.add_argument('-z', '--layers', type=int, default=3, help="set nnumber of layers in classifier")
args = parser.parse_args()

print(" ")
print(" ")
print("       Model Settings:")
print("         * using ERA5-Australia specific %s input at %d hPa level over %d bins" % (args.variable,args.levels,args.bins))
print("         * model will run for %2d epochs" % (args.epochs))
print("         * a batch size of %2d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))
print("         * dropout ratio is %f" % (args.dropout))
print("         * L2 regularization enabled with constant of %f used" % (args.reg_constant))
print("         * using 3-layer classifier with %d, %d and %d numbers of hidden nodes" % (8*args.num_nodes,4*args.num_nodes,args.num_nodes))

##
## Set some important parameters
##

image_width = 240
image_height = 360
num_gpus = 1

##
## Construct the neural network 
##

input_layer = layers.Input(shape = (image_width, image_height, 1))
net = classifier( input_layer, args.num_nodes, args.bins, args.dropout, args.reg_constant, args.layers )

if num_gpus>1:
   with tf.device("/cpu:0"):
        model = models.Model(inputs=input_layer, outputs=net)
   model = multi_gpu_model( model, gpus=num_gpus)
else:
   model = models.Model(inputs=input_layer, outputs=net)

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
    if epoch<=25:
       return args.learn_rate
    elif epoch>25 and epoch<=50:
       return args.learn_rate / 10.0
    else:
       return args.learn_rate/100.0

lrate = LearningRateScheduler( step_decay )
earlystopper = EarlyStopping( monitor='val_acc', 
                              patience=10 )
checkpointer = ModelCheckpoint( filepath='checkpoints/bestmodel_' + args.variable + "_" + str(args.bins) + "bins.hdf5", 
                                save_best_only=True,
                                period=5 )

##
## Train model.  Only output information for the validation steps only.
##

print(" ")
print(" ")
print("                                       Model Training Output")
print("*---------------------------------------------------------------------------------------------------*")

t1 = datetime.now()
history = model.fit( x=x_train, y=y_train, 
                     batch_size=args.batch_size*num_gpus, 
                     epochs=args.epochs, 
                     verbose=2,
                     shuffle=False, 
                     validation_split=0.25, 
                     callbacks=[lrate] )
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
                        batch_size=args.batch_size*num_gpus,
                        verbose=0 )
prediction_time = datetime.now() - t1
print("       Test on %s samples, Accuracy of %4.3f"  % (num_images,score[1]))
print( score )
print("       Prediction time was", prediction_time)
print(" ")

