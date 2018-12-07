import argparse
import numpy as np
import mxnet as mx

from datetime import datetime

from mxnet import gluon, autograd
from mxnet.gluon import nn, data

##------------------------------------------------------------------------------
## ERA5 Specific Configuration Details
##------------------------------------------------------------------------------

features_input_datafile = "datasets/z_era5_native_NWHC.npy"
labels_input_datafile = "datasets/tp_era5_native.npy"
num_input_images = 30000
num_test_images = 640 
image_width = 240
image_height = 360
channels = [1,2]

print(" ")
print(" ")
print("*===========================================================================================================*")
print("*                               RAINFALL AUTO-ENCODER / DECODER PREDICTOR                                   *")
print("*===========================================================================================================*")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_gpus', type=int, default=1, help="number of GPUs to be used")
parser.add_argument('-e', '--epochs', type=int, default=1, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=64, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set intial learning rate for optimizer")
parser.add_argument('-f', '--train_fraction', type=float, default=0.7, help="fraction of input images used for training")
args = parser.parse_args()

print(" ")
print(" ")
print("       Starting MxNet run on %1d GPUS" % (args.num_gpus))
print(" ")
print("       Model Settings:")
print("         * model will run for %2d epochs" % (args.epochs))
print("         * a batch size of %3d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))

##
## Construct the neural network 
##

net = nn.Sequential()
with net.name_scope():
     # Encodong section
     net.add( nn.BatchNorm( axis=3, in_channels=2 ) )
     net.add( nn.Conv2D( channels=64, kernel_size=5, strides=2, layout='NHWC', activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=2 ) )
     net.add( nn.Conv2D( channels=128, kernel_size=3, strides=2, layout='NHWC', activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=2 ) )
     net.add( nn.Conv2D( channels=256, kernel_size=3, strides=2, layout='NHWC', activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=2 ) )
     net.add( nn.Conv2D( channels=256, kernel_size=5, strides=3, layout='NHWC', activation='relu' ) )

     # Decoding section

## mpch - padding is WAYYY off
     net.add( nn.BatchNorm( axis=3, in_channels=2 ) )
     net.add( nn.Conv2DTranspose( channels=128, kernel_size=3, layout='NHWC', strides=3, in_channels=256, activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=2 ) )
     net.add( nn.Conv2DTranspose( channels=128, kernel_size=3, strides=2, layout='NHWC', in_channels=128, activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=2 ) )
     net.add( nn.Conv2DTranspose( channels=64, kernel_size=3, strides=2, layout='NHWC', padding=1, in_channels=128, activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=2 ) )
     net.add( nn.Conv2DTranspose( channels=1, kernel_size=5, strides=2, output_padding=1, output_padding=2, layout='NHWC', in_channels=64, activation='relu' ) )

##
## Define the loss function to be used
##

loss_fn = gluon.loss.L1Loss()

##
## Setup the neural network on the GPU and initialise the weight values
##

net.initialize( mx.init.Xavier(), ctx=mx.gpu(0) )
#ctx = [mx.gpu(i) for i in range(args.num_gpus)]
#net.collect_params().initialize(ctx=ctx)

##
## Define the loss function to be used in training and the optimizer
##

opt = gluon.Trainer( net.collect_params(), 'adam', {'learning_rate': args.learn_rate})

## 
## Load the raw input data from disk
##

x = np.load( features_input_datafile )
y = 6000*np.load( labels_input_datafile )[:, :, :, None]

##
## Form the training, validation and test datasets
##

num_training_images = int( num_input_images*args.train_fraction )
x_train = x[1:num_training_images+1, 0:image_width, 0:image_height, channels]
y_train = y[1:num_training_images+1, 0:image_width, 0:image_height]

dataset_train = data.ArrayDataset( x_train, y_train )
train_iter = data.DataLoader( dataset_train, 
                              batch_size=args.batch_size*args.num_gpus, 
                              last_batch='rollover',
                              shuffle=True )

n = num_input_images + 1
x_verify = x[num_training_images+1:n, 0:image_width, 0:image_height, channels]
y_verify = y[num_training_images+1:n, 0:image_width, 0:image_height]

dataset_valid = data.ArrayDataset( x_verify, y_verify )
valid_iter = data.DataLoader( dataset_valid, 
                              batch_size=args.batch_size*args.num_gpus, 
                              last_batch='rollover' )

n = num_input_images + num_test_images + 1
x_test = x[num_input_images+1:n, 0:image_width, 0:image_height, channels]
y_test = y[num_input_images+1:n, 0:image_width, 0:image_height]

dataset_test = data.ArrayDataset( x_test, y_test )
test_iter = data.DataLoader( dataset_test, 
                             batch_size=args.batch_size*args.num_gpus, 
                             last_batch='discard' )

##
## Train model.  Only output information for the validation steps only.
##

#print(" ")
#print(" ")
#print("                                           Model Training Output")
#print("*-----------------------------------------------------------------------------------------------------------*")

#t1 = datetime.now()

for epoch in range(args.epochs):

    # training data
#    cumulative_loss = mx.nd.zeros(1, mx.gpu(0))
#    num_samples = 0
    for batch_idx, (data, label) in enumerate(train_iter):
        data = data.as_in_context(mx.gpu(0))
        label = label.as_in_context(mx.gpu(0))
        with autograd.record():
            output = net(data)
        print( output.shape )
        print( label.shape )
        break
    break
#            loss = loss_fn(output, label)
#        loss.backward()
#        opt.step(data.shape[0])
#        cumulative_loss += loss.sum()
#        num_samples += data.shape[0]
#    train_loss = cumulative_loss.asscalar()/num_samples

    # validation data
#    cumulative_loss = mx.nd.zeros(1, mx.gpu(0))
#    num_samples = 0
#    for batch_idx, (data, label) in enumerate(valid_iter):
#        data = data.as_in_context(mx.gpu(0))
#        label = label.as_in_context(mx.gpu(0))
#        output = net(data)
#        loss = loss_fn(output, label)
#        cumulative_loss += loss.sum()
#        num_samples += data.shape[0]
#    valid_loss = cumulative_loss.asscalar()/num_samples

#    print("Epoch {}, training loss: {:.2f}, validation loss: {:.2f}".format(epoch, train_loss, valid_loss))

#training_time = datetime.now() - t1
#print(" ")
#print("       Training time was", training_time)

##
## End by sending test data through the trained model
##

#t1 = datetime.now()

#cumulative_loss = mx.nd.zeros(1, mx.gpu(0))
#num_samples = 0
#for batch_idx, (data, label) in enumerate(test_iter):
#        data = data.as_in_context(mx.gpu(0))
#        label = label.as_in_context(mx.gpu(0))
#        output = net(data)
#        loss = loss_fn(output, label)
#        cumulative_loss += loss.sum()
#        num_samples += data.shape[0]
#    test_loss = cumulative_loss.asscalar()/num_samples
#
#prediction_time = datetime.now() - t1
#print("       Test on %s samples, MAE of %4.3f"  % (num_test_images,test_loss))
#print("       Prediction time was", prediction_time)
#print(" ")
