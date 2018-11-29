import argparse
import numpy as np
import mxnet as mx

from mxnet import gluon, autograd, ndarray 
from mxnet.gluon import nn, data

print(" ")
print(" ")
print("*===========================================================================================*")
print("*                               CHANNEL CORRELATION RUNS                                    *")
print("*===========================================================================================*")

##
## Set the number of images used for training, verification and testing
##

num_training_images = 2048 #40000
num_verification_images = 256 #13000
#num_testing_images = 1000

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_gpus', type=int, default=1, help="set number of GPUs to be used")
parser.add_argument('-e', '--epochs', type=int, default=1, help="set number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=128, help="set batch size per GPU")
parser.add_argument('-l', '--lrate', type=float, default=0.001, help="set learn rate for optimizer")
args = parser.parse_args()

print(" ")
print(" ")
print("       Starting MxNet run on %1d GPUS using float32 precision" % (args.num_gpus))
print(" ")
print("       Model Settings:")
print("         * model will run for %2d epochs" % (args.epochs))
print("         * a batch size of %3d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.lrate))
print(" ")

## 
## Load the raw input data from hard disk
##

x = np.load("datasets/10zlevels.npy")
y = 1000*np.expand_dims(np.load("datasets/full_tp_1980_2016.npy"), axis=3)

##
## Form the training and validation datasets
##

x_train = x[:num_training_images, :, :, [0,2,6]]
y_train = y[:num_training_images, :]

dataset_train = data.ArrayDataset( x_train, y_train )
train_iter = data.DataLoader( dataset_train, 
                              batch_size=args.batch_size*args.num_gpus, 
                              num_workers=8,
                              last_batch='rollover',
                              shuffle=True )

n = num_training_images + num_verification_images
y_verify = y[num_training_images+1:n+1:, :]
x_verify = x[num_training_images+1:n+1, :, :, [0,2,6]]

dataset_valid = data.ArrayDataset( x_verify, y_verify )
valid_iter = data.DataLoader( dataset_valid, 
                              batch_size=args.batch_size*args.num_gpus, 
                              num_workers=8,
                              last_batch='rollover' )

##
## Define the neural network design
##

net = nn.Sequential()
with net.name_scope():
     net.add( nn.BatchNorm( axis=3, in_channels=3 ) )
     net.add( nn.Conv2D( channels=64, kernel_size=5, strides=2, layout='NHWC', padding=1, activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=3 ) )
     net.add( nn.Conv2D( channels=128, kernel_size=3, strides=2, layout='NHWC', activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=3 ) )
     net.add( nn.Conv2D( channels=256, kernel_size=3, strides=2, layout='NHWC', activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=3 ) )
     net.add( nn.Conv2DTranspose( channels=128, kernel_size=3, layout='NHWC', strides=2, in_channels=256, activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=3 ) )
     net.add( nn.Conv2DTranspose( channels=64, kernel_size=3, strides=2, layout='NHWC', in_channels=128, activation='relu' ) )
     net.add( nn.BatchNorm( axis=3, in_channels=3 ) )
     net.add( nn.Conv2DTranspose( channels=1, kernel_size=5, strides=2, padding=1, output_padding=1, layout='NHWC', in_channels=64, activation='relu' ) )

##
## Define the loss function to be used
##

loss_fn = gluon.loss.L1Loss()

##
## Setup the neural network on the GPU and initialise the weight values
##

if args.num_gpus==1:
   net.initialize( mx.init.Xavier(), ctx=mx.gpu(0) )
else:
   ctx = [mx.gpu(i) for i in range(args.num_gpus)]
   net.collect_params().initialize(ctx=ctx)

##
## Define the loss function to be used in training and the optimizer
##

opt = gluon.Trainer( net.collect_params(), 'adam', {'learning_rate': args.lrate})

##
## Perform the training 
##

for epoch in range(args.epochs):

    # training data
    cumulative_train_loss = mx.nd.zeros(1, mx.gpu(0))
    training_samples = 0
    for batch_idx, (data, label) in enumerate(train_iter):
        data = data.as_in_context(mx.gpu(0))
        label = label.as_in_context(mx.gpu(0))
        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        loss.backward()
        opt.step(data.shape[0])
        cumulative_train_loss += loss.sum()
        training_samples += data.shape[0]
    train_loss = cumulative_train_loss.asscalar()/training_samples

    # validation data
    cumulative_valid_loss = mx.nd.zeros(1, mx.gpu(0))
    valid_samples = 0
    for batch_idx, (data, label) in enumerate(valid_iter):
        data = data.as_in_context(mx.gpu(0))
        label = label.as_in_context(mx.gpu(0))
        output = net(data)
        loss = loss_fn(output, label)
        cumulative_valid_loss += loss.sum()
        valid_samples += data.shape[0]
    valid_loss = cumulative_valid_loss.asscalar()/valid_samples

    print("Epoch {}, training loss: {:.2f}, validation loss: {:.2f}".format(epoch, train_loss, valid_loss))
