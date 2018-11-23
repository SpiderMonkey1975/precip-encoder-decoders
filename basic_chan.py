import argparse
import itertools
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers, datasets, iterators, training, backends
from chainer.training import extensions
from chainer.backends import cuda

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
print("       Starting chainer run on %1d GPUS" % (args.num_gpus))
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

class encoder_decoder(Chain):
    def __init__(self):

        # make encoder_decoder a child of the Chain class
        super(encoder_decoder, self).__init__()

        # define all the individuals components in the networks (CNNs, BatchNorms, etc)
        with self.init_scope():
             self.bn = L.BatchNormalization(axis=3)
             self.conv1 = L.Convolution2D( in_channels=args.channels, out_channels=64, ksize=5, stride=2 )
             self.conv2 = L.Convolution2D( in_channels=64, out_channels=128, ksize=3, stride=2 )
             self.conv3 = L.Convolution2D( in_channels=128, out_channels=256, ksize=3, stride=2 )
             self.deconv1 = L.Deconvolution2D( in_channels=256, out_channels=128, ksize=3, stride=2 )
             self.deconv2 = L.Deconvolution2D( in_channels=128, out_channels=64, ksize=3, stride=2 )
             self.deconv3 = L.Deconvolution2D( in_channels=64, out_channels=1, ksize=5, stride=2 )

    # define the forward propagation process along with activation functions
    def forward(self, x):
        h1 = F.relu( self.conv1(self.bn(x)) )
        h2 = F.relu( self.conv2(self.bn(h1)) )
        h3 = F.relu( self.conv3(self.bn(h2)) )
        h4 = F.relu( self.deconv1(self.bn(h3)) )
        h5 = F.relu( self.deconv2(self.bn(h4)) )
        return F.relu( self.deconv3(self.bn(h5)) )

model = L.Classifier( encoder_decoder(), lossfun=F.mean_absolute_error )

opt = optimizers.Adam( alpha=args.learn_rate )
opt.setup( model )

##
## Create separate data-parallel instances of the neural net on each GPU
##

for n in range(args.num_gpus):
    chainer.backends.cuda.get_device_from_id(n).use()
    model.to_gpu()

workspace = int(1 * 2**30)
chainer.cuda.set_max_workspace_size(workspace)
chainer.config.use_cudnn = 'always'


## 
## Read in the raw input from hard disk
##

x = np.load("/scratch/pawsey0001/mcheeseman/weather_data/10zlevels.npy")
y = 1000*np.expand_dims(np.load("/scratch/pawsey0001/mcheeseman/weather_data/full_tp_1980_2016.npy"), axis=3)

dataset = chainer.datasets.TupleDataset( x, y )


##
## Divide the input data into training, verification and testing sets
##

n = num_training_images + num_verification_images
n2 = n + num_testing_images + 1

y_train = y[:num_training_images, :]
y_verify = y[num_training_images+1:n+1:, :]
y_test = y[n:n2, :]

##
## Iterate through all channels looking for the optimial correlation 
##

min_score = 1.0
channels = range(2)
if args.channels==1:
   for i in channels:
     if i<1:
       x_train = x[:num_training_images, :, :, i:i+1]
       x_verify = x[num_training_images+1:n+1, :, :, i:i+1]
       x_test = x[n:n2, :, :, i:i+1]

       # convert to Chainer dataset objects
       train_dataset = datasets.TupleDataset( x_train, y_train )
       valid_dataset = datasets.TupleDataset( x_verify, y_verify )

       # construct iterator objects for automated training
       train_iter = iterators.SerialIterator( train_dataset, args.batch_size )
       valid_iter = iterators.SerialIterator( valid_dataset, args.batch_size, repeat=False, shuffle=False )

       updater = training.updaters.StandardUpdater( train_iter, opt, device=0 )
       trainer = training.Trainer(updater, (args.epochs, 'epoch'), out='channel_corr_results')
       trainer.extend( extensions.Evaluator(valid_iter, model, device=0) )
       trainer.extend( extensions.PrintReport(['epoch', 'validation/main/loss', 'elapsed_time']) )
       trainer.extend( extensions.ProgressBar() )
       trainer.run()

#       valid_updater = training.updaters.StandardUpdater( valid_iter, opt, device=0 )
#       valid_trainer = training.Trainer(valid_updater, (args.epochs, 'epoch'), out='channel_corr_results')
#       valid_trainer.extend( extensions.PrintReport(['epoch', 'validation/mean_absolute_error', 'elapsed_time']) )
#       valid_trainer.run()

#       if score[1]<min_score:
#          min_score = score[1]
#          corr_str = "Channels %1d - %1d" % (i,i+1)

#elif args.channels==2:
   
##
## Iterate through all channel combinations in the input datasets looking for correlation
##

#   for i,j in itertools.combinations(channels,2):
#       x_train = x[:num_training_images, :, :, [i,j]]
#       x_verify = x[num_training_images+1:n+1, :, :, [i,j]]
#       x_test = x[n:n2, :, :, [i,j]]

#       history = model.fit( x_train, y_train, 
#                            batch_size=args.batch_size*args.num_gpus, 
#                            epochs=args.epochs, 
#                            verbose=0, 
#                            validation_data=(x_verify, y_verify) )
#       score = model.evaluate( x=x_test, y=y_test, 
#                               batch_size=args.batch_size*args.num_gpus, 
#                               verbose=0 )
#       print("       channel correlation ", i, j, "   evaluation MSE: ", score[1])
#       if score[1]<min_score:
#          min_score = score[1]
#          corr_str = "Channels %1d - %1d" % (i,j)

#elif args.channels==3:

##
## Iterate through all channel combinations in the input datasets looking for correlation
##

#   for i,j,k in itertools.combinations(channels,3):
#       x_train = x[:num_training_images, :, :, [i,j,k]]
#       x_verify = x[num_training_images+1:n+1, :, :, [i,j,k]]
#       x_test = x[n:n2, :, :, [i,j]]

#       history = model.fit( x_train, y_train, 
#                            batch_size=args.batch_size*args.num_gpus, 
#                            epochs=args.epochs, 
#                            verbose=0, 
#                            validation_data=(x_verify, y_verify) )
#       score = model.evaluate( x=x_test, y=y_test, 
#                               batch_size=args.bat#ch_size*args.num_gpus, 
#                               verbose=0 )
#       print("       channel correlation ", i, j, k, "   evaluation MSE: ", score[1])
#       if score[1]<min_score:
#          min_score = score[1]
#          corr_str = "Channels %1d - %1d - %1d" % (i,j,k)

#print(" ")
#print("      minimum MSE of %f achieved with %s" % (min_score,corr_str))
#print(" ")
