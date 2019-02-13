import argparse, itertools, chainer
import numpy as np

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
parser.add_argument('-e', '--epochs', type=int, default=1, help="maximum number of epochs")
parser.add_argument('-b', '--batch_size', type=int, default=128, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set intial learning rate for optimizer")
args = parser.parse_args()


print(" ")
print(" ")
print("       Model Settings:")
print("         * model will run for %2d epochs" % (args.epochs))
print("         * a batch size of %3d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (args.learn_rate))
print(" ")


##
## Contruct the neural network
##

class simple_classifier(Chain):
    def __init__(self):

        # make encoder_decoder a child of the Chain class
        super(simple_classifier, self).__init__()

        # define all the individuals components in the networks (CNNs, BatchNorms, etc)
        with self.init_scope(): 
             self.fc1 = L.Linear( None, 8*50 )
             self.fc2 = L.Linear( None, 4*50 )
             self.fc3 = L.Linear( None, 50 )
             self.fc4 = L.Linear( None, 6 )

    # define the forward propagation process along with activation functions
    def forward(self, x):
        h1 = F.relu( self.fc1(x) )
        h2 = F.relu( self.fc2(h1) )
        h3 = F.relu( self.fc3(h2) )
        return self.fc4(h3) 

model = L.Classifier( simple_classifier() )

opt = optimizers.Adam( alpha=args.learn_rate )
opt.setup( model )

##
## Create separate data-parallel instances of the neural net on each GPU
##

for n in range(1):
    chainer.backends.cuda.get_device_from_id(n).use()
    model.to_gpu()

workspace = int(1 * 2**30)
chainer.cuda.set_max_workspace_size(workspace)
chainer.config.use_cudnn = 'always'


## 
## Read in the raw input from hard disk
##

x = np.load("input_data/training/800hPa/rh_era5_au_6bins.npy")
y = np.load("input_data/training/au_labels_6bins.npy")

dataset = chainer.datasets.TupleDataset( x, y )


##
## Divide the input data into training, verification and testing sets
##

x_train = x[:20000,:,:]
x_verify = x[20000:,:,:]

y_train = y[:20000, :]
y_verify = y[20000:, :]

##
## convert to Chainer dataset objects
##

train_dataset = datasets.TupleDataset( x_train, y_train )
valid_dataset = datasets.TupleDataset( x_verify, y_verify )

##
## construct iterator objects for automated training
##

train_iter = iterators.SerialIterator( train_dataset, args.batch_size )
valid_iter = iterators.SerialIterator( valid_dataset, args.batch_size, repeat=False, shuffle=False )

updater = training.updaters.StandardUpdater( train_iter, opt, device=0 )
trainer = training.Trainer(updater, (args.epochs, 'epoch'), out='channel_corr_results')
trainer.extend( extensions.Evaluator(valid_iter, model, device=0) )
trainer.extend( extensions.PrintReport(['epoch', 'validation/main/loss', 'elapsed_time']) )
trainer.extend( extensions.ProgressBar() )
trainer.run()

