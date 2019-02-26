import numpy as np
import tensorflow as tf
import sys, argparse

from tensorflow.keras import models, layers
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

import neural_nets
from neural_nets import classifier


print(" ")
print(" ")
print("*===================================================================================================*")
print("*                                EVALUATION OF RAINFALL CLASSIFIER                                  *")
print("*===================================================================================================*")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--variable', type=str, default="rh", help="prognostic variable to be used in test. Valid options: rh, t, z, 3var")
parser.add_argument('-l', '--level', type=str, default="800", help="pressure level to be used in test. Valid options: 500, 800, 1000, all_levels")
parser.add_argument('-b', '--batch_size', type=int, default=250, help="set batch size per GPU")
parser.add_argument('-c', '--num_classes', type=int, default=6, help="set number of classes for the image classification")
args = parser.parse_args()

if args.level == "all_levels" and args.variable=="3var":
   print("ERROR: cannot have 3 variable output from all pressure levels")
   sys.exit(0)

num_channels = 1
if args.level == "all_levels" or args.variable=="3var":
   num_channels = 3

if args.level != "all_levels":
   lev_dir = args.level + "hPa/"
else:
   lev_dir = "all_levels"

##
## Set some important parameters
##

image_width = 240
image_height = 360

num_nodes = 50
dropout = 0.3
layers = 3
learn_rate = 0.0001

print(" ")
print(" ")
print("       Model Settings:")
print("         * using ERA5-Australia specific input over %d bins" % (args.num_classes))
print("         * a batch size of %2d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (learn_rate))
print("         * dropout ratio is %f" % (dropout))
print("         * using 3-layer classifier with %d, %d and %d numbers of hidden nodes" % (8*num_nodes,4*num_nodes,num_nodes))
print(" ")

##
## Construct the neural network 
##

input_layer = Input(shape = (image_width, image_height, num_channels))
net = classifier( input_layer, int(num_nodes), args.num_classes, float(dropout), 3 )

model = models.Model(inputs=input_layer, outputs=net)

##
## Load weights
##

filename = "../model_backups/" + str(args.num_classes) + "bins/model_weights_" + var + "_" + lev + ".h5"
print("       Weights File: %s" % (filename))
model.load_weights( filename )

##
## Set the appropriate optimizer and loss function 
##

opt = Adam(lr=learn_rate)
model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## 
## Load the test input data from disk
##

filename = "../input/test/au/" + lev_dir + args.variable + "_normalized.npy"
print("       Features File: %s" % (filename))
x_test = np.load( filename )
if num_channels==1:
   x_test = np.expand_dims( x_test, axis=3 )

filename = "../input/au/test/labels_" + str(args.num_classes) + "bins.npy"
print("       Labels File: %s" % (filename))
y_test = np.load( filename )

num_images = np.amin( [x_test.shape[0],y_test.shape[0]] )

x_test = x_test[ :num_images, :image_width, :image_height, : ]
y_test = y_test[ :num_images, : ]

##
## Send test data through the trained model
##

score = model.evaluate( x=x_test, y=y_test, batch_size=args.batch_size, verbose=0 )

print(" ")
print(" ")
print("                                        Model Evaluation")
print("*---------------------------------------------------------------------------------------------------*")
print(" ")
print("    Overall Accuracy: %4.3f" % (score[1]))

actual = np.argmax( y_test, axis=1 )
predicted = np.argmax( model.predict( x_test ), axis=1 )

true_bin_counts = np.bincount( actual )
print( true_bin_counts, np.sum(true_bin_counts) )

tmp = np.argmax( np.bincount( predicted ))
cnt = 0
for i in range( len(actual) ):
    if actual[i] == tmp:
       cnt = cnt + 1
print("      Null Error Rate: %4.3f" % (cnt/len(actual)))
print(" ")


for n in range(args.num_classes):
    print("   Bin %d Statistics " % (n))
   
    true_pos_cnt = 0 
    false_pos_cnt = 0 
    true_neg_cnt = 0
    false_neg_cnt = 0

    for i in range( len(actual) ):
        if actual[i]==n and predicted[i]==n:
           true_pos_cnt = true_pos_cnt + 1
        elif actual[i]!=n and predicted[i]==n:
           false_pos_cnt = false_pos_cnt + 1
        elif actual[i]!=n and predicted[i]!=n:
           true_neg_cnt = true_neg_cnt + 1
        elif actual[i]==n and predicted[i]!=n:
           false_neg_cnt = false_neg_cnt + 1

    print( len(actual), true_bin_counts[n] )
    print("                  accuracy: %4.3f" % ((true_pos_cnt+true_neg_cnt)/len(actual)))
    print("        true positive rate: %4.3f" % (true_pos_cnt/true_bin_counts[n]))
    print("       false positive rate: %4.3f" % (false_pos_cnt/(len(actual)-true_bin_counts[n])))
    print("        true negative rate: %4.3f" % (true_neg_cnt/(len(actual)-true_bin_counts[n])))
    print("       false negative rate: %4.3f" % (false_neg_cnt/(len(actual)-true_bin_counts[n])))
    print("     prevalance of Bin %d : %4.3f" % (n,true_bin_counts[n]/len(actual)))
    print(" ")

