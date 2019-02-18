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
print("*                                         RAINFALL CLASSIFIER                                       *")
print("*===================================================================================================*")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_file', type=str, help="name of the file containing initial weights")
parser.add_argument('-f', '--features_file', type=str, help="name of the file containing feature data")
parser.add_argument('-b', '--batch_size', type=int, default=250, help="set batch size per GPU")
parser.add_argument('-l', '--levels', type=int, default=800, help="set pressure level to be used")
parser.add_argument('-c', '--channels', type=int, default=3, help="set number of channels")
args = parser.parse_args()

##
## Set some important parameters
##

image_width = 240
image_height = 360

num_nodes = 50
bins = 6
dropout = 0.3
layers = 3
learn_rate = 0.0001

print(" ")
print(" ")
print("       Model Settings:")
print("         * using ERA5-Australia specific input over %d bins" % (bins))
print("         * a batch size of %2d images per GPU will be employed" % (args.batch_size))
print("         * the ADAM optimizer with a learning rate of %6.4f will be used" % (learn_rate))
print("         * dropout ratio is %f" % (dropout))
print("         * using 3-layer classifier with %d, %d and %d numbers of hidden nodes" % (8*num_nodes,4*num_nodes,num_nodes))

##
## Construct the neural network 
##

input_layer = Input(shape = (image_width, image_height, args.channels))
net = classifier( input_layer, int(num_nodes), int(bins), 0.3, args.channels )

model = models.Model(inputs=input_layer, outputs=net)

##
## Load weights
##

model.load_weights( args.weights_file )

##
## Set the appropriate optimizer and loss function 
##

opt = Adam(lr=learn_rate)
model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## 
## Load the test input data from disk
##

x_test = np.load( args.features_file )

filename = "../input_data/test/au_labels_" + str(bins) + "bins.npy"
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
print("          Error Rate: %4.3f" % (1.0-score[1]))

actual = np.argmax( y_test, axis=1 )
predicted = np.argmax( model.predict( x_test ), axis=1 )

true_bin_counts = np.bincount( actual )

tmp = np.argmax( np.bincount( predicted ))
cnt = 0
for i in range( len(actual) ):
    if actual[i] == tmp:
       cnt = cnt + 1
print("      Null Error Rate: %4.3f" % (cnt/len(actual)))
print(" ")


for n in range(bins):
    print("   Bin %d Statistics " % (n))
   
    true_pos_cnt = 0 
    false_pos_cnt = 0 
    true_neg_cnt = 0

    for i in range( len(actual) ):
        if actual[i]==n and predicted[i]==n:
           true_pos_cnt = true_pos_cnt + 1
        elif actual[i]!=n and predicted[i]==n:
           false_pos_cnt = false_pos_cnt + 1
        elif actual[i]!=n and predicted[i]!=n:
           true_neg_cnt = true_neg_cnt + 1

    print("                  accuracy: %4.3f" % ((true_pos_cnt+true_neg_cnt)/len(actual)))
    print("                error rate: %4.3f" % (1.0-(true_pos_cnt+true_neg_cnt)/len(actual)))
    print("        true positive rate: %4.3f" % (true_pos_cnt/true_bin_counts[n]))
    print("       false positive rate: %4.3f" % (false_pos_cnt/(len(actual)-true_bin_counts[n])))
    print("        true negative rate: %4.3f" % (1.0 - false_pos_cnt/(len(actual)-true_bin_counts[n])))
    print("      classifier precision: %4.3f" % (true_pos_cnt/(true_pos_cnt+false_pos_cnt)))
    print("                prevalance: %4.3f" % (true_bin_counts[n]/len(actual)))
    print(" ")

