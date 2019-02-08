import numpy as np
import tensorflow as tf
import sys

from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, applications
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.utils import multi_gpu_model

##
## Set some important parameters
##

num_images = 30648
image_width = 240
image_height = 360

batch_size = 32
epochs = 100
num_classes = 6
num_nodes = 32

##
## Construct the neural network 
##

input_layer = layers.Input(shape = (image_width, image_height, 3))

base_model = VGG16( input_tensor=input_layer,
                    input_shape=(image_width, image_height, 3), 
                    weights='imagenet', 
                    include_top=False ) 

net = base_model.output
net = Flatten()(net)

factor = 8
for n in range(2):
    net = Dense( factor*num_nodes, activation='relu' )(net)
    net = Dropout(0.3)(net)
    factor = factor / 2
    if factor<1:
       factor = 1
#    net = Dense( 4*num_nodes, activation='relu', kernel_regularizer=l2(reg_constant) )(net)
 #   net = Dropout(dropout_ratio)(net)
#    net = Dense( num_nodes, activation='relu', kernel_regularizer=l2(reg_constant) )(net)
net = Dense( num_classes, activation='softmax' )(net)

#x=GlobalAveragePooling2D()(x)
#x=Dense( 1024, activation='relu' )(x) 
#x=Dense( 1024, activation='relu' )(x) 
#x=Dense(  512, activation='relu' )(x) 
#net=Dense( num_classes, activation='softmax' )(x)

with tf.device("/cpu:0"):
        model = models.Model(inputs=input_layer, outputs=net)
model = multi_gpu_model( model, gpus=4 )

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

##
## Set the appropriate optimizer and loss function 
##

opt = Adam(lr=0.0001)
model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## 
## Load all 3 variables for the selected pressure level
##

x_train = np.empty((30000,image_width,image_height,3),dtype=float)

variables = ['z','t','rh']
cnt = 0
for var in variables:
    inputfile = "input_data/training/800hPa/" + var + "_era5_au_6bins.npy"
    tmp = np.load( inputfile )
    x_train[ :,:,:,cnt ] = tmp[ :,:image_width,:image_height ] 
    cnt = cnt + 1

inputfile = "input_data/training/au_labels_6bins.npy"
y_train = np.load( inputfile )

##
## Train model.  Only output information for the validation steps only.
##

print(" ")
print(" ")
print("                                       Model Training Output")
print("*---------------------------------------------------------------------------------------------------*")

history = model.fit( x=x_train, y=y_train, 
                     batch_size=4*batch_size, 
                     epochs=epochs, 
                     verbose=2,
                     shuffle=False, 
                     validation_split=0.25 )
print(" ")

sys.exit(0)
## 
## Load the test input data from disk
##

x_test = np.empty((648,image_width,image_height,3),dtype=float)

cnt = 0
for var in variables:
    inputfile = "input_data/test/800hPa/" + var + "_era5_au_6bins.npy"
    tmp = np.load( inputfile )
    x_test[ :,:,:,cnt ] = tmp[ :,:image_width,:image_height ] 
    cnt = cnt + 1

inputfile = "input_data/test/au_labels_6bins.npy"
y_test = np.load( inputfile )

##
## End by sending test data through the trained model
##

print(" ")
print(" ")
print("                                        Model Prediction Test")
print("*---------------------------------------------------------------------------------------------------*")

score = model.evaluate( x=x_test, y=y_test, 
                        batch_size=batch_size,
                        verbose=0 )
print("       Test on %s samples, Accuracy of %4.3f"  % (num_images,score[1]))
print( score )
print(" ")

