import numpy as np
import tensorflow as tf
import argparse

from datetime import datetime

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import EarlyStopping
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
parser.add_argument('-b', '--batch_size', type=int, default=128, help="set batch size per GPU")
parser.add_argument('-l', '--learn_rate', type=float, default=0.001, help="set intial learning rate for optimizer")
parser.add_argument('-r', '--l1_reg', type=float, default=0.01, help="set L1 regularization parameter")
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
if args.l1_reg > 0.0001:
   print("         * L1 regularization is enabled with L1 =", args.l1_reg)


##
## Contruct the neural network
##

inputs = layers.Input(shape = (80, 120, 3))

net = inputs
reg = regularizers.l1( args.l1_reg )
filter_size = 64

# Encoder
for n in range(5):
    for i in range(2):
        bn = BatchNormalization(axis=3)(net)
        net = layers.Conv2D(filter_size, 3, activation='relu', padding='same', bias_regularizer=reg)(bn)
    if n == 3:
       net = layers.MaxPool2D(pool_size=(2,3))(net)
    elif n > 3:
       net = layers.MaxPool2D( 2 )(net)
       filter_size = 2 * filter_size


#def get_vgg16():
#    model = Sequential()

    # Encoder
    # Block 1
#    model.add(BatchNormalization(axis=3, input_shape=(80, 120, 3)))
#    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block1_conv1', input_shape=(80,120,3)))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block1_conv2'))
#    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
#
    # Block 2
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block2_conv1'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block2_conv2'))
#    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block3_conv1'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block3_conv2'))
#    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4i
#    model.add(BatchNormalization(axis=3))
 #   model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block4_conv1'))
 #   model.add(BatchNormalization(axis=3))
 #   model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block4_conv2'))
  #  model.add(MaxPooling2D((2, 3), strides=(2, 3), name='block4_pool'))


    # Block 5
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block5_conv1'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block5_conv2'))

# Decoder
for n in range(4):
    if n == 0:
       net = layers.UpSampling2D(size=(2, 3))(net)
    else:
       net = layers.UpSampling2D( 2 )(net)
    for i in range(2):
        bn = BatchNormalization(axis=3)(net)
        net = layers.Conv2D(filter_size, 3, activation='relu', padding='same', bias_regularizer=reg)(bn)
    filter_size = int(filter_size / 2)

bn = BatchNormalization(axis=3)(net)
net = layers.Conv2D(1, 1, activation='relu', padding='same', bias_regularizer=regularizers.l1(args.l1_reg))(bn) 

    # Block 6
#    model.add(UpSampling2D((2, 3), name='block6_upsampl'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block6_conv1'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block6_conv2'))

    # Block 7
#    model.add(UpSampling2D((2, 2), name='block7_upsampl'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block7_conv1'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block7_conv2'))

    # Block 8
#    model.add(UpSampling2D((2, 2), name='block8_upsampl'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block8_conv1'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block8_conv2'))

    # Block 9
#    model.add(UpSampling2D((2, 2), name='block9_upsampl'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block9_conv1'))
#    model.add(BatchNormalization(axis=3))
#    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block9_conv2'))

    # Output
#    model.add(BatchNormalization(axis=3))
 #   model.add(Conv2D(1, (1, 1), padding='same', activation='relu', bias_regularizer=regularizers.l1(0.01), name='block10_conv1'))


##
## Create separate data-parallel instances of the neural net on each GPU
##

if ( args.num_gpus <= 1 ):
   model = models.Model(inputs=inputs, outputs=net)
else:
   with tf.device("/cpu:0"):
        model = models.Model(inputs=inputs, outputs=net)
   model = multi_gpu_model( model, gpus=args.num_gpus )


##
## Set the optimizer to be used and compile model
##

opt = Adam(lr=args.learn_rate)
model.compile(loss='mae', optimizer=opt, metrics=['mse'])


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
