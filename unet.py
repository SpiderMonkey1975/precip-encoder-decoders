import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras import layers, models
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras.utils.training_utils import multi_gpu_model

## Not currently used

#from keras.optimizers import Adam
#import pickle

##
## Set some important parameters
##

num_training_images = 40000
num_gpus = 4
batch_size = 10 # if using float16 precision, if using float32 precision, set it to 64
epochs = 50

##
## Define a subroutine that contructs the required neural network
##

def get_unet( num_gpu  ):
    concat_axis = 3
    inputs = layers.Input(shape = (80, 120, 3))

    bn0 = BatchNormalization(axis=3)(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(bn0)
    bn1 = BatchNormalization(axis=3)(conv1)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(bn1)
    bn2 = BatchNormalization(axis=3)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(bn2)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    bn3 = BatchNormalization(axis=3)(conv2)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn3)
    bn4 = BatchNormalization(axis=3)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    bn5 = BatchNormalization(axis=3)(conv3)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(bn5)
    bn6 = BatchNormalization(axis=3)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(bn6)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    bn7 = BatchNormalization(axis=3)(conv4)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(bn7)
    bn8 = BatchNormalization(axis=3)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 3))(bn8)

    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    bn9 = BatchNormalization(axis=3)(conv5)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(bn9)
    bn10 = BatchNormalization(axis=3)(conv5)

    up_conv5 = layers.UpSampling2D(size=(2, 3))(bn10)
    up6 = layers.concatenate([up_conv5, conv4], axis=concat_axis)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    bn11 = BatchNormalization(axis=3)(conv6)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(bn11)
    bn12 = BatchNormalization(axis=3)(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(bn12)
    up7 = layers.concatenate([up_conv6, conv3], axis=concat_axis)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    bn13 = BatchNormalization(axis=3)(conv7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(bn13)
    bn14 = BatchNormalization(axis=3)(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(bn14)
    up8 = layers.concatenate([up_conv7, conv2], axis=concat_axis)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    bn15 = BatchNormalization(axis=3)(conv8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn15)
    bn16 = BatchNormalization(axis=3)(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(bn16)
    up9 = layers.concatenate([up_conv8, conv1], axis=concat_axis)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    bn17 = BatchNormalization(axis=3)(conv9)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(bn17)
    bn18 = BatchNormalization(axis=3)(conv9)

    conv10 = layers.Conv2D(1, (1, 1))(bn18)

    if ( num_gpu <= 1 ):
       model = models.Model(inputs=inputs, outputs=conv10)
    else:
       with tf.device("/cpu:0"):
            model = models.Model(inputs=inputs, outputs=conv10)
       model = multi_gpu_model( model, gpus=num_gpu )

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mae', optimizer=sgd, metrics=['mse'])
    #model.compile(loss='mae', optimizer=Adam(lr=0.001), metrics=['mse'])
    #print model.summary()

    return model

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
## Divide the input dataset into training and testing sets
##

y_train = y[:num_training_images, :]
y_test = y[num_training_images:54000, :]

# Levels [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]

x_train = x[:num_training_images, :, :, [0,2,6]]
x_test = x[num_training_images:54000, :, :, [0,2,6]]

##
## Construct the neural network and start training
##

model = get_unet( num_gpus )

batch_size = num_gpus*batch_size 
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Commented out for benchmarking purposes
#with open('trainHistoryDict_unet1_{}-{}-{}'.format(i, j, k), 'wb') as file_pi:
#    pickle.dump(history.history, file_pi)

#model.save('/datasets/unet1_{}-{}-{}_.h5')
