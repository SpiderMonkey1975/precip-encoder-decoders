import numpy as np

print(" ")
print(" ")
print("*===================================================================================*")
print("*                           ERA5 INPUT DATA PREPARATION                             *")
print("*===================================================================================*")
print(" ")

##
## Set some important parameters
##

image_width = 240
image_height = 360
num_training_images = 30000
num_test_images = 648
levels = [500,800,1000]

variables = ['z','t','rh']
varnames = ['atmospheric pressure','atmospheric temperature','relative humidity']
#my_str = ['', '_not']
#approach_str = ['image_classification', 'regression']
my_str = ['_not']
approach_str = ['regression']
print(" ")

##
## Collect variable data for each pressure level
##

for j in range( len(my_str) ):

    print("      Preparing feature data for the %s approach:" % (approach_str[j]))
    for n in range( len(levels) ):
        print("         - collecting training variable data for %d hPa pressure level" % (levels[n]))
   
        train_dir = "../input/au/training/" + str(levels[n]) + "hPa/"
        test_dir = "../input/au/test/" + str(levels[n]) + "hPa/"

        features_data = np.empty((num_training_images,image_width,image_height,len(variables)), dtype=np.float)
        for i in range( len(variables) ):
            filename = train_dir  + variables[i] + my_str[j] + "_normalized.npy"
            tmp = np.load( filename )
            features_data[ :,:,:,i ] = tmp[ :num_training_images,:image_width, :image_height ]

        filename = train_dir + "3var" + my_str[j] + "_normalized.npy"
        np.save( filename, features_data )

        print("         - collecting test variable data for %d hPa pressure level" % (levels[n]))
        features_data = np.empty((num_test_images,image_width,image_height,len(variables)), dtype=np.float)
        for i in range( len(variables) ):
            filename = test_dir + variables[i] + my_str[j] + "_normalized.npy"
            tmp = np.load( filename )
            features_data[ :,:,:,i ] = tmp[ :num_test_images,:image_width, :image_height ]

        filename = test_dir + "3var" + my_str[j] + "_normalized.npy"
        np.save( filename, features_data )
        print(" ")

