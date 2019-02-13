import argparse
import numpy as np

print(" ")
print(" ")
print("*===================================================================================*")
print("*                           ERA5 INPUT DATA PREPARATION                             *")
print("*===================================================================================*")
print(" ")

##
## Parse any user given commandline arguments 
##

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_bins', type=int, default=4, help="number of rainfall classification bins")
args = parser.parse_args()

if args.num_bins<2:
   args.num_bins = 2
elif args.num_bins>6:
   args.num_bins = 6

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
print("      splitting ERA5 Australia-specific data into %d bins" % (args.num_bins))
print(" ")

##
## Collect variable data for each pressure level
##

for n in range( len(levels) ):
    print("      collecting training variable data for %d hPa pressure level" % (levels[n]))
    
    features_data = np.empty((num_training_images,image_width,image_height,len(variables)), dtype=np.float)
    for i in range( len(variables) ):
        filename = "../input_data/training/" + str(levels[n]) + "hPa/"  + variables[i] + "_era5_au_" + str(args.num_bins) + "bins.npy"
        features_data[ :,:,:,i ] = np.load( filename )

    filename = "../input_data/training/era5_au_" + str(args.num_bins) + "bins.npy"
    np.save( filename, features_data )

    print("      collecting test variable data for %d hPa pressure level" % (levels[n]))
    features_data = np.empty((num_test_images,image_width,image_height,len(variables)), dtype=np.float)
    for i in range( len(variables) ):
        filename = "../input_data/test/" + str(levels[n]) + "hPa/"  + variables[i] + "_era5_au_" + str(args.num_bins) + "bins.npy"
        features_data[ :,:,:,i ] = np.load( filename )

    filename = "../input_data/test/era5_au_" + str(args.num_bins) + "bins.npy"
    np.save( filename, features_data )
    print(" ")

