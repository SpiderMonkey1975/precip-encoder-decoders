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
parser.add_argument('-d', '--data', type=str, default='au', help="dataset type: native, au")
parser.add_argument('-n', '--num_bins', type=int, default=4, help="number of rainfall classification bins")
args = parser.parse_args()

if args.data != "au" and args.data != "native":
   args.data = "au"

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

if args.data == "native":
   variables = ['z']
   varnames = ['atmospheric pressure']
   print("      splitting global ERA5 data into %d bins" % (args.num_bins))
else:
   variables = ['z','t','rh']
   varnames = ['atmospheric pressure','atmospheric temperature','relative humidity']
   print("      splitting ERA5 Australia-specific data into %d bins" % (args.num_bins))
print(" ")

##
## Split variable data between the 3 different pressure levels
##

for i in range( len(variables) ):
   print("      splitting %s among pressure levels" % (varnames[i]))
   filename = "../input_data/training/" + variables[i] + "_era5_" + args.data + "_" + str(args.num_bins) + "bins.npy" 
   features_data = np.load( filename )
   print("              * data read from hard disk")

   for n in range( len(levels) ):
       filename = "../input_data/training/" + str(levels[n]) + "hPa/"  + variables[i] + "_era5_" + args.data + "_" + str(args.num_bins) + "bins.npy" 
       np.save( filename, features_data[ :,:,:,n ] )
   print("              * features data written to hard disk")

   filename = "../input_data/test/" + variables[i] + "_era5_" + args.data + "_" + str(args.num_bins) + "bins.npy" 
   features_data = np.load( filename )
   print("              * data read from hard disk")

   for n in range( len(levels) ):
       filename = "../input_data/test/" + str(levels[n]) + "hPa/"  + variables[i] + "_era5_" + args.data + "_" + str(args.num_bins) + "bins.npy" 
       np.save( filename, features_data[ :,:,:,n ] )
   print("              * features data written to hard disk")
   print(" ")
print(" ")
