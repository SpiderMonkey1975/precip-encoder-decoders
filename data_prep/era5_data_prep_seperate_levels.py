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
args = parser.parse_args()

if args.data != "au" and args.data != "native":
   args.data = "au"

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
print(" ")

approach_str = ['image classification', 'regression']
my_str = ['', '_not']

##
## Split variable data between the 3 different pressure levels
##

for j in range( len(approach_str) ):
    print("      preparing data for %s approach" % (approach_str[j]))
    for i in range( len(variables) ):
        var_str = variables[i] + my_str[j] + "_normalized.npy"

        print("      splitting %s among pressure levels" % (varnames[i]))
        filename = "../input/" + args.data + "/training/all_levels/" + var_str 
        features_data = np.load( filename )
        print("              * data read from hard disk")

        for n in range( len(levels) ):
            filename = "../input/" + args.data + "/training/" + str(levels[n]) + "hPa/" + var_str  
            np.save( filename, features_data[ :,:,:,n ] )
        print("              * features data written to hard disk")

        filename = "../input/" + args.data + "/test/all_levels/" + var_str 
        features_data = np.load( filename )
        print("              * data read from hard disk")

        for n in range( len(levels) ):
            filename = "../input/" + args.data + "/test/" + str(levels[n]) + "hPa/" + var_str 
            np.save( filename, features_data[ :,:,:,n ] )
        print("              * features data written to hard disk")
        print(" ")
    print(" ")

