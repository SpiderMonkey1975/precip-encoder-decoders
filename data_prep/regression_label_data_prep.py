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
   print("      splitting global ERA5 data")
else:
   variables = ['z','t','rh']
   varnames = ['atmospheric pressure','atmospheric temperature','relative humidity']
   print("      splitting ERA5 Australia-specific data")
print(" ")

##
## Read in total raw precipitation data
##

#filename = "/scratch/director2107/ERA5_Data/ERA5_Native/tp_era5_" + args.data + ".npy"
#precip_data = np.load( filename )
#print("      STATUS:")
#print("      [1/2] precipitation data preparation")
#print("           -> raw data read")

##
## Generate index sets for the traininig and test data sets 
##

#num_records = precip_data.shape[0]
#indicies = np.empty((num_records,), dtype=int)
#for i in range(num_records):
#    indicies[i] = i
#np.random.shuffle( indicies )

#training_indicies = indicies[ :num_training_images ]
#test_indicies = indicies[ num_training_images: ]
#print("           -> indicies selected")

##
## Output label data to hard disk
##

#varname = args.data + "_labels.npy"       
#filename = "../input_data/training/" + varname
#train_set = precip_data[ training_indicies,:,: ]
#np.save( filename, train_set )

#filename = "../input_data/test/" + varname
#test_set = precip_data[test_indicies,:,: ]
#np.save( filename, test_set )
#print("           -> label data written to hard disk")
#print(" ")
#print(" ")

##
## Read in and process features data 
##

#print("      [2/2] processing features data")

#cnt = 0
#for var in variables:
#   print("           -> %s" % (varnames[cnt]))
#   filename = "/scratch/director2107/ERA5_Data/ERA5_Native/" + var + "_era5_" + args.data + ".npy"
#   features_data = np.load( filename )
#   print("              * data read from hard disk")

#   features_data = np.moveaxis(features_data, 1, 3)
#   print("              * axis swap performed")
#   cnt = cnt + 1

##
## Output processed features data to hard disk
##

#   filename = "../input_data/training/" + var + "_era5_" + args.data + ".npy" 
#   np.save( filename, features_data[ training_indicies,:,:,: ] )

#   filename = "../input_data/test/" + var + "_era5_" + args.data + ".npy" 
#   np.save( filename, features_data[ test_indicies,:,:,: ] )

#   for n in range( len(levels) ):
#       filename = "../input_data/training/" + str(levels[n]) + "hPa/" + var + "_era5_" + args.data + ".npy" 
#       np.save( filename, features_data[ training_indicies,:,:,n ] )
#       filename = "../input_data/test/" + str(levels[n]) + "hPa/" + var + "_era5_" + args.data + ".npy" 
#       np.save( filename, features_data[ test_indicies,:,:,n ] )
#
#   print("              * features data written to hard disk")
#   print(" ")


features_data = np.empty((30000,240,360,3),dtype=np.float)
features_data2 = np.empty((648,240,360,3),dtype=np.float)

for n in range( len(levels) ):
    for var in variables:

        filename = "../input_data/training/" + str(levels[n]) + "hPa/" + var + "_era5_" + args.data + ".npy"
        tmp = np.load( filename )
        features_data[ :,:,:,n ] = tmp[ :,:240,:360 ]

        filename = "../input_data/test/" + str(levels[n]) + "hPa/" + var + "_era5_" + args.data + ".npy"
        tmp = np.load( filename )
        features_data2[ :,:,:,n ] = tmp[ :,:240,:360 ]
            
    filename = "../input_data/training/" + str(levels[n]) + "hPa/era5_all_vars_" + args.data + ".npy"
    np.save( filename, features_data ) 
            
    filename = "../input_data/test/" + str(levels[n]) + "hPa/era5_all_vars_" + args.data + ".npy"
    np.save( filename, features_data2 ) 
