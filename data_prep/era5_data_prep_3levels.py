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
## Read in total precipitation data
##

filename = "/scratch/director2107/ERA5_Data/ERA5_Native/tp_era5_" + args.data + ".npy"
precip_data = np.load( filename )
print("      STATUS:")
print("      [1/2] precipitation data preparation")
print("           -> raw data read")

##
## Determine maximum precipitation value for every record
##

num_records = precip_data.shape[0]
max_precip_vals = np.empty((num_records,), dtype=float)

for n in range( num_records ):
    max_precip_vals[n] = np.amax(precip_data[n,:,:]) 

indicies = np.argsort( max_precip_vals )

##
## Generate an one-hot encoding for the precipitation data 
##

records_per_bin = int(np.floor( num_records/args.num_bins ))
num_records = int(records_per_bin*args.num_bins)

one_hot_encoding = np.zeros((num_records,args.num_bins), dtype=int)

bin_vals = np.empty((args.num_bins), dtype=float)

print("      Bin Values:")
idx = 0
n = records_per_bin - 1
for i in range(args.num_bins):
    bin_vals[i] = 1000*max_precip_vals[indicies[idx]]
    print( "          %3.0f - %3.0f mm" % (bin_vals[i],1000*max_precip_vals[indicies[n]]))
    idx = idx + records_per_bin
    n = n + records_per_bin

for i in range(args.num_bins):
    for j in range(records_per_bin):
        idx = indicies[ i*records_per_bin + j ]
        one_hot_encoding[idx,i] = 1
print("           -> one-hot encoding completed")

##
## Generate index sets for the traininig and test data sets 
##

indicies = np.empty((num_records,), dtype=int)
for i in range(num_records):
    indicies[i] = i
np.random.shuffle( indicies )

training_indicies = indicies[ :num_training_images ]
test_indicies = indicies[ num_training_images: ]
print("           -> indicies selected")

##
## Output label data to hard disk
##

varname = args.data + "_labels_" + str(args.num_bins) + "bins.npy"       
filename = "../input_data/training/" + varname
train_set = one_hot_encoding[ training_indicies,: ]
np.save( filename, train_set )

filename = "../input_data/test/" + varname
test_set = one_hot_encoding[ test_indicies,: ]
np.save( filename, test_set )
print("           -> label data written to hard disk")
print(" ")
print(" ")

##
## Read in and process features data 
##

print("      [2/2] processing features data")

cnt = 0
for var in variables:
   print("           -> %s" % (varnames[cnt]))
   filename = "/scratch/director2107/ERA5_Data/ERA5_Native/" + var + "_era5_" + args.data + ".npy"
   features_data = np.load( filename )
   print("              * data read from hard disk")

   features_data = np.moveaxis(features_data, 1, 3)
   print("              * axis swap performed")

   for i in range( len(levels) ):
       for n in range( features_data.shape[0] ):
           mean_val = np.mean( features_data[n,:,:,i] )
           std_dev = np.std( features_data[n,:,:,i] )
           features_data[n,:,:,i] = (features_data[n,:,:,i] - mean_val) / std_dev
   print("              * normalization completed")
   cnt = cnt + 1

##
## Output processed features data to hard disk
##

   filename = "../input_data/training/" + var + "_era5_" + args.data + "_" + str(args.num_bins) + "bins.npy" 
   np.save( filename, features_data[ training_indicies,:,:,: ] )

   filename = "../input_data/test/" + var + "_era5_" + args.data + "_" + str(args.num_bins) + "bins.npy" 
   np.save( filename, features_data[ test_indicies,:,:,: ] )
   print("              * features data written to hard disk")
   print(" ")
print(" ")
