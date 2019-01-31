import argparse, sys
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

print("      splitting ERA5-%s data into %d bins" % (args.data,args.num_bins))
print(" ")

#if args.data == "native":
#   limits = [5.18112,6.6853,8.922]
#else:
#   limits = [14.7539,19.222,23.9865]

##
## Read in total precipatation data
##

filename = "/scratch/director2107/ERA5_Data/ERA5_Native/tp_era5_" + args.data + ".npy"
precip_data = np.load( filename )
print("      STATUS:")
print("      [1/2] precipitation data preparation")
print("           -> raw data read")

##
## Determine maximum precipatation value for every record
##

num_records = precip_data.shape[0]
max_precip_vals = np.empty((num_records,), dtype=float)

for n in range( num_records ):
    max_precip_vals[n] = np.amax(precip_data[n,:,:]) 

indicies = np.argsort( max_precip_vals )

##
## Generate an one-hot encoding for the precipatation data 
##

records_per_bin = int(np.floor( num_records/args.num_bins ))
num_records = int(records_per_bin*args.num_bins)

one_hot_encoding = np.zeros((num_records,args.num_bins), dtype=int)


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

training_indicies = indicies[ :30000 ]
test_indicies = indicies[ 30000: ]
print("           -> indicies selected")

##
## Output label data to hard disk
##

varname = args.data + "_labels_" + str(args.num_bins) + "bins.npy"       
filename = "input_data/training/" + varname
train_set = one_hot_encoding[ training_indicies,: ]
np.save( filename, train_set )

filename = "input_data/test/" + varname
test_set = one_hot_encoding[ test_indicies,: ]
np.save( filename, test_set )
print("           -> label data written to hard disk")
print(" ")
print(" ")

##
## Read in and process features data 
##

print("      [2/2] processing features data")
if args.data == "native":
   variables = ['z']
   varnames = ['atmospheric pressure']
else:
   variables = ['z','t','rh']
   varnames = ['atmospheric pressure','atmospheric temperature','relative humidity']

cnt = 0
for var in variables:
   print("           -> %s" % (varnames[cnt]))
   filename = "/scratch/director2107/ERA5_Data/ERA5_Native/" + var + "_era5_" + args.data + ".npy"
   features_data = np.load( filename )
   print("              * data read from hard disk")

   features_data = np.moveaxis(features_data, 1, 3)
   print("              * axis swap performed")

   for i in range( 3 ):
       for n in range( features_data.shape[0] ):
           mean_val = np.mean( features_data[n,:,:,i] )
           std_dev = np.std( features_data[n,:,:,i] )
           features_data[n,:,:,i] = (features_data[n,:,:,i] - mean_val) / std_dev
   print("              * normalization completed")
   cnt = cnt + 1

##
## Output processed features data to hard disk
##

   levels = [500,800,1000]

   train_set = features_data[ training_indicies,:,:,: ]
   test_set = features_data[ test_indicies,:,:,: ]
   print("              * training/test set selection done")

   for i in range(3):
       varname = var + "_era5_" + args.data + "_" + str(args.num_bins) + "bins.npy"
       filename = "input_data/training/"  + str(levels[i]) + "hPa/"+ varname
       np.save( filename, train_set[ :,:,:,i ] )

       filename = "input_data/test/"  + str(levels[i]) + "hPa/"+ varname
       np.save( filename, test_set[ :,:,:,i ] )
   print("              * features data written to hard disk")
   print(" ")
print(" ")
