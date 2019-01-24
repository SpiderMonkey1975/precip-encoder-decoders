import argparse, sys
import numpy as np

print(" ")
print(" ")
print("*===================================================================================*")
print("*                           ERA5 INPUT DATA PREPARATION                             *")
print("*===================================================================================*")
print(" ")

##
## User selected Native or Australian region-specific data
##

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='au', help="dataset type: native, au")
args = parser.parse_args()

if args.data == "native":
   limits = [5.18112,6.6853,8.922]
   print("      working on global ERA5 data")
else:
   limits = [14.7539,19.222,23.9865]
   print("      working on Australia-specific ERA5 data")
print(" ")

##
## Read in total precipatation data
##

filename = "/scratch/director2107/ERA5_Data/ERA5_Native/tp_era5_" + args.data + ".npy"
precip_data = 1000.0 * np.load( filename )
print("      STATUS:")
print("      [1/2] precipitation data read")
print("           -> total precipitation data read")

##
## Generate labels array(s)
##

num_records = precip_data.shape[0]
one_hot_encoding = np.zeros((num_records,4), dtype=int)

idx = 0
for n in range( num_records ):
    val = np.amax(precip_data[n,:,:])

    # Class 1 - extreme rainfall events (over 30mm per hour)
    if val > limits[2]:
       one_hot_encoding[idx,0] = 1

    # Class 2 - heavy rainfall events (between 30 and 15mm per hour)
    elif val > limits[1] and val <= limits[2]:
       one_hot_encoding[idx,1] = 1

    # Class 3 - rainfall events (between 10 and 15mm per hour)
    elif val > limits[0] and val <= limits[1]:
       one_hot_encoding[idx,2] = 1

    # Class 4 - insignificant or no rainfall events (under 10mm per hour)
    else:
       one_hot_encoding[idx,3] = 1
    idx = idx + 1
print("           -> one-hot encoding completed")

##
## Select the same number of rainfall class instances for the image recognition 
##

num_records = np.min( np.sum(one_hot_encoding,axis=0) )

cnt = np.zeros( 4, dtype=int )
indicies = np.empty( 4*num_records, dtype=int )

idx = 0
for n in range( one_hot_encoding.shape[0] ):
    for i in range( 4 ):
        if one_hot_encoding[n,i] == 1 and cnt[i] < num_records:
           cnt[i] = cnt[i] + 1
           indicies[idx] = n
           idx = idx + 1 

np.random.shuffle( indicies )
np.random.shuffle( indicies )

training_indicies = indicies[ :30000 ]
np.random.shuffle( training_indicies )

test_indicies = indicies[ 30000: ]
np.random.shuffle( test_indicies )
print("           -> indicies selected")

##
## Output label data to hard disk
##
       
filename = "input_data/training/" + args.data + "_one_hot_encoding.npy"
train_set = one_hot_encoding[ training_indicies,: ]
np.save( filename, train_set )

filename = "input_data/test/" + args.data + "_one_hot_encoding.npy"
test_set = one_hot_encoding[ training_indicies,: ]
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
       filename = "input_data/training/" + var + "_era5_" + args.data + "_" + str(levels[i]) + "hPa.npy"
       np.save( filename, train_set[ :,:,:,i ] )

       filename = "input_data/test/" + var + "_era5_" + args.data + "_" + str(levels[i]) + "hPa.npy"
       np.save( filename, test_set[ :,:,:,i ] )
   print("              * features data written to hard disk")
   print(" ")
print(" ")
