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
print("      [1/8] precipitation data read")

##
## Generate labels array(s)
##

num_records = precip_data.shape[0]
one_hot_encoding = np.zeros((num_records,4), dtype=int)

idx = 0
for n in range( num_records ):
    val = np.amax(precip_data[n,:,:])

    # Class 1 - extreme rainfall  
    if val > limits[2]:
       one_hot_encoding[idx,0] = 1
       idx = idx + 1

    # Class 2 - heavy rainfall 
    if val > limits[1] and val <= limits[2]:
       one_hot_encoding[idx,1] = 1
       idx = idx + 1

    # Class 3 - rainfall  
    if val > limits[0] and val <= limits[1]:
       one_hot_encoding[idx,2] = 1
       idx = idx + 1

    # Class 4 - insignificant or no rainfall 
    if val <= limits[0]:
       one_hot_encoding[idx,3] = 1
       idx = idx + 1
print("      [2/8] one-hot encoding completed")

##
## Select the same number of rainfall class instances for the image recognition 
##

num_records = np.min(np.sum(one_hot_encoding, axis=0)) 
num_test_records = int( 0.1 * num_records )
num_train_records = num_records - num_test_records 

cnt = np.zeros( 4, dtype=int )
indicies = np.empty( (num_records,4), dtype=int )

for i in range( 4 ):
    for n in range( one_hot_encoding.shape[0] ):
        if one_hot_encoding[n,i] == 1 and cnt[i] < num_records:
           indicies[cnt[i],i] = n
           cnt[i] = cnt[i] + 1

##
## Create the training / test dataset split 
##

train_indicies = indicies[ :num_train_records,: ]
train_indicies.flatten()
np.random.shuffle( train_indicies )

test_indicies = indicies[ num_train_records:num_records,: ]
test_indicies.flatten()
np.random.shuffle( test_indicies )
print("      [3/8] indicies selected")
 
train_one_hot_encoding = one_hot_encoding[ train_indicies,: ]
test_one_hot_encoding = one_hot_encoding[ test_indicies,: ]

##
## Output label data to hard disk
##

filename = "input_data/training/" + args.data + "_one_hot_encoding.npy"
np.save( filename, train_one_hot_encoding )
filename = "input_data/test/" + args.data + "_one_hot_encoding.npy"
np.save( filename, test_one_hot_encoding )
print("      [4/8] label data written to hard disk")

##
## Read in and process features data 
##

filename = "/scratch/director2107/ERA5_Data/ERA5_Native/z_era5_" + args.data + ".npy"
z_data = np.load( filename )
print("      [5/8] atmospheric pressure read")

z_data = np.moveaxis(z_data, 1, 3)
print("      [6/8] axis swap performed")

for n in range( z_data.shape[0] ):
    mean_val = np.mean( z_data[n,:,:,:] )
    std_dev = np.std( z_data[n,:,:,:] )
    z_data[n,:,:,:] = (z_data[n,:,:,:] - mean_val) / std_dev
print("      [7/8] normalization completed")

##
## Output processed features data to hard disk
##

z_train_data = z_data[train_indicies,:,:,:]
filename = "input_data/training/z_era5_" + args.data + "_CLASSIFICATION.npy"
np.save( filename, z_train_data )
z_test_data = z_data[test_indicies,:,:,:]
filename = "input_data/test/z_era5_" + args.data + "_CLASSIFICATION.npy"
np.save( filename, z_test_data )
print("      [8/8] features data written to hard disk")
print(" ")
