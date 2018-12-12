import argparse
import numpy as np

print(" ")
print(" ")
print("*===========================================================================================================*")
print("*                                   ERA5 Z-POTENTIAL DATASET AXIS SWAP                                      *")
print("*===========================================================================================================*")

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='native', help="datset type: native, au")
args = parser.parse_args()

infile = "datasets/z_era5_" + args.data + ".npy"
outfile = "datasets/z_era5_" + args.data + "_NWHC.npy"

## 
## Load the raw features input data from disk
##

features = np.load( infile )

print( "Initial axis layout: ", features.shape )

##
## Perform axis swap 
##

new_features = np.moveaxis(features, 1, 3)

print( "New axis layout: ", new_features.shape )

##
## Write modified dataset to hard disk
##

np.save( outfile, new_features )
