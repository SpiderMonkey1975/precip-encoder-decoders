import numpy as np

print(" ")
print(" ")
print("*===========================================================================================================*")
print("*                                    ERA5 FEATURES DATASET AXIS SWAP                                        *")
print("*===========================================================================================================*")

## 
## Load the raw features input data from disk
##

features = np.load("datasets/z_era5_native.npy")

print( "Initial axis layout: ", features.shape )

##
## Perform axis swap 
##

new_features = np.moveaxis(features, 1, 3)

print( "New axis layout: ", new_features.shape )

##
## Write modified dataset to hard disk
##

np.save( "datasets/z_era5_native_NWHC.npy", new_features )
