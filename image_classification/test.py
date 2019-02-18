import numpy as np
import sys

cnt = 0
num_samples = 0
levels = ["500","800","1000"]
#variables = ["z", "t", "rh"]
variables = ["rh"]

## 
## Load the label data fileand determine the real rainfall categories 
##

tmp = np.load( "../input_data/test/au_labels_6bins.npy" )
actual = np.argmax( tmp,axis=1 )

##
## Determine the overall match accuracy using 3 variables per level output
##

for lev in levels:
    filename = "predictions_3vars_" + lev + "hPa.npy"
    predicted = np.load( filename )

    for n in range( actual.shape[0] ):
        if actual[n] == predicted[n]:
           cnt = cnt + 1

    num_samples = num_samples + actual.shape[0]

print( "observed accuracy was %d percent" % (100*cnt/num_samples))

##
## Determine the overall match accuracy using 3 pressure levels per variable output
##

for var in variables:
    filename = "predictions_" + var + "all_levels.npy"
    predicted = np.load( filename )

    for n in range( actual.shape[0] ):
        if actual[n] == predicted[n]:
           cnt = cnt + 1
    num_samples = num_samples + actual.shape[0]

print( "observed accuracy was %d percent" % (100*cnt/num_samples))

##
## Determine the overall match accuracy using single variable per level output
##

for var in variables:
    for lev in levels:
        filename = "predictions_" + var + "_" + lev + "hPa.npy"
        predicted = np.load( filename )

        for n in range( actual.shape[0] ):
            if actual[n] == predicted[n]:
               cnt = cnt + 1
        num_samples = num_samples + actual.shape[0]

print( "observed accuracy was %d percent" % (100*cnt/num_samples))

