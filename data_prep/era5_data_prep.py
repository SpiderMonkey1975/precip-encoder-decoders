import argparse, h5py
import numpy as np

print(" ")
print(" ")
print("*===================================================================================*")
print("*                           ERA5 INPUT DATA PREPARATION                             *")
print("*===================================================================================*")
print(" ")

##
## Important input data dimensions
##

image_width = 240
image_height = 360

num_levels = 3
levels = ['500hPa', '800hPa', '1000hPa']

num_training_images = 30000
num_test_images = 648

variables = ['z','t','rh']
varnames = ['atmospheric pressure','atmospheric temperature','relative humidity']

##
## Parse any user given commandline arguments 
##

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_bins', type=int, default=4, help="number of rainfall classification bins")
parser.add_argument('-f', '--hdf5_format', type=int, default=0, help="use HDF5 output format if set to 1")
parser.add_argument('-s', '--separate_levels', type=int, default=0, help="if set to 1, output each pressure level of the feature data in separate files/datasets")
args = parser.parse_args()

if args.num_bins<2:
   args.num_bins = 2
elif args.num_bins>6:
   args.num_bins = 6

if args.separate_levels != 1:
   args.separate_levels = 0

if args.hdf5_format != 1:
   args.hdf5_format = 0
else:
   if args.separate_levels != 1:
      filename = "../input_data/era5_au_input_" + str(args.num_bins) + "bins.h5"
   else:
      filename = "../input_data/era5_au_input_" + str(args.num_bins) + "bins_separate_levels.h5"
   fid = h5py.File( filename, 'w' )
   dset = fid.create_dataset( "num_bins", (1,), dtype='i4' )
   dset[0] = args.num_bins

print("      splitting Australia-specific ERA5 data into %d bins" % (args.num_bins))
if args.hdf5_format == 1:
   print("      HDF5 format requested for output datafiles")
print(" ")

##
## Read in total precipitation data
##

precip_data = np.load(  "/scratch/director2107/ERA5_Data/ERA5_Native/tp_era5_au.npy" )
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
## Generate an one-hot encoding for the precipatation data 
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

if args.hdf5_format == 1:
   dset = fid.create_dataset( "bins_values", (args.num_bins,) )
   dset[:args.num_bins ] = bin_vals
#filename = "input_data/rainfall_maxvals_" + str(args.num_bins) + "bins.npy"
#np.save( filename, bin_vals )

for i in range(args.num_bins):
    for j in range(records_per_bin):
        idx = indicies[ i*records_per_bin + j ]
        one_hot_encoding[idx,i] = 1
print("           -> one-hot encoding completed")

##
## Generate index sets for the training and test data sets 
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

if args.hdf5_format == 1:
   train_grp = fid.create_group( 'training' )
   dset = train_grp.create_dataset( "labels", (num_training_images,args.num_bins), compression='gzip' )
   dset[ :,: ] = one_hot_encoding[ training_indicies,: ]

   test_grp = fid.create_group( 'test' )
   dset = test_grp.create_dataset( "labels", (num_test_images,args.num_bins), compression='gzip' )
   dset[ :,: ] = one_hot_encoding[ test_indicies,: ]
else:
   varname = "au_labels_" + str(args.num_bins) + "bins.npy"       
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

if args.hdf5_format == 1 and if args.separate_levels == 1:
   train_grps = []
   test_grps = []
   for n in range(num_levels):
       train_grps.append( train_grp.create_group( levels[n] ) )
       test_grps.append( test_grp.create_group( levels[n] ) )

cnt = 0
for var in variables:
   print("           -> %s" % (varnames[cnt]))
   filename = "/scratch/director2107/ERA5_Data/ERA5_Native/" + var + "_era5_au.npy"
   features_data = np.load( filename )
   print("              * data read from hard disk")

   features_data = np.moveaxis(features_data, 1, 3)
   print("              * axis swap performed")

   for i in range( num_levels ):
       for n in range( features_data.shape[0] ):
           mean_val = np.mean( features_data[n,:,:,i] )
           std_dev = np.std( features_data[n,:,:,i] )
           features_data[n,:,:,i] = (features_data[n,:,:,i] - mean_val) / std_dev
   print("              * normalization completed")
   cnt = cnt + 1

##
## Output processed features data to hard disk
##

   if args.separate_levels == 0:
      if args.hdf5_format == 1:
         dset = train_grp.create_dataset( var, (num_training_images,image_width,image_height,num_levels,), compression='gzip' )
         dset[ :,:,:,: ] = features_data[ training_indicies,:,:,: ]

         dset = test_grp.create_dataset( var, (num_test_images,image_width,image_height,num_levels,), compression='gzip' )
         dset[ :,:,:,: ] = features_data[ test_indicies,:,:,: ]
      else:
         varname = var + "_era5_au_" + str(args.num_bins) + "bins.npy"
         filename = "../input_data/training/"+ varname
         np.save( filename, features_data[ training_indicies,:,:,: ] )

         filename = "../input_data/test/"+ varname
         np.save( filename, features_data[ test_indicies,:,:,: ] )
   else:
      if args.hdf5_format == 1:
         for n in range(num_levels):
             dset = train_grps[n].create_dataset( var, (num_training_images,image_width,image_height,), compression='gzip' )
             dset[ :,:,: ] = features_data[ training_indicies,:,:,n ]
             dset = test_grps[n].create_dataset( var, (num_test_images,image_width,image_height,), compression='gzip' )
             dset[ :,:,: ] = features_data[ test_indicies,:,:,n ]
      else:
         for n in range(num_levels):
             filename = "../input_data/training/" + levels[n] + "/" + var + "_era5_au_" + str(args.num_bins) + "bins.npy"
             np.save( filename, features_data[ training_indicies,:,:,n ] )
             filename = "../input_data/test/" + levels[n] + "/" + var + "_era5_au_" + str(args.num_bins) + "bins.npy"
             np.save( filename, features_data[ test_indicies,:,:,n ] )
   print("              * features data written to hard disk")
   print(" ")
print(" ")

