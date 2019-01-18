#!/bin/bash

variables=('z' 't' 'rh')
num_filters=(16 32)
num_hidden_nodes=(16 32 64)
num_unet_layers=(1 2)

input_file="jobscripts/run_era5_classifier_pascal"

cnt=0
for num_nodes in "${num_hidden_nodes[@]}"
do
   for f in "${num_filters[@]}"
   do
       for num_nodes in "${num_hidden_nodes[@]}"
       do
           for var in "${variables[@]}"
           do
               for layer in "${num_unet_layers[@]}"
               do
                   output_file="jobscript_${cnt}"
                   if [ -e $output_file ]
                   then
                      rm $output_file
                   fi
    
                   while read LINE; do
                         if echo $LINE | grep -q "VARIABLE="; then
                            LINE="VARIABLE=$var" 
                         fi
                         if echo $LINE | grep -q "NUM_FILTERS="; then
                            LINE="NUM_FILTERS=$f" 
                         fi
                         if echo $LINE | grep -q "NUM_HIDDEN_NODES="; then
                            LINE="NUM_HIDDEN_NODES=$num_nodes" 
                         fi
                         if echo $LINE | grep -q "NUM_UNET_LAYERS="; then
                            LINE="NUM_UNET_LAYERS=$layer" 
                         fi

                         echo $LINE >> $output_file
                   done < $input_file

                   sbatch $output_file
                   cnt=$((cnt+1))
               done
           done
       done
   done
done

