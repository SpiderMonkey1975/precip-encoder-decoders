#!/bin/bash

levels=('800')
num_bins=('4' '6')

input_file="jobscripts/run_era5_classifier_pascal"

cnt=0
for lev in "${levels[@]}"
do
    for bin in "${num_bins[@]}"
    do
            output_file="jobscript_${cnt}"
            if [ -e $output_file ]
            then
               rm $output_file
            fi
   
            while read LINE; do
                  if echo $LINE | grep -q "PRESSURE_LEVEL="; then
                     LINE="PRESSURE_LEVEL=$lev" 
                  fi
                  if echo $LINE | grep -q "NUM_BINS="; then
                     LINE="NUM_BINS=$bin" 
                  fi
                  echo $LINE >> $output_file
            done < $input_file

            sbatch $output_file
            cnt=$((cnt+1))
   done
done

