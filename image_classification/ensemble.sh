#!/bin/bash

variables=('rh' 'z' 't')
levels=('500' '800' '1000')
num_bins=4

cnt=0
input_file="../jobscripts/run_era5_classifier"
for var in "${variables[@]}"
do
    for lev in "${levels[@]}"
    do
        output_file="jobscript_${cnt}"
        if [ -e $output_file ]
        then
           rm $output_file
        fi
   
        while read LINE; do
              if echo $LINE | grep -q "NUM_BINS="; then
                 LINE="NUM_BINS=$num_bins" 
              fi
              if echo $LINE | grep -q "PRESSURE_LEVEL="; then
                 LINE="PRESSURE_LEVEL=$lev" 
              fi
              if echo $LINE | grep -q "VARIABLE="; then
                 LINE="VARIABLE=$var" 
              fi
              echo $LINE >> $output_file
        done < $input_file

        sbatch $output_file
        cnt=$((cnt+1))
    done
done

input_file="../jobscripts/run_era5_classifier2"
for var in "${variables[@]}"
do
        output_file="jobscript_${cnt}"
        if [ -e $output_file ]
        then
           rm $output_file
        fi

        while read LINE; do
              if echo $LINE | grep -q "NUM_BINS="; then
                 LINE="NUM_BINS=$num_bins" 
              fi
              if echo $LINE | grep -q "VARIABLE="; then
                 LINE="VARIABLE=$var"
              fi
              echo $LINE >> $output_file
        done < $input_file

        sbatch $output_file
        cnt=$((cnt+1))
done

input_file="../jobscripts/run_era5_classifier3"
for lev in "${levels[@]}"
do
        output_file="jobscript_${cnt}"
        if [ -e $output_file ]
        then
           rm $output_file
        fi

        while read LINE; do
              if echo $LINE | grep -q "NUM_BINS="; then
                 LINE="NUM_BINS=$num_bins" 
              fi
              if echo $LINE | grep -q "PRESSURE_LEVEL="; then
                 LINE="PRESSURE_LEVEL=$lev"
              fi
              echo $LINE >> $output_file
        done < $input_file

        sbatch $output_file
        cnt=$((cnt+1))
done

