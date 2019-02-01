#!/bin/bash

hidden_node_counts=('25' '50' '100')
dropout_fractions=('0.2' '0.3')
learn_rates=('0.0001' '0.00001')

input_file="jobscripts/run_era5_classifier"

cnt=0
for nodes in "${hidden_node_counts[@]}"
do
    for fraction in "${dropout_fractions[@]}"
        do
        for lrate in "${learn_rates[@]}"
        do
            output_file="jobscript_${cnt}"
            if [ -e $output_file ]
            then
               rm $output_file
            fi
   
            while read LINE; do
                  if echo $LINE | grep -q "DROPOUT_FRACTION="; then
                     LINE="DROPOUT_FRACTION=$fraction" 
                  fi
                  if echo $LINE | grep -q "LEARN_RATE="; then
                     LINE="LEARN_RATE=$lrate" 
                  fi
                  if echo $LINE | grep -q "NUM_HIDDEN_NODES="; then
                     LINE="NUM_HIDDEN_NODES=$nodes" 
                  fi
                  echo $LINE >> $output_file
            done < $input_file

            sbatch $output_file
            cnt=$((cnt+1))
        done
    done
done

