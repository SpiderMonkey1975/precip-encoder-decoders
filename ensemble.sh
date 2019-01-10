#!/bin/bash

num_epoch=50
batch=8
num_filters=(32)
num_hidden_nodes=(32)
learn_rates=("0.002" "0.0002" "0.00002")

input_file="jobscripts/run_era5_classifier"

cnt=0
for num_nodes in "${num_hidden_nodes[@]}"
do
   for f in "${num_filters[@]}"
   do
       for lrate in "${learn_rates[@]}"
       do
           output_file="jobscript_${cnt}"
           if [ -e $output_file ]
           then
              rm $output_file
           fi
    
           while read LINE; do
                 if echo $LINE | grep -q "EPOCHS="; then
                    LINE="EPOCHS=$num_epoch" 
                 fi
                 if echo $LINE | grep -q "BATCH_SIZE="; then
                    LINE="BATCH_SIZE=$batch" 
                 fi
                 if echo $LINE | grep -q "LEARN_RATE="; then
                    LINE="LEARN_RATE=$lrate" 
                 fi
                 if echo $LINE | grep -q "L2_REG="; then
                    LINE="L2_REG=$l2" 
                 fi
                 if echo $LINE | grep -q "MAX_FILTERS="; then
                    LINE="MAX_FILTERS=$f" 
                 fi
                 if echo $LINE | grep -q "MAX_HIDDEN_NODES="; then
                    LINE="MAX_HIDDEN_NODES=$num_nodes" 
                 fi

                 echo $LINE >> $output_file
           done < $input_file

           sbatch $output_file
           cnt=$((cnt+1))
       done
   done
done

