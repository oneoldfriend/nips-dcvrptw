#!/bin/bash

running_times=$1  # parallelism degree
instances=( $(ls -1 ./instances/test_set/*) )
num_instances=${#instances[@]}

for i in $(seq 0 1) # $num_instances)
do
    echo ${instances[$i]}
    for k in $(seq 0 $running_times)
    do
        python controller.py --instance ${instances[$i]} --epoch_tlim 10 --static -- python solver.py &
    done
    wait
#    for j in $(seq 0 $running_times)
#    do
#        echo ${instances[$i]}&
#        python controller.py --instance ${instances[$i]} --epoch_tlim 60 -- python solver.py
#    done
#    wait
done

