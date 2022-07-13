#!/bin/bash

running_times=$1  # parallelism degree
rm results.txt
instances=( $(ls -1 ./instances/test_set/*) )
num_instances=`expr ${#instances[@]} - 1`

for i in $(seq 0 1) # $num_instances)
do
    echo ${instances[$i]}
    for k in $(seq 1 $running_times)
    do
        python controller.py --instance ${instances[$i]} --epoch_tlim 300 --static -- python solver.py &
    done
    wait
    for j in $(seq 1 $running_times)
    do
        python controller.py --instance ${instances[$i]} --epoch_tlim 60 -- python solver.py &
    done
    wait
done

