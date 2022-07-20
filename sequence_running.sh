#!/bin/bash

running_times=$1  # parallelism degree
rm -rf results/*.*
rm -rf results/static/*
rm -rf results/dynamic/*
instances=( $(ls -1 ./instances/*) )
num_instances=`expr ${#instances[@]} - 1`

for i in $(seq 0 $num_instances)
do
    echo ${instances[$i]}
    for k in $(seq 1 $running_times)
    do
        python controller.py --instance ${instances[$i]} --epoch_tlim 7 --static -- python solver.py &
    done
    wait
    for j in $(seq 1 $running_times)
    do
        python controller.py --instance ${instances[$i]} --epoch_tlim 5 -- python solver.py &
    done
    wait
done
python -c "import tools as utils; utils.results_process(\"results/obj_results_raw.txt\")"

