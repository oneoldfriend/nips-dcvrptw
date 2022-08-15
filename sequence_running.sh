#!/bin/bash

parallel_level=$1 # parallelism degree
rm -rf results/*.*
rm -rf results/static/*
rm -rf results/dynamic/*
instances=($(ls -1 ./instances/*))
num_instances=$(expr ${#instances[@]} - 1)
cur_paralel=1

for i in $(seq 0 $num_instances); do
  echo ${instances[$i]}
  echo $cur_paralel
  #    for k in $(seq 1 $parallel_level)
  #    do
  #        python controller.py --instance ${instances[$i]} --epoch_tlim 7 --static -- python solver.py &
  #    done
  #    wait
  python controller.py --instance ${instances[$i]} --epoch_tlim 5 -- python solver.py --model gnn_64_2_mean_greedy_greedy_0.5_nr_99 &
  if [ "$cur_paralel" -gt "$parallel_level" ]; then
    cur_paralel=1
    wait
  else
    cur_paralel=$((cur_paralel + 1))
  fi
done
wait
python -c "import tools as utils; utils.results_process(\"results/obj_results_raw.txt\")"
