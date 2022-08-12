#!/bin/bash

instances=( $(ls -1 ./instances/*) )
num_instances=`expr ${#instances[@]} - 1`

for i in $(seq 0 6) # $num_instances)
do
    ./genvrp ${instances[$i]} 60 -seed 1 -veh -1 -useWallClockTime 1&
done

