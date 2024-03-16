#! /usr/bin/env bash


for h_tol in 0.6 0.7 0.8 # higher tolerance
do
    for l_tol in 0.2 0.3 0.4 # lower tolerance
    do 
        for val in 0.4 0.5 0.6 # Land value
        do
            for land_weight in 0.8 0.9 1.0 # Land weight
            do 
                sbatch ./run_high_low_land_weight.sh $h_tol $l_tol $val $land_weight &
                #./run_high_low_land_weight.sh $h_tol $l_tol $val $land_weight
            done 
        done 
    done 
done