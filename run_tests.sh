#! /usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=1GB

module load conda
source activate pythonenv

# Run test high_tol
for h_tol in 0.6 0.7 0.8 # higher tolerance
do 
    sbatch ./run_high_low_land_weight.sh $h_tol 0.3 0.5 1 &
done

# Run test low_tol
for l_tol in 0.2 0.3 0.4 # lower tolerance
do 
    sbatch ./run_high_low_land_weight.sh 0.6 $l_tol 0.5 1 &
done

# Run test land_value
for val in 0.4 0.5 0.6 # Land value
do 
    sbatch ./run_high_low_land_weight.sh 0.6 0.3 $val 1 &
done

# Run test land_weight
for land_weight in 0.8 0.9 1.0 # Land weight
do 
    sbatch ./run_high_low_land_weight.sh 0.6 0.3 0.5 $land_weight &
done

# for h_tol in 0.6 0.7 0.8 # higher tolerance
# do
#     for l_tol in 0.2 0.3 0.4 # lower tolerance
#     do 
#         for val in 0.4 0.5 0.6 # Land value
#         do
#             for land_weight in 0.8 0.9 1.0 # Land weight
#             do 
#                 sbatch ./run_high_low_land_weight.sh $h_tol $l_tol $val $land_weight &
#                 #./run_high_low_land_weight.sh $h_tol $l_tol $val $land_weight
#             done 
#         done 
#     done 
# done