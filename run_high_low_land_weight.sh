#! /usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2GB

echo $1 $2 $3 $4

python3 ./run_tests.py --tolerance_higher $1 --tolerance_lower $2 --land_value $3 --neighbour_to_land_weight $4