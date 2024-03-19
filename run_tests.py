from model.schelling import Schelling
from model.agent import Agent

import sys
import os
import argparse



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Schelling model')
    parser.add_argument('--n_agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--n_iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--grid_size', type=int, default=50, help='Size of the grid')
    parser.add_argument('--occupied_ratio', type=int, default=90, help='Occupied ratio')
    parser.add_argument('--tolerance_higher', type=float, default=0.6, help='Tolerance higher')
    parser.add_argument('--tolerance_lower', type=float, default=0.3, help='Tolerance lower')
    parser.add_argument('--land_start', type=tuple, default=(20, 20), help='Land start')
    parser.add_argument('--land_end', type=tuple, default=(30, 30), help='Land end')
    parser.add_argument('--land_value', type=float, default=0.5, help='Land value')
    parser.add_argument('--neighbour_satisfaction', type=float, default=0.5, help='Neighbour satisfaction')
    parser.add_argument('--neighbour_to_land_weight', type=float, default=1.0, help='Neighbour to land weight')
    parser.add_argument('--test', type=str, default=None, help='Name of folder to save test results to')
    args = parser.parse_args()

    # Run the model
    model = Schelling((args.grid_size, args.grid_size), 
                        args.occupied_ratio, 
                        args.n_agents, 
                        args.tolerance_higher, 
                        args.tolerance_lower,
                        args.land_start,
                        args.land_end,
                        args.land_value,
                        args.neighbour_satisfaction,
                        args.neighbour_to_land_weight,
                        args.test
                    )
    
    model.run_simulation(args.n_iterations)
    model._plot_satisfaction_history()
    

