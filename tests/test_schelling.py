import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from lib.model.schelling import Schelling


class TestSchelling(unittest.TestCase):
    def setUp(self):
        self.grid_size = (10, 10)
        self.vacant_ratio = 0.3
        self.num_groups = 2
        self.tolerance = 0.3
        self.max_iterations = 100
        self.schelling = Schelling(self.grid_size, self.vacant_ratio, self.num_groups, self.tolerance)

    def test_initialise_grid(self):
        self.schelling._initialise_grid()
        self.assertEqual(self.schelling.grid.shape, self.grid_size)

    def test_run_simulation(self):
        self.schelling.run_simulation(self.max_iterations)
        self.assertEqual(self.schelling.grid.shape, self.grid_size)


    def test_grid_values(self):
        self.schelling._initialise_grid()
        self.assertTrue(np.all(np.isin(self.schelling.grid, [0, 1, 2])))

        # Test the number of agents in each group
        num_agents = np.sum(self.schelling.grid != 0)
        num_agents_per_group = num_agents // self.num_groups
        for group_id in range(1, self.num_groups + 1):
            self.assertEqual(np.sum(self.schelling.grid == group_id), num_agents_per_group)

    def test_metrics(self):
        self.schelling.run_simulation(self.max_iterations)
        self.assertTrue(self.schelling._satisfaction >= 0)
        self.assertTrue(self.schelling._iterations_to_equilibrium >= 0)
        self.assertTrue(self.schelling._num_agents_moved >= 0)

    
if __name__ == '__main__':
    unittest.main()