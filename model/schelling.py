# This file contains the Schelling class, used to simulate Schelling's model of segregation.
import numpy as np
import matplotlib.pyplot as plt

class Schelling:
    def __init__(self, grid_size: tuple, vacant_ratio: float, num_groups: int, tolerance: float) -> None:
        """
        Initialise the Schelling's model simulation.

        Parameters:
            grid_size (tuple): The size of the grid (m,n).
            vacant_ratio (float): The ratio of vacant cells in the grid.
            num_groups (int): The number of groups of agents in the grid.
            tolerance (float): The tolerance threshold of agents.
        """
        self._grid_size: tuple = grid_size
        self._vacant_ratio: float = vacant_ratio
        self._num_groups: int = num_groups
        self._tolerance: float = tolerance
        self.grid: np.ndarray = None


        # Logging metrics
        self._satisfaction: float = 0.0
        self._iterations_to_equilibrium: int = 0
        self._num_agents_moved: int = 0

        self._initialise_grid()

    def _initialise_grid(self) -> None:
        """
        Initialise the grid with agents and vacant cells.
        """
        num_cells = self._grid_size[0] * self._grid_size[1]
        num_agents = int((1 - self._vacant_ratio) * num_cells)
        num_agents_per_group = num_agents // self._num_groups # Assume equal number of agents per group
        num_vacant_cells = num_cells - num_agents

        agent_types = []
        for group_id in range(self._num_groups):
            num_agents_in_group = num_agents_per_group
            if group_id == self._num_groups - 1:  # Adjust for rounding error
                num_agents_in_group += num_agents % self._num_groups
            agent_types.extend([group_id + 1] * num_agents_in_group)

        agent_types.extend([0] * num_vacant_cells)
        np.random.shuffle(agent_types)

        self.grid = np.array(agent_types).reshape(self._grid_size)

    
    def run_simulation(self, max_iterations: int | None) -> None:
        """
        Run the Schelling's model simulation.

        Parameters:
            max_iterations (int): The maximum number of iterations to run the simulation for.
        """
        if max_iterations is None:
            max_iterations = 1000

        for _ in range(max_iterations):
            self._update_grid()
        
        self._visualise_grid()


    def _update_grid(self) -> None:
        """
        Update the grid based on the Schelling's model rules.
        """
        # Create copy of the grid to update
        new_grid: np.ndarray = np.copy(self.grid)
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):

                # Skip vacant cells
                if self.grid[i, j] == 0:
                    continue

                # Check if agent is satisfied
                num_same_neighbours: int = self._count_same_neighbours(i, j)
                if num_same_neighbours / 8 < self._tolerance:
                    # Agent is not satisfied, move to a vacant cell
                    vacant_cells: np.ndarray = np.argwhere(self.grid == 0)
                    if len(vacant_cells) == 0:
                        continue

                    # Move to a random vacant cell
                    new_cell = vacant_cells[np.random.randint(len(vacant_cells))]

                    new_grid[new_cell[0], new_cell[1]] = self.grid[i, j]

                    # Set the old cell to vacant
                    new_grid[i, j] = 0

                    # Log metrics
                    self._num_agents_moved += 1

        # Update the grid
        self.grid = new_grid


    def _count_same_neighbours(self, i: int, j: int) -> int:
        """
        Count the number of same neighbours of the agent at (i, j).

        Parameters:
            i (int): The row index of the agent.
            j (int): The column index of the agent.

        Returns:
            int: The number of same neighbours of the agent at (i, j).
        """
        num_same_neighbours: int = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                if x == 0 and y == 0:
                    continue
                if i + x < 0 or i + x >= self._grid_size[0] or j + y < 0 or j + y >= self._grid_size[1]:
                    continue
                if self.grid[i + x, j + y] == self.grid[i, j]:
                    num_same_neighbours += 1
        return num_same_neighbours


    def _visualise_grid(self, filename: str | None) -> None:
        """
        Visualise the grid.

        Parameters:
            filename (str): The filename to save the visualisation to. If None, the visualisation is displayed.
        """
        plt.imshow(self.grid, cmap="viridis", interpolation="nearest")
        plt.savefig("results/schelling.png")
    

    def create_metrics(self) -> float:
        """
        Create metrics for the Schelling's model simulation.
        """
        # Find average satisfaction of agents
        # Find number of iterations to reach equilibrium
        # Find number of agents that moved
        print(f"Average satisfaction of agents: ", self._satisfaction)
        print(f"Number of iterations to reach equilibrium: ", self._iterations_to_equilibrium)
        print(f"Number of agents that moved: ", self._num_agents_moved)

        

if __name__ == "__main__":
    schelling = Schelling((50,50), 0.3, 2, 0.1)
    schelling.run_simulation(max_iterations=10000)
    schelling.create_metrics()