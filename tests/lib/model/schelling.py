# This file contains the Schelling class, used to simulate Schelling's model of segregation.
import matplotlib
matplotlib.use("TkAgg")

from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import random
import os



from agent import Agent, Person


class Schelling:
    def __init__(self, grid_size: tuple, vacant_ratio: int, num_groups: int, tolerance: float) -> None:
        """
        Initialise the Schelling's model simulation.

        Parameters:
            grid_size (tuple): The size of the grid (m,n).
            vacant_ratio (float): The ratio of vacant cells in the grid.
            num_groups (int): The number of groups of agents in the grid.
            tolerance (float): The tolerance threshold of agents. A number between 0 and 1, representing the minimum proportion of same group neighbours for an agent to be satisfied.
        """
        self._grid_size: tuple = grid_size
        self._vacant_ratio: int = vacant_ratio
        self._num_groups: int = num_groups
        self._tolerance: float = tolerance
        self.grid: np.ndarray = None
        self._empty_cells: list = None
        self._next = [i+1 for i in range(self._grid_size[0]-1)]
        self._next.append(0)
        self._previous = [self._grid_size[0]-1]
        self._previous.extend([i for i in range(self._grid_size[0]-1)])

        # Logging metrics
        self._iterations_to_equilibrium: int = 0
        self._num_agents_moved: int = 0

    def __repr__(self) -> str:
        return f"Schelling(grid_size={self._grid_size}, vacant_ratio={self._vacant_ratio}, num_groups={self._num_groups}, tolerance={self._tolerance})"

    def _initialise_grid(self) -> None:
        """
        Initialise the grid with agents and vacant cells.
        """

        # Create empty Grid
        self.grid = np.zeros(self._grid_size, dtype=object)
        assert self.grid.shape == self._grid_size

        self._empty = []
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                r = random.randint(0, 100)
                if r <= self._vacant_ratio:
                    self.grid[i][j] = Person()
                else:
                    self.grid[i][j] = 0
                    self._empty.append([i,j])

        self._visualise_grid("results/initial_schelling.png")

    
    def run_simulation(self, max_iterations: int | None) -> None:
        """
        Run the Schelling's model simulation.

        Parameters:
            max_iterations (int): The maximum number of iterations to run the simulation for. If None, the simulation runs until equilibrium is reached.
        """
        self._initialise_grid()

        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            self._iterations_to_equilibrium += 1
            self._update_grid()
            self._visualise_animated_grid()

        if max_iterations is not None:
            ani = animation.FuncAnimation(fig, update, frames=max_iterations, repeat=False)
        else:
            print(f"Running until all agents are satisfied or until maximum limit is reached...")
            print(f"WARNING: THIS MAY TAKE A LONG TIME.")
            max_limit = 1e7
            ani = animation.FuncAnimation(fig, update, frames=max_limit, repeat=False)

        ani.save("results/schelling_animation.gif", writer="pillow", fps=2)  # Adjust fps as needed
        plt.show()
        self._visualise_grid("results/schelling.png")

    def _visualise_animated_grid(self) -> None:
        """
        Visualise the grid using plt object.
        """
        # grid to plot of results
        plotGrid = np.zeros((self._grid_size[0], self._grid_size[1]))
        # black, white and gray
        colours = ['#49beaa','#ef767a','#456990']
        cmap = {0: '#ef767a', 1:'#456990', 2:'#49beaa'}
        labels = {0: 'empty', 1: 'Group A', 2: 'Group B'}
        patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
        tmp = mpl.colors.ListedColormap(colours)

        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                if self.grid[i][j] != 0:
                    plotGrid[i][j] = self.grid[i][j]._type

        plt.imshow(plotGrid, cmap=tmp)
        plt.legend(handles=patches, shadow=True, facecolor='#6A6175',
                bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
        plt.title(f'Iteration: {self._iterations_to_equilibrium}')
        plt.draw()  # Change this line
    


    def _update_grid(self) -> None:
        """
        Update the grid based on the Schelling's model rules.
        """
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                if self.grid[i][j] != 0:
                    if self._is_agent_satisfied(i,j) == False:
                        _ = random.choice(self._empty)
                        self._empty.remove(_)
                        self.grid[_[0]][_[1]] = Person()
                        self.grid[_[0]][_[1]]._type = self.grid[i][j]._type
                        self.grid[i][j] = 0
                        self._empty.append([i,j])
                        self._num_agents_moved += 1


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
        for x in range(max(-1, -i), min(2, self._grid_size[0] - i)):
            for y in range(max(-1, -j), min(2, self._grid_size[1] - j)):
                if x == 0 and y == 0:
                    continue
                if self.grid[i + x, j + y] == self.grid[i, j]:
                    num_same_neighbours += 1
        return num_same_neighbours
    

    def _is_agent_satisfied(self, x: int, y: int) -> bool:
        [all, the_same] = [0, 0]
        if self.grid[self._next[x]][y] != 0:
            all += 1
            if self.grid[self._next[x]][y]._type == self.grid[x][y]._type:
                the_same += 1
        if self.grid[x][self._next[y]] != 0:
            all += 1
            if self.grid[x][self._next[y]]._type == self.grid[x][y]._type:
                the_same += 1
        if self.grid[x][self._previous[y]] != 0:
            all += 1
            if self.grid[x][self._previous[y]]._type == self.grid[x][y]._type:
                the_same += 1
        if self.grid[self._previous[x]][y] != 0:
            all += 1
            if self.grid[self._previous[x]][y]._type == self.grid[x][y]._type:
                the_same += 1
        if self.grid[self._previous[x]][self._previous[y]] != 0:
            all += 1
            if self.grid[self._previous[x]][self._previous[y]]._type == self.grid[x][y]._type:
                the_same += 1
        if self.grid[self._next[x]][self._previous[y]] != 0:
            all += 1
            if self.grid[self._next[x]][self._previous[y]]._type == self.grid[x][y]._type:
                the_same += 1
        if self.grid[self._next[x]][self._next[y]] != 0:
            all += 1
            if self.grid[self._next[x]][self._next[y]]._type == self.grid[x][y]._type:
                the_same += 1
        if self.grid[self._previous[x]][self._next[y]] != 0:
            all += 1
            if self.grid[self._previous[x]][self._next[y]]._type == self.grid[x][y]._type:
                the_same += 1
        if all != 0:
            return False if the_same/all < self._tolerance else True
        return False
        
    def _visualise_grid(self, filename: str | None) -> None:
        """
        Visualise the grid.

        Parameters:
            filename (str): The filename to save the visualisation.
        """
        # grid to plot of results
        plotGrid = np.zeros((self._grid_size[0], self._grid_size[1]))
        # black, white and gray
        colours = ['#49beaa','#ef767a','#456990']
        cmap = {0: '#ef767a', 1:'#456990', 2:'#49beaa'}
        labels = {0:'empty', 1:'Group A', 2:'Group B', }
        patches = [mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
        tmp = mpl.colors.ListedColormap(colours)

        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                if self.grid[i][j] != 0:
                    plotGrid[i][j] = self.grid[i][j]._type

        plt.imshow(plotGrid, cmap=tmp)
        plt.legend(handles=patches,shadow=True, facecolor='#6A6175',
                    bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(filename)
        plt.close()

    def create_metrics(self) -> float:
        """
        Create metrics for the Schelling's model simulation.
        """
        # Find average satisfaction of agents
        # Find number of iterations to reach equilibrium
        # Find number of agents that moved
        print(f"Number of iterations to reach equilibrium: ", self._iterations_to_equilibrium)
        print(f"Number of agents that moved: ", self._num_agents_moved)
        

if __name__ == "__main__":
    schelling = Schelling((50,50), 90, 2, 0.6)
    schelling.run_simulation(max_iterations=1000)
    schelling.create_metrics()