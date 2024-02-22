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

class GridPoint:
    def __init__(self, agent=None, land_value=0.0):
        self.agent = agent
        self.land_value = land_value


class Schelling:
    def __init__(self, grid_size: tuple, vacant_ratio: int, num_groups: int, tolerance_higher: float, tolerance_lower: float, valuable_area_start: tuple, valuable_area_end: tuple, land_value: float) -> None:
        """
        Initialise the Schelling's model simulation.

        Parameters:
            grid_size (tuple): The size of the grid (m,n).
            vacant_ratio (float): The ratio of vacant cells in the grid.
            num_groups (int): The number of groups of agents in the grid.
            tolerance_higher (float): The tolerance threshold of agents with greater tolerance. A number between 0 and 1, representing the minimum proportion of same group neighbours for an agent to be satisfied.
            tolerance_higher (float): The tolerance threshold of agents with less tolerance
            valuable_area_start (tuple): The start coordinates of the valuable area (start_x, start_y).
            valuable_area_end (tuple): The end coordinates of the valuable area (end_x, end_y).
            land_value (float): The value assigned to the land within the valuable area.
        """
        self._grid_size: tuple = grid_size
        self._vacant_ratio: int = vacant_ratio
        self._num_groups: int = num_groups
        self._tolerance_higher: float = tolerance_higher
        self._tolerance_lower: float = tolerance_lower
        self._valuable_area_start: tuple = valuable_area_start
        self._valuable_area_end: tuple = valuable_area_end
        self._land_value: float = land_value
        self.grid: np.ndarray = None
        self._empty_cells: list = None
        self._next = [i+1 for i in range(self._grid_size[0]-1)]
        self._next.append(0)
        self._previous = [self._grid_size[0]-1]
        self._previous.extend([i for i in range(self._grid_size[0]-1)])

        # Logging metrics
        self._iterations_to_equilibrium: int = 0
        self._num_agents_moved: int = 0

        self._current_directory = os.getcwd()
        
    def __repr__(self) -> str:
        return f"Schelling(grid_size={self._grid_size}, vacant_ratio={self._vacant_ratio}, num_groups={self._num_groups}, tolerance_higher={self._tolerance_higher}, tolerace_lower={self._tolerance_lower})"

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
                    self.grid[i][j] = GridPoint(agent=Person(self._tolerance_higher, self._tolerance_lower), land_value=0.0) 
                else:
                    self.grid[i][j] = GridPoint(agent=None, land_value=0.0) 
                    self._empty.append([i,j])

        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                if self.grid[i][j].agent is not None:
                    self.grid[i][j].agent._type = random.randint(1, self._num_groups)  # Assign random group

        start_x, start_y = self._valuable_area_start
        end_x, end_y = self._valuable_area_end
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                self.grid[i][j].land_value = 1.0  # Set higher land value within the valuable area

        self._visualise_grid(self._current_directory + os.sep + "model" + os.sep + "results" + os.sep + "initial_schelling.png")

    
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
            self._visualise_animated_grid(max_iterations)
            if self._iterations_to_equilibrium > max_iterations:
                fig.canvas.draw()
                ani.event_source.stop()
                ani.save(self._current_directory + os.sep + "model" + os.sep + "results" + os.sep + "schelling_animation.gif", writer="pillow", fps=10)
            

        if max_iterations is not None: 
            ani = animation.FuncAnimation(fig, update, frames=max_iterations, repeat=False)
        else:
            print(f"Running until all agents are satisfied or until maximum limit is reached...")
            print(f"WARNING: THIS MAY TAKE A LONG TIME.")
            max_limit = 1e7
            ani = animation.FuncAnimation(fig, update, frames=max_limit, repeat=False)
            ani.save(self._current_directory + os.sep + "model" + os.sep + "results" + os.sep + "schelling_animation.gif", writer="pillow", fps=10)
            
        # Adjust fps as needed
        plt.show()
        self._visualise_grid(self._current_directory + os.sep + "model" + os.sep + "results" + os.sep + "schelling.png")
    
    def _update_grid(self) -> None:
        """
        Update the grid based on the Schelling's model rules.
        """
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                current_cell = self.grid[i][j]
                if current_cell.agent is not None:
                    if self._is_agent_satisfied(i,j) == False:
                        empty_cell = random.choice(self._empty)
                        self._empty.remove(empty_cell)

                        new_agent = Person(self._tolerance_higher, self._tolerance_lower)
                        new_agent._type = current_cell.agent._type  # Access the agent's type directly
                        self.grid[empty_cell[0]][empty_cell[1]].agent = new_agent
                        current_cell.agent = None

                        self._empty.append([i,j])
                        self._num_agents_moved += 1
                        
    def _visualise_animated_grid(self, max_iterations) -> None:
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
                if self.grid[i][j].agent is not None:
                    plotGrid[i][j] = self.grid[i][j].agent._type

        plt.imshow(plotGrid, cmap=tmp)
        plt.legend(handles=patches, shadow=True, facecolor='#6A6175',
                bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
        plt.title(f'Iteration: {self._iterations_to_equilibrium}')
        plt.draw()

        if max_iterations is not None:
                one_third = round(max_iterations / 3)
                two_thirds = round(2 * max_iterations / 3)
                if self._iterations_to_equilibrium == one_third:
                    plt.savefig(self._current_directory + os.sep + "model" + os.sep + "results" + os.sep + "schellingAtThird.png")
                if self._iterations_to_equilibrium == two_thirds:
                    plt.savefig(self._current_directory + os.sep + "model" + os.sep + "results" + os.sep + "schellingAtTwoThird.png")



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
                if self.grid[i][j].agent is not None:
                    plotGrid[i][j] = self.grid[i][j].agent._type

        plt.imshow(plotGrid, cmap=tmp)
        plt.legend(handles=patches,shadow=True, facecolor='#6A6175',
                    bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(filename)
        plt.close()


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
        all_neighbors = 0
        same_neighbors = 0
        current_agent = self.grid[x][y].agent
        current_land_value = self.grid[x][y].land_value
        if current_agent is not None:
            # Check all eight neighbors
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (i != 0 or j != 0) and 0 <= x + i < self._grid_size[0] and 0 <= y + j < self._grid_size[1]:
                        neighbor = self.grid[x + i][y + j].agent
                        if neighbor is not None:
                            all_neighbors += 1
                            if neighbor._type == current_agent._type:
                                same_neighbors += 1

        neighbor_satisfaction = 0.5
        if all_neighbors != 0:
            neighbor_satisfaction = same_neighbors / all_neighbors

        combined_satisfaction = (neighbor_satisfaction + current_land_value) / 2

        if current_agent._subgroup_id == 1:
            return combined_satisfaction >= self._tolerance_higher
        else:
            return combined_satisfaction >= self._tolerance_lower
        

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
    schelling = Schelling((50,50), 90, 2, 0.6, 0.3, (20,20), (30,30), 0.5)    
    schelling.run_simulation(max_iterations=30)
    schelling.create_metrics()