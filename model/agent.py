import random
from abc import ABC, abstractmethod


class Agent(ABC):
    """
    The base class for all agents.
    """

    def __init__(self) -> None:
        super().__init__()



class Person(Agent):
    """
    The class for the person agent.

    Attributes:
        _id (int): The unique identifier of the agent.
        _position (tuple): The position of the agent in the grid.
        _group_id (int): The group identifier of the agent.
        _subgroup_id (int): The subgroup identifier of the agent.
        _tolerance (float): The preference for in-group neighbors.
    """

    def __init__(self, tolerance_higher: float, tolerance_lower: float) -> None:
        """
        Initialise the person agent.

        Parameters:
            id (int): The unique identifier of the agent.
            position (tuple): The position of the agent in the grid.
            group_id (int): The group identifier of the agent.
            subgroup_id (int): The subgroup identifier of the agent.
            preference (float): The preference for in-group neighbors. Currently hardcoded, should be made into argument
        """
        self._type = random.choice([1, -1])
        self._subgroup_id = random.choice([1, -1])
        if (self._subgroup_id == 1):
            self._tolerance = tolerance_higher
        else:
            self._tolerance = tolerance_lower