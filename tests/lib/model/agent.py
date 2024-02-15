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
    """

    def __init__(self) -> None:
        """
        Initialise the person agent.

        Parameters:
            id (int): The unique identifier of the agent.
            position (tuple): The position of the agent in the grid.
            group_id (int): The group identifier of the agent.
        """
        self._type = random.choice([1, -1])

    

