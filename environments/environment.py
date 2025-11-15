from abc import ABC, abstractmethod
from typing import Self
import numpy as np


class Environment(ABC):
    def __init__(self):
        super().__init__()
        self.state = None
        self.reward = None
        self.n_states = None
        self.n_actions = None
        self.step_number = None
        self.max_steps = None
        self.done = None

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action: int) -> Self:
        pass
