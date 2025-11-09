from abc import ABC, abstractmethod
from typing import Self
import numpy as np


class Environment(ABC):
    def __init__(self):
        super().__init__()
        self.n_states = 1
        self.n_actions = 0
        self.max_steps = 0
        self.state = 0
        self.reset()

    def reset(self) -> None:
        self.reward = 0
        self.step_number = 0
        self.done = False
        self.state = np.random.randint(self.n_states)

    @abstractmethod
    def step(self, action: int) -> Self:
        pass
