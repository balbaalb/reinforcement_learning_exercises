from abc import ABC, abstractmethod
from typing import Self


class Environment(ABC):
    def __init__(self):
        super().__init__()
        self.n_states = 0
        self.n_action = 0
        self.reset()

    def reset(self) -> None:
        self.reward = 0
        self.episode_number = 0
        self.done = False

    @abstractmethod
    def step(self) -> Self:
        pass
