from abc import ABC, abstractmethod
from typing import Self


class Environment(ABC):
    def reset(self) -> None:
        self.reward = 0
        self.episode_number = 0
        self.done = False

    @abstractmethod
    def step(self) -> Self:
        pass
