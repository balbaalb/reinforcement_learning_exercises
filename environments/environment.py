from abc import ABC, abstractmethod
from typing import Self
import numpy as np


class Environment(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action: int) -> Self:
        pass
