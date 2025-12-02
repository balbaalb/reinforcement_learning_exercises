import numpy as np
from typing import Self
from environments.environment import Environment


class RandomJumps(Environment):
    """
    A version of random walk with continous state for the location in the axis.
    In this first version, the rules are hard-coded:
    players start at state = 0
    If |state| >= 5, the player loses and the game is over.
    If state is in the range [1.51 , 2.49], the player loses and the game is over.
    If state is in the range [2.99, 3.01], the player wins and game is over.
    Actions:
    0: jump right by 0.5
    1: jump right by 1.0
    2: jump left by -0.5
    3: jump left by -1.0
    slip: The probablity of jump direction reversal with the same hump magnitude.
    """

    def __init__(self, slip: float = 0) -> None:
        super().__init__()
        self.state = 0
        self.reward = 0
        self.gain = 0
        self.n_states = None
        self.n_actions = 4
        self.step_number = 0
        self.max_steps = 1000
        self.done = False
        # for environments with continous states and actions:
        self.n_state_features = 1  # position x
        self.n_action_features = None
        self.slip = slip

    def reset(self) -> None:
        self.state = 0
        self.reward = 0
        self.gain = 0
        self.step_number = 0
        self.done = False

    def step(self, action: int) -> Self:
        if self.done:
            self.reward = 0
            return self
        dx = 0
        self.step_number += 1
        match action:
            case 0:  # walk forward
                dx = 0.1
            case 1:  # jump forward
                dx = 1.0
            case 2:  # walk backward
                dx = -0.1
            case 3:  # jump backward
                dx = -1.0
        f = np.random.rand()
        dx *= -1 if f < self.slip else 1
        self.state += dx
        if self.state <= -5.0:
            self.done = True
        if 1.51 <= self.state <= 2.49:
            self.done = True
        if 2.99 <= self.state <= 3.01:
            self.done = True
            self.reward = 1
            self.gain = 1
        if 5.0 <= self.state:
            self.done = True
        return self
