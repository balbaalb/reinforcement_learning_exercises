import numpy as np
from typing import Self
from environments.environment import Environment

"""
Creating mdp and environment for the random walk problem.
The hole is always the left most state and the goal is always 
the most right state. The starting position can be specified at
initialization time.
"""


def get_mdp_bandit(n_states: int = 5, slip: float = 0.4):
    mdp = dict()
    goal = n_states - 1
    done = lambda s1: s1 == 0 or s1 == goal
    reward = lambda s, s1: 1.0 if (s != goal and s1 == goal) else 0
    for s in range(n_states):
        mdp[s] = dict()
        for a in range(2):
            mdp[s][a] = []
            if s == 0 or s == goal:
                mdp[s][a].append((1.0, 0, 0.0, True))
            else:
                s1 = s + (-1 if a == 0 else +1)
                mdp[s][a].append(
                    (
                        1.0 - slip,
                        s1,
                        reward(s, s1),
                        done(s1),
                    )
                )
                if slip > 0:
                    s1 = s - (-1 if a == 0 else +1)
                    mdp[s][a].append(
                        (
                            slip,
                            s1,
                            reward(s, s1),
                            done(s1),
                        )
                    )
    return mdp


class RandomWalk(Environment):
    def __init__(self, n_states: int = 5, start_state: int = 2, slip: float = 0.4):
        self.n_states = n_states
        self.n_actions = 2
        self.slip = slip
        self.start_state = start_state
        self.max_steps = 10
        self.reset()

    def reset(self) -> None:
        self.done = False
        self.step_number = 0
        self.reward = 0
        self.state = self.start_state

    def step(self, action: int) -> Self:
        if not self.done:
            d_state = -1 if action == 0 else +1
            x = np.random.rand()
            d_state *= -1 if x < self.slip else 1
            self.state += d_state
            self.done = (self.state == 0) or (self.state == self.n_states - 1)
            self.reward = 1.0 if (self.state == self.n_states - 1) else 0.0
            self.step_number += 1
        return self


# py -m  environments.mdp_random_walk
