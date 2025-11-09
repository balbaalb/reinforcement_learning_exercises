import numpy as np
from typing import Self
from environments.environment import Environment

"""
The best way to generate the slippery frozen lake game is to use openAI's gym which is now FF's gymnasium. 
But just to get a feel of the mdp, the game is built here. Another advantage is that now we don't have to worry 
about who owns gym/gymnasium and whether its name and output structure of the functions has changed again!
"""


def get_mdp_frozen_slippery_lake(
    size: int, hole_pos: list[int], slip: float = 1.0 / 3.0
) -> dict:
    """
    Creates Markov decision process for slippery frozen lake.
    Inputs:
    - size: side size of the game board which is square
    - hole_pose: a list of location indices of the holes.
    - slip: The probablity of slipping in perpendicular direction of the intended move.
    Output:
    - mdp of the game in the form of: mdp[state][action] = list[(probablity, next_state, reward, is_done)]
    Example:
    mdp = get_mdp_frozen_slippery_lake(size=3, hole_pos=[4], slip=0.2) generates a game in the form
    of a 3x3 square with the central cell being a hole. Each move has 0.2 probablity of being diverted in a
    perpendicular direction. For example, a move to left means 60% chance of going to the left cell,
    20% chance of going to the bottom cell and 20% chance of going to the top cell.
    see test file test/test_mdp_slippery_frozen_lake.py for details of the example.
    """
    n_states = size * size
    mdp = dict()
    goal_pos = n_states - 1
    terminal_states = hole_pos + [goal_pos]
    go_left = 0
    go_bottom = 1
    go_right = 2
    go_top = 3
    side_moves = dict()

    side_moves[go_left] = [go_bottom, go_top]
    side_moves[go_right] = [go_bottom, go_top]
    side_moves[go_top] = [go_left, go_right]
    side_moves[go_bottom] = [go_left, go_right]

    def get_reward(state, next_state):
        return 1.0 if state != goal_pos and next_state == goal_pos else 0

    for state in range(n_states):
        r = state // size
        c = state % size
        mdp[state] = dict()
        if state in terminal_states:
            for action in range(4):
                mdp[state][action] = [(1.0, state, 0.0, True)]
        else:
            for action in range(4):
                mdp[state][action] = []
                targets = [
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0],
                ]  # (dr, dc, prob)
                for i in range(4):
                    target = targets[i]
                    rt = r + target[0]
                    ct = c + target[1]
                    if rt < 0 or rt >= size or ct < 0 or ct >= size:
                        target = targets[4]
                    p = (
                        (1 - 2 * slip)
                        if (i == action)
                        else (
                            slip
                            if (i == (action + 1) % 4 or i == (action + 3) % 4)
                            else 0
                        )
                    )
                    target[2] += p
                for target in targets:
                    p = target[2]
                    if p > 0:
                        rt = r + target[0]
                        ct = c + target[1]
                        n = rt * size + ct
                        reward = get_reward(state, n)
                        is_terminal = n in terminal_states
                        mdp[state][action].append((p, n, reward, is_terminal))
    return mdp


class SlipperyFrozenLake(Environment):
    """
    A class encapsulating the game mdp mimicing gym/gymnasium methods.
    """

    def __init__(self, size: int, hole_pos: list[int], slip: float = 1.0 / 3.0) -> None:
        self.size = size
        self.hole_pose = hole_pos
        self.slip = slip
        self.mdp = get_mdp_frozen_slippery_lake(size=size, hole_pos=hole_pos, slip=slip)
        self.start_pos = 0
        self.n_states = size**2
        self.n_actions = 4
        self.max_steps = 1000
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.state = 0

    def step(self, action: int) -> Self:
        if not self.done:
            outcomes = self.mdp[self.state][action]
            probs = []
            for i in range(len(outcomes)):
                probs.append(outcomes[i][0])
            probs_cumsum = np.zeros(len(probs) + 1)
            probs_cumsum[1:] = np.cumsum(probs)

            x = np.random.rand()
            idx = np.searchsorted(probs_cumsum, x) - 1
            outcome = outcomes[idx]
            self.state = outcome[1]
            self.reward += outcome[2]
            self.done = outcome[3]
        else:
            self.reward = 0
        return self
