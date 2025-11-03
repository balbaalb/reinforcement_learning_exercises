import numpy as np
from typing import Self


def get_mdp_frozen_slippery_lake(
    size: int, hole_pos: list[int], slip: float = 1.0 / 3.0
) -> dict:
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


class SlipperyFrozenLake:
    def __init__(self, size: int, hole_pos: list[int], slip: float = 1.0 / 3.0) -> None:
        self.size = size
        self.hole_pose = hole_pos
        self.slip = slip
        self.mdp = get_mdp_frozen_slippery_lake(size=size, hole_pos=hole_pos, slip=slip)
        self.start_pos = 0
        self.reset()

    def reset(self) -> None:
        self.state = 0
        self.reward = 0
        self.done = False

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
