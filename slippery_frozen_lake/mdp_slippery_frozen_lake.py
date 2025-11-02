import numpy as np


def get_mdp_frozen_slippery_lake(
    size: int, hole_pos: list[int], slip: float = 1.0 / 3.0
):
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
            left_state = state - 1 if c > 0 else state
            bottom_state = state + size if r < size - 1 else state
            right_state = state + 1 if c < size - 1 else state
            top_state = state - size if r > 0 else state

            mdp[state][go_left] = [
                (
                    1.0 - 2.0 * slip,
                    left_state,
                    get_reward(state, left_state),
                    left_state in terminal_states,
                ),
                (
                    slip,
                    top_state,
                    get_reward(state, top_state),
                    top_state in terminal_states,
                ),
                (
                    slip,
                    bottom_state,
                    get_reward(state, bottom_state),
                    bottom_state in terminal_states,
                ),
            ]
            mdp[state][go_right] = [
                (
                    1.0 - 2.0 * slip,
                    right_state,
                    get_reward(state, right_state),
                    right_state in terminal_states,
                ),
                (
                    slip,
                    top_state,
                    get_reward(state, top_state),
                    top_state in terminal_states,
                ),
                (
                    slip,
                    bottom_state,
                    get_reward(state, bottom_state),
                    bottom_state in terminal_states,
                ),
            ]
            mdp[state][go_top] = [
                (
                    1.0 - 2.0 * slip,
                    top_state,
                    get_reward(state, top_state),
                    top_state in terminal_states,
                ),
                (
                    slip,
                    left_state,
                    get_reward(state, left_state),
                    left_state in terminal_states,
                ),
                (
                    slip,
                    right_state,
                    get_reward(state, right_state),
                    right_state in terminal_states,
                ),
            ]
            mdp[state][go_bottom] = [
                (
                    1.0 - 2.0 * slip,
                    bottom_state,
                    get_reward(state, bottom_state),
                    bottom_state in terminal_states,
                ),
                (
                    slip,
                    left_state,
                    get_reward(state, left_state),
                    left_state in terminal_states,
                ),
                (
                    slip,
                    right_state,
                    get_reward(state, right_state),
                    right_state in terminal_states,
                ),
            ]
    return mdp


class SlipperyFrozenLake:
    def __init__(self, size: int, hole_pos: list[int], slip: float = 1.0 / 3.0):
        self.size = size
        self.hole_pose = hole_pos
        self.slip = slip
        self.mdp = get_mdp_frozen_slippery_lake(size=size, hole_pos=hole_pos, slip=slip)
        self.start_pos = 0
        self.reset()

    def reset(self):
        self.state = 0
        self.reward = 0
        self.done = False

    def step(self, action: int):
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


if __name__ == "__main__":
    size = 3
    hole_pos = [4]
    slip = 0.2  # if action = go-right, there is slip probability that the players go up or down instead!
    mdp = get_mdp_frozen_slippery_lake(size=size, hole_pos=hole_pos, slip=slip)
    for state in mdp:
        for action in range(4):
            print(f"mdp[{state}][{action}] = {mdp[state][action]}")
