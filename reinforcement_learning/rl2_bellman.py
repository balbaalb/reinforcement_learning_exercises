from typing import Callable
import numpy as np
from environments.mdp_slippery_frozen_lake import (
    get_mdp_frozen_slippery_lake,
    SlipperyFrozenLake,
)
from models.bellman_eq_methods import policy_iteration


def get_action(policy: Callable[[int, int], float], state: int):
    q = np.array([policy(state, a) for a in range(4)])
    return np.argmax(q)


def main():
    size = 4
    hole_pos = [5, 11]
    slip = 1.0 / 3.0
    mdp = get_mdp_frozen_slippery_lake(size=size, hole_pos=hole_pos, slip=slip)
    starting_policy = lambda s, a: 0.25  # Random policy
    _, optimal_policy = policy_iteration(
        mdp=mdp, starting_policy=starting_policy, gamma=0.99
    )
    game = SlipperyFrozenLake(size=size, hole_pos=hole_pos, slip=slip)
    num_episodes = 1000
    max_steps = 100
    episodes_reward = []
    for epsiode in range(num_episodes):
        game.reset()
        step = 0
        reward = 0
        while not game.done and step < max_steps:
            step += 1
            q = np.array([optimal_policy(game.state, a) for a in range(4)])
            game.step(action=np.argmax(q))
            reward += game.reward
        episodes_reward.append(reward)
        if (epsiode + 1) % 100 == 0:
            print(f"Epsiode {epsiode + 1}")
    wins = np.array(episodes_reward) == 1
    win_percent = np.sum(wins) / num_episodes * 100
    print(f"win % = {win_percent} %")
    """ 
    win % = 99.4 %        
    """


if __name__ == "__main__":
    main()


# py -m reinforcement_learning.rl2_bellman
