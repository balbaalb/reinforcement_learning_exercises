from typing import Callable
import numpy as np
from environments.mdp_slippery_frozen_lake import (
    get_mdp_frozen_slippery_lake,
    SlipperyFrozenLake,
)
from models.bellman_eq_methods import policy_iteration

""" 
Here, the game of frozen slippery lake is played using policy iteration, an iteration of which 
includes policy evaluation followed by policy improvement. The resulting policy wins the 
game 99.4% of episodes out of 1000 episodes. However, this high percentage of winning comes
from the knowing the full picture of the game which is not case in most applications
of the reinforcement learning. 

Policy iteration is part of dynamic programing in reinforcement learning and the methods are
directly derived from the Bellman equation. 

Note that a more efficient version of policy iteration is value iteration that combines 
policy evaluation and policy improvement in a single iteration. That is not developed here
as I will not use the dynamic programming methods in next steps. 

See  Sutton & Barto 2015 Reinforcement Learning: An Introduction, for more details.
"""


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
