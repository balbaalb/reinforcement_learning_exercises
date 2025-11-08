from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from environments.mdp_slippery_frozen_lake import (
    get_mdp_frozen_slippery_lake,
    SlipperyFrozenLake,
)
from environments.mdp_jack_car_rental import (
    get_mdp_jack_car_rental,
    JackCarRentalProblem,
)

from models.bellman_eq_methods import policy_iteration, value_iteration

""" 
Here, the game of frozen slippery lake is played using policy iteration, an iteration of which 
includes policy evaluation followed by policy improvement. The resulting policy wins the 
game 99.4% of episodes out of 1000 episodes. However, this high percentage of winning comes
from the knowing the full picture of the game which is not case in most applications
of the reinforcement learning. 

Policy iteration is part of dynamic programing in reinforcement learning and the methods are
directly derived from the Bellman equation. 

Next, a more efficient version of policy iteration is value iteration that combines 
policy evaluation and policy improvement in a single iteration.  

See Sutton & Barto 2015 Reinforcement Learning: An Introduction, for more details.
"""


def play_slippery_frozen_lake():
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
    print(f"win % using policy iteration= {win_percent} %")
    """ 
    win % using policy iteration= 99.7 %      
    """
    _, optimal_policy = value_iteration(mdp=mdp, gamma=0.99)
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
    print(f"win % using value iteration= {win_percent} %")
    """
    win % using value iteration= 99.6 %
    """


def solve_jack_car_rental_problem():
    # playing randomly
    env = JackCarRentalProblem(seed=42)
    n_episodes = 1000
    actions = np.random.randint(11, size=n_episodes)
    for m_index in actions:
        env.step(m_index)
    print(f"Random moves: Average income  = ${env.reward / n_episodes}")
    """ 
    Random moves: Average income  = $43.946
    """
    mdp = get_mdp_jack_car_rental()
    print(f"mdp obtained")
    _, _, optimal_actions = value_iteration(mdp=mdp, gamma=0.9, verbose=True)
    print(f"value iteration done")
    env2 = JackCarRentalProblem(seed=42)
    for _ in range(n_episodes):
        env2.step(int(optimal_actions[env2.state]))
    print(f"Optimized moves: Average income  = ${env2.reward / n_episodes}")
    """ 
    Optimized moves: Average income  = $48.21
    """
    x = np.arange(0, 21)
    y = np.arange(0, 21)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros([21, 21])
    for s in range(env2.n_states):
        c1 = s // (env2.cap + 1)
        c2 = s % (env2.cap + 1)
        zz[c1, c2] = optimal_actions[s] - 5

    levels = np.arange(-5, 6)
    _, ax = plt.subplots(figsize=(5,5))
    contours = ax.contour(
        xx,
        yy,
        zz,
        levels=levels,
        colors = 'k',
    )
    ax.clabel(contours, inline=True, fontsize=10)
    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    # play_slippery_frozen_lake()
    solve_jack_car_rental_problem()


# py -m reinforcement_learning.rl1_bellman
