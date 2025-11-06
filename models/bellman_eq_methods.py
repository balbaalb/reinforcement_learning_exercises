import numpy as np
import numpy.typing as npt
from typing import Callable

"""
Dynamic programming methods that rely on the known Markov decision process (MDP) of the environment.
"""


def policy_evaluation(
    mdp: dict, policy: Callable[[int, int], float], gamma: float = 0.9
) -> npt.ArrayLike:
    """
    Produces the state value array from
    - mdp: environment MDP
    - policy: Action probability in each state: π(state, action)
    - gamma: discount factor, 0 < gamma < 1
    """
    if gamma <= 0 or gamma >= 1:
        raise Exception("Discount factor gamma must be between 0 to 1.")
    n_states = len(mdp)
    v = np.zeros(n_states)
    v_prev = np.zeros_like(v)
    convergence_reached = False
    convergence_tolerance = 1.0e-10
    iter = 0
    max_iter = 1000
    while not convergence_reached and iter < max_iter:
        for s in range(n_states):
            n_actions = len(mdp[s])
            v_new = 0
            for a in range(n_actions):
                pi = policy(s, a)
                for prob, s_next, reward, _ in mdp[s][a]:
                    v_new += pi * prob * (reward + gamma * v[s_next])
            v[s] = v_new
        convergence_reached = np.max(np.abs(v - v_prev)) < convergence_tolerance
        iter += 1
        v_prev = v.copy()
    return v


def policy_improvement(v: npt.ArrayLike, mdp: dict, gamma: float = 0.9):
    n_states = len(mdp)
    optimal_actions = np.zeros(n_states)
    for s in range(n_states):
        n_actions = len(mdp[s])
        q = np.zeros(n_actions)
        for a in range(n_actions):
            q[a] = 0
            for prob, s_next, reward, _ in mdp[s][a]:
                q[a] += prob * (reward + gamma * v[s_next])
        optimal_actions[s] = np.argmax(q)
    improved_policy = lambda s, a: (1.0 if a == optimal_actions[s] else 0)
    return improved_policy


def policy_iteration(
    mdp: dict, starting_policy: Callable[[int, int], float], gamma: float = 0.9
) -> tuple[npt.ArrayLike, Callable[[int, int], float]]:
    convergence_reached = False
    convergence_tolerance = 1.0e-10
    iter = 0
    max_iter = 1000
    v_prev = policy_evaluation(mdp=mdp, policy=starting_policy, gamma=gamma)
    while not convergence_reached and iter < max_iter:
        improved_policy = policy_improvement(v=v_prev, mdp=mdp, gamma=gamma)
        v_new = policy_evaluation(mdp=mdp, policy=improved_policy, gamma=gamma)
        convergence_reached = np.max(np.abs(v_new - v_prev)) < convergence_tolerance
        iter += 1
        v_prev = v_new.copy()
    return v_new, improved_policy
