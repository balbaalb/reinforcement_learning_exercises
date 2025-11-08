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
    - mdp: Known Markov decision process (mdp) of the environment
    - policy: Action probability in each state: π(state, action)
    - gamma: the discount factor for rewards in the next step, 0 < gamma < 1
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


def policy_improvement(
    v: npt.ArrayLike, mdp: dict, gamma: float = 0.9
) -> Callable[[int, int], float]:
    """
    Generates a greedy policy that based on maximum value
    of state-action pair.
    Inputs:
    - v: numpy array of state values
    - mdp: Known Markov decision process (mdp) of the environment
    - gamma: the discount factor for rewards in the next step, 0 < gamma < 1
    Output:
    - Improved policy, π*(state, action)
    """
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
    """
    Combines policy evaluation and policy improvement iteratively to produce
    an optimal policy.
    Inputs:
    - mdp: Known Markov decision process (mdp) of the environment
    - starting_policy: Initial action probability in each state: π0(state, action)
    - gamma: the discount factor for rewards in the next step, 0 < gamma < 1
    Output:
    - v: Final state-values in numpy array format
    - π*(state, action): Final optimal policy
    """
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


def value_iteration(
    mdp: dict, gamma: float = 0.9, verbose:bool = False
) -> tuple[npt.ArrayLike, Callable[[int, int], float]]:
    """
    Combines policy evaluation and policy improvement iterations used in policy improvement
    into a combined iteration.
    Inputs:
    - mdp: Known Markov decision process (mdp) of the environment
    - gamma: the discount factor for rewards in the next step, 0 < gamma < 1
    Output:
    - v: Final state-values in numpy array format
    - π*(state, action): Final optimal policy
    """
    n_states = len(mdp)
    v = np.zeros(n_states)
    v_prev = np.zeros_like(v)
    convergence_reached = False
    convergence_tolerance = 1.0e-10
    iter = 0
    max_iter = 1000
    optimal_actions = np.zeros(n_states)
    while not convergence_reached and iter < max_iter:
        for s in range(n_states):
            n_actions = len(mdp[s])
            v_new = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, s_next, reward, _ in mdp[s][a]:
                    v_new[a] += prob * (reward + gamma * v[s_next])
            a_optimum = np.argmax(v_new)
            v[s] = v_new[a_optimum]
            optimal_actions[s] = a_optimum
        convergence_reached = np.max(np.abs(v - v_prev)) < convergence_tolerance
        iter += 1
        if verbose and iter % 10 == 0:
            print(f"iter = {iter}, err = {np.max(np.abs(v - v_prev))}")
        v_prev = v.copy()
    if iter == max_iter:
        print(f"max iter reached!!!")
    optimal_policy = lambda s, a: 1.0 if a == optimal_actions[s] else 0.0
    return v, optimal_policy, optimal_actions
