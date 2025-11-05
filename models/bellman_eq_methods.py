import numpy as np
from typing import Callable


def value_evaluation(
    mdp: dict, policy: Callable[[int, int], float], gamma: float = 0.9
):
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
