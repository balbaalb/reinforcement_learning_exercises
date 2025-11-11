import numpy as np
from typing import Callable

PolicyType = Callable[[int, int], int]


def run_policy(policy: PolicyType, state: int, n_actions: int) -> int:
    cdfs = [0]
    cdf = 0
    for action in range(n_actions - 1):
        cdf += policy(state, action)
        cdfs.append(cdf)
    x = np.random.rand()
    return np.searchsorted(cdfs, x) - 1
