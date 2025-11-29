import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Callable
from environments.environment import Environment

PolicyType = Callable[[int, int], int]
PolicyType_Continous = Callable[[npt.ArrayLike, int], int]


def run_policy(
    policy: PolicyType | PolicyType_Continous,
    state: int | npt.ArrayLike,
    n_actions: int,
) -> int:
    """
    Runs a deterministic or undeterministic policy.
    State can be state index or array of state features.
    """
    cdfs = [0]
    cdf = 0
    for action in range(n_actions - 1):
        cdf += policy(state, action)
        cdfs.append(cdf)
    x = np.random.rand()
    return np.searchsorted(cdfs, x) - 1


def play_env(
    env: Environment, n_episodes: int, policy: PolicyType | PolicyType_Continous
):
    total_rewards = 0
    for _ in range(n_episodes):
        env.reset()
        while not env.done and env.step_number < env.max_steps:
            a = run_policy(policy, state=env.state, n_actions=env.n_actions)
            env.step(a)
            total_rewards += env.reward
    win_percent = total_rewards / n_episodes * 100
    return win_percent


def gen_trajectory(env: Environment, policy: PolicyType, gamma: float = 0):
    env.reset()
    trajectory = pd.DataFrame(columns=["state", "action", "reward"])
    while not env.done and env.step_number < env.max_steps:
        env.step_number += 1
        prev_state = env.state
        action = run_policy(policy=policy, state=prev_state, n_actions=env.n_actions)
        env.step(action)
        trajectory.loc[len(trajectory)] = [prev_state, action, env.reward]
    gains = np.zeros(len(trajectory))
    rewards = trajectory["reward"].values
    for i in range(len(trajectory) - 1, -1, -1):
        gains[i] = rewards[i] + gamma * (gains[i + 1] if i < len(trajectory) - 1 else 0)
    trajectory["gain"] = gains
    return trajectory
