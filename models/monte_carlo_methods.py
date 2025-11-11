import pandas as pd
import numpy as np
from environments.environment import Environment
from environments.run_policy import run_policy, PolicyType


def play_env(env: Environment, n_episodes: int, policy: PolicyType):
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


def monte_carlo_control_1v_on(
    env: Environment,
    n_episodes: int = 1000,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.99,
    verbose_frequency: int | None = None,
) -> PolicyType:
    """
    First-visit on-policy Monte Carlo control method
    """
    policy = lambda s, a: 1.0 / env.n_actions
    q = np.zeros([env.n_states, env.n_actions], dtype=float)
    n_q = np.zeros([env.n_states, env.n_actions], dtype=int)
    a_optimum = np.zeros(env.n_states)
    epsilon = epsilon_start
    win_ratios = []
    for episode in range(n_episodes):
        if verbose_frequency is not None and (episode + 1) % verbose_frequency == 0:
            win_ratios.append(play_env(env=env, n_episodes=1000, policy=policy))
            print(f"Episode {episode + 1}: win ratio= {round(win_ratios[-1], 2)} %")
        trajectory = gen_trajectory(env=env, policy=policy, gamma=gamma)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                mask = (trajectory["state"] == s) & (trajectory["action"] == a)
                if len(trajectory[mask]) > 0:
                    g = trajectory[mask]["gain"].values[0]
                    n_q[s, a] += 1
                    alpha = 1.0 / n_q[s, a]
                    diff = g - q[s, a]
                    q[s, a] += alpha * diff
            a_optimum[s] = np.argmax(q[s, :])
        epsilon *= epsilon_decay
        policy = lambda s, a: epsilon / env.n_actions + (
            (1.0 - epsilon) if a == a_optimum[s] else 0
        )
    return policy, win_ratios
