import numpy as np
from environments.environment import Environment
from environments.run_policy import PolicyType, play_env, gen_trajectory


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
