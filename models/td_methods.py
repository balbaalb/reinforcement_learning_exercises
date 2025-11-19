import numpy as np
from environments.environment import Environment
from environments.run_policy import run_policy, PolicyType, play_env


def td_control(
    env: Environment,
    n_episodes: int = 1000,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.99,
    alpha_start: float = 1.0,
    alpha_decay: float = 0.99,
    lambda_td: float = 0,
    verbose_frequency: int | None = None,
    on_policy: bool = False,
) -> PolicyType:
    """
    Control method that uses one step TD-learning.
    Eligibility trace type: accumulating traces
    """
    policy = lambda s, a: 1.0 / env.n_actions
    q = np.zeros([env.n_states, env.n_actions], dtype=float)
    a_optimum = np.zeros(env.n_states, dtype=int)
    epsilon = epsilon_start
    alpha = alpha_start
    win_ratios = []
    for episode in range(n_episodes):
        if verbose_frequency is not None and (episode + 1) % verbose_frequency == 0:
            win_ratios.append(play_env(env=env, n_episodes=1000, policy=policy))
            print(f"Episode {episode + 1}: win ratio= {round(win_ratios[-1], 2)} %")
        env.reset()
        E = np.zeros_like(q)
        while not env.done and env.step_number < env.max_steps:
            s0 = env.state
            a0 = run_policy(policy=policy, state=s0, n_actions=env.n_actions)
            env.step(a0)
            s1 = env.state
            r = env.reward
            if on_policy:  # SARSA
                a1 = run_policy(policy=policy, state=s1, n_actions=env.n_actions)
            else:  # Q-learning
                a1 = a_optimum[s1]
                if lambda_td > 1.0e-10:
                    ap = run_policy(policy=policy, state=s1, n_actions=env.n_actions)
            g = r + gamma * q[s1, a1]
            diff = g - q[s0, a0]
            q[s0, a0] += alpha * diff
            if lambda_td > 1.0e-10:
                q += alpha * diff * E
                if on_policy or ap == a1:
                    E[s0, a0] += 1
                    E = gamma * lambda_td * E
                else:
                    E = 0
            a_optimum[s0] = np.argmax(q[s0, :])
            policy = lambda s, a: epsilon / env.n_actions + (
                (1.0 - epsilon) if a == np.argmax(q[s, :]) else 0
            )
        alpha *= alpha_decay
        epsilon *= epsilon_decay
    return policy, win_ratios
