import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from environments.environment import Environment
from environments.run_policy import *


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
                    E = np.zeros_like(E)
            a_optimum[s0] = np.argmax(q[s0, :])
            policy = lambda s, a: epsilon / env.n_actions + (
                (1.0 - epsilon) if a == np.argmax(q[s, :]) else 0
            )
        alpha *= alpha_decay
        epsilon *= epsilon_decay
    return policy, win_ratios


class Network(nn.Sequential):
    def __init__(
        self, in_features: int = 1, depths: list[int] = [], out_features: int = 1
    ) -> None:
        super().__init__()
        self.layers = []
        n_input = in_features
        for i, depth in enumerate(depths):
            self.add_module(
                name=f"Layer-{i}",
                module=nn.Linear(in_features=n_input, out_features=depth),
            )
            self.add_module(name=f"Activation-{i - 1}", module=nn.ReLU())
            n_input = depth
        self.add_module(
            name="Output",
            module=nn.Linear(in_features=n_input, out_features=out_features),
        )
        self.add_module(
            name="Softmax",
            module=nn.Softmax(dim=1),
        )

    def forward(self, x):
        return super().forward(x)


def deep_q_learning(
    env: Environment,
    n_episodes: int = 1000,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.99,
    verbose_frequency: int | None = None,
    q_network_depths: list[int] = [256, 256],
    n_epochs: int = 10,
    batch_size=1000,
    lr: float = 0.001,
) -> DeterminisiticPolicyType:
    random_policy = lambda s: np.random.choice(np.arange(env.n_actions))
    policy = random_policy
    qs_network = Network(
        in_features=env.n_state_features,
        depths=q_network_depths,
        out_features=env.n_actions,
    )
    optimizer = torch.optim.Adam(params=qs_network.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epsilon = epsilon_start
    win_ratios = []
    sars_traces = []
    for episode in range(n_episodes):
        if verbose_frequency is not None and (episode + 1) % verbose_frequency == 0:
            with torch.no_grad():
                win_ratios.append(
                    play_env_det_policy(env=env, n_episodes=1000, det_policy=policy)
                )
            print(
                f"Episode {episode + 1}: trace length: {len(sars_traces)}, "
                + f"win ratio= {round(win_ratios[-1], 2)} %, epsilon = {epsilon}"
            )
        env.reset()
        while not env.done and env.step_number < env.max_steps:
            s0 = env.state
            a0 = policy(s0)
            env.step(a0)
            s1 = env.state
            r = env.reward
            done = 1 if env.done else 0
            sars_traces.append([s0, a0, r, s1, done])
            if len(sars_traces) >= batch_size:
                np_sars_traces = np.array(sars_traces)
                s0_scaled = np_sars_traces[:, 0].reshape(-1, 1) / 5.0
                a0 = np_sars_traces[:, 1].astype(int)
                r = np_sars_traces[:, 2].reshape(-1, 1)
                s1_scaled = np_sars_traces[:, 3].reshape(-1, 1) / 5.0
                d_coeff = (np_sars_traces[:, 4] - 1.0).reshape(-1, 1)

                s0_torch = torch.FloatTensor(s0_scaled)
                s1_torch = torch.FloatTensor(s1_scaled)
                r_torch = torch.FloatTensor(r)
                d_coeff_torch = torch.FloatTensor(d_coeff)
                for epoch in range(n_epochs):
                    qs0_torch = qs_network(s0_torch)
                    q0_torch = qs0_torch[a0]
                    qs1_torch = qs_network(s1_torch)
                    a1 = qs1_torch.max(-1)[1]
                    q1_torch = qs1_torch[a1]
                    g = q1_torch * d_coeff_torch * gamma + r_torch
                    # print(f"shapes: {s0_torch.shape} , {qs0_torch.shape}, {a0.shape}, {q0_torch.shape}")
                    optimizer.zero_grad()
                    # print(f"g.shape = {g.shape}, q0.shape = {q0_torch.shape}")
                    loss = criterion(g, q0_torch)
                    loss.backward()
                    optimizer.step()
                print(f"epoch = {epoch + 1}, loss = {loss.item()}")
                sars_traces = []

                def policy(s):
                    if np.random.randn() < epsilon:
                        return random_policy(s)
                    with torch.no_grad():
                        s_torch = torch.tensor([s / 5.0]).reshape(-1, 1)
                        return qs_network(s_torch).max(-1)[1].item()

        epsilon *= epsilon_decay
    return policy, win_ratios
