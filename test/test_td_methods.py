from models.td_methods import *
from environments.mdp_random_walk import RandomWalk


def test_td_control_1():
    """
    off-policy, lambda = 0
    """
    np.random.seed(123)
    env = RandomWalk(n_states=5, start_state=2, slip=0.4)
    optimal_policy, _, _ = td_control(
        env=env,
        n_episodes=100,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        alpha_start=1.0,
        alpha_decay=0.9,
        on_policy=False,
    )
    for s in range(1, env.n_states - 1):  # states_with_optimal_move:
        assert optimal_policy(s, 0) < optimal_policy(s, 1)


def test_td_control_2():
    """
    on-policy, lambda = 0
    """
    np.random.seed(123)
    env = RandomWalk(n_states=5, start_state=2, slip=0.4)
    optimal_policy, _, _ = td_control(
        env=env,
        n_episodes=300,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        alpha_start=1.0,
        alpha_decay=0.999,
        on_policy=True,
    )
    for s in range(1, env.n_states - 1):  # states_with_optimal_move:
        assert optimal_policy(s, 0) < optimal_policy(s, 1)


def test_td_control_3():
    """
    on-policy, lambda = 0.5
    """
    np.random.seed(123)
    env = RandomWalk(n_states=5, start_state=2, slip=0.4)
    optimal_policy, _, _ = td_control(
        env=env,
        n_episodes=1000,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        alpha_start=1.0,
        alpha_decay=0.999,
        on_policy=True,
        lambda_td=0.5,
    )
    for s in range(1, env.n_states - 1):  # states_with_optimal_move:
        assert optimal_policy(s, 0) < optimal_policy(s, 1)


def test_td_control_4():
    """
    off-policy, lambda = 0.5
    """
    np.random.seed(123)
    env = RandomWalk(n_states=5, start_state=2, slip=0.4)
    optimal_policy, _, _ = td_control(
        env=env,
        n_episodes=100,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        alpha_start=1.0,
        alpha_decay=0.9,
        on_policy=False,
        lambda_td=0.5,
    )
    for s in range(1, env.n_states - 1):  # states_with_optimal_move:
        assert optimal_policy(s, 0) < optimal_policy(s, 1)
