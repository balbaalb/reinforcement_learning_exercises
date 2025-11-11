from models.monte_carlo_methods import *
from environments.mdp_slippery_frozen_lake import SlipperyFrozenLake
from environments.mdp_random_walk import RandomWalk
from environments.run_policy import run_policy


def test_monte_carlo_control_1v_on():
    np.random.seed(123)
    env = RandomWalk(n_states=5, start_state=2, slip=0.4)
    optimal_policy, _ = monte_carlo_control_1v_on(
        env=env,
        n_episodes=100,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.995,
    )
    for s in range(1, env.n_states - 1):  # states_with_optimal_move:
        assert optimal_policy(s, 0) < optimal_policy(s, 1)

    env = RandomWalk(n_states=5, start_state=2, slip=0.6)
    optimal_policy, _ = monte_carlo_control_1v_on(
        env=env,
        n_episodes=100,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.995,
    )
    for s in range(1, env.n_states - 1):  # states_with_optimal_move:
        assert optimal_policy(s, 0) > optimal_policy(s, 1)


test_monte_carlo_control_1v_on()
# py -m test.test_monte_carlo_methods
