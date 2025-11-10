from models.monte_carlo_methods import *
from environments.mdp_slippery_frozen_lake import SlipperyFrozenLake
from environments.bandit import Bandit


##  🚧🚧🚧🚧🚧🚧🚧🚧  In Progress 🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧🚧
def t_monte_carlo_control_1v_on():
    slip = 0.33  # 1.0 / 3.0
    size = 4
    hole_pos = [5, 7, 11, 12]
    env = SlipperyFrozenLake(size=size, hole_pos=hole_pos, slip=slip)
    # env = Bandit(n_states=5, start_state=2, slip=0.4)
    print(f"n_states = {env.n_states}")
    print(f"n_actions = {env.n_actions}")
    optimal_policy = monte_carlo_control_1v_on(
        env=env,
        n_episodes=10000,
        gamma=0.99,
        verbose_frequency=100,
        epsilon_start=1,
        epsilon_decay=0.99,
    )
    # states_with_optimal_move = [0, 1, 2, 3, 4, 8, 9, 10, 13, 14]
    # optimal_moves =            [0, 3, 3, 3, 0, 3, 1, 0,  2,  1]
    for s in range(env.n_states):  # states_with_optimal_move:
        print(f"pi({s}) = {[optimal_policy(s , a) for a in range(4)]}")
        # for a in range(4):
        #     assert optimal_policy(s, a) == (1 if a == optimal_moves[i] else 0)


# t_monte_carlo_control_1v_on()
# py -m test.test_monte_carlo_methods
