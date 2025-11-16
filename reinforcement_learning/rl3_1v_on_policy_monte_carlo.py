from models.monte_carlo_methods import *
from environments.mdp_slippery_frozen_lake import SlipperyFrozenLake
import matplotlib.pyplot as plt
from pathlib import Path

"""
Here the 1st-visit on-policy Monte Carlo method is used to improve win % for slippery frozen lake.
Outcome:
Episode 10000: win ratio = 50.8 %
"""


def main() -> None:
    np.random.seed(45)
    slip = 0.33  # 1.0 / 3.0
    size = 4
    hole_pos = [5, 7, 11, 12]
    env = SlipperyFrozenLake(size=size, hole_pos=hole_pos, slip=slip)
    optimal_policy, win_ratios = monte_carlo_control_1v_on(
        env=env,
        n_episodes=10000,
        gamma=0.999,
        verbose_frequency=100,
        epsilon_start=1,
        epsilon_decay=0.999,
    )

    for s in range(env.n_states):  # states_with_optimal_move:
        print(f"pi({s}) = {[optimal_policy(s , a) for a in range(4)]}")
    this_dir = Path(__file__).parent.resolve()
    plt.plot(win_ratios)
    plt.xlabel("Episode")
    plt.ylabel("Total Win %")
    plt.title("First-visit on-policy Monte Carlo control\nfor slippery frozen lake")
    plt.savefig(this_dir / "images/Slippery-frozen-lake-1v-on-policy-MonteCarlo")
    plt.show()


if __name__ == "__main__":
    main()

# py -m reinforcement_learning.rl3_1v_on_policy_monte_carlo
