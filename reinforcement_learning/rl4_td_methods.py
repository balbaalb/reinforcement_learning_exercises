from models.td_methods import *
from environments.mdp_slippery_frozen_lake import SlipperyFrozenLake
from environments.env_random_jumps import RandomJumps
import matplotlib.pyplot as plt
from pathlib import Path

"""
Here the 1st-visit on-policy Monte Carlo method is used to improve win % for slippery frozen lake.
Outcome:
Episode 10000: win ratio = 50.8 %
"""


def play_slippery_frozen_lake(
    on_policy: bool, fig_number: int, lambda_td: float = 0
) -> None:
    np.random.seed(45)
    slip = 0.33
    size = 4
    hole_pos = [5, 7, 11, 12]
    env = SlipperyFrozenLake(size=size, hole_pos=hole_pos, slip=slip)
    report_freq = 100
    n_episodes = 10000
    optimal_policy, win_ratios, _ = td_control(
        env=env,
        n_episodes=n_episodes,
        gamma=0.999,
        verbose_frequency=report_freq,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        alpha_start=1.0,
        alpha_decay=0.999,
        on_policy=on_policy,
        lambda_td=lambda_td,
    )

    for s in range(env.n_states):  # states_with_optimal_move:
        print(f"pi({s}) = {[optimal_policy(s , a) for a in range(4)]}")
    this_dir = Path(__file__).parent.resolve()
    plt.plot(np.arange(0, n_episodes, report_freq), win_ratios)
    plt.xlabel("Episode")
    plt.ylabel("Total Win %")
    title = (
        "Slippery Frozen Lake: "
        + (f"SARSA(λ = {lambda_td})" if on_policy else f"Q-learning(λ = {lambda_td})")
        + f" , Final win% = {round(win_ratios[-1], 1)} %"
    )
    fig_file_name = f"images/TD.Fig{fig_number}.png"
    plt.title(title)
    plt.savefig(this_dir / fig_file_name)
    plt.show()


def play_random_jumps(on_policy: bool, fig_number: int):
    np.random.seed(42)
    env = RandomJumps(slip=0, mode=RandomJumps.MODE.DISCRETE)
    report_freq = 100
    n_episodes = 10000
    lambda_td = 0
    optimal_policy, win_ratios, q = td_control(
        env=env,
        n_episodes=n_episodes,
        gamma=0.999,
        verbose_frequency=report_freq,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        alpha_start=1.0,
        alpha_decay=0.999,
        on_policy=on_policy,
        lambda_td=lambda_td,
    )
    this_dir = Path(__file__).parent.resolve()
    plt.plot(np.arange(0, n_episodes, report_freq), win_ratios)
    plt.xlabel("Episode")
    plt.ylabel("Total Win %")
    title = (
        "Random Jumps (Discrete): "
        + (f"SARSA(λ = {lambda_td})" if on_policy else f"Q-learning(λ = {lambda_td})")
        + f" , Final win% = {round(win_ratios[-1], 1)} %"
    )
    fig_file_name = f"images/TD.Fig{fig_number}.png"
    plt.title(title)
    plt.savefig(this_dir / fig_file_name)
    plt.show()

    x = np.linspace(-5.0, 5.0, 101)
    plt.subplot(2, 1, 1)
    for a in range(env.n_actions):
        plt.plot(x, q[:, a], label=f"action = {a}")
    plt.ylabel("Q(state, action)")
    plt.legend()
    fig_file_name = f"images/TD.Fig{fig_number}.q.png"
    plt.title(title)
    plt.subplot(2, 1, 2)
    a_opt = np.argmax(q, axis=1)
    plt.plot(x, a_opt)
    plt.xlabel("x")
    plt.ylabel("optimum action")
    plt.savefig(this_dir / fig_file_name)
    plt.show()


def main():
    play_slippery_frozen_lake(on_policy=True, fig_number=1)
    play_slippery_frozen_lake(on_policy=False, fig_number=2)
    play_slippery_frozen_lake(on_policy=True, fig_number=3, lambda_td=0.5)
    play_slippery_frozen_lake(on_policy=False, fig_number=4, lambda_td=0.1)
    play_slippery_frozen_lake(on_policy=False, fig_number=5, lambda_td=0.4)
    play_random_jumps(on_policy=True, fig_number=7)
    play_random_jumps(on_policy=False, fig_number=8)


if __name__ == "__main__":
    main()

# py -m reinforcement_learning.rl4_td_methods
