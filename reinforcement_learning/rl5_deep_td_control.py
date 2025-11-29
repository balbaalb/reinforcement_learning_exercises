from environments.env_random_jumps import *
from models.td_methods import *
import matplotlib.pyplot as plt
from pathlib import Path


def play_episodes() -> None:
    np.random.seed(45)
    torch.manual_seed(123)
    slip = 0
    env = RandomJumps(slip=slip)
    report_freq = 1
    n_episodes = 1000
    _, win_ratios = deep_q_learning(
        env=env,
        n_episodes=n_episodes,
        gamma=0.999,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        verbose_frequency=report_freq,
        q_network_depths=[64, 128, 64],
        n_epochs=1000,
        batch_size=10,
        lr=1.0e-5,
    )

    plt.plot(np.arange(0, n_episodes, report_freq), win_ratios)
    plt.xlabel("Episode")
    plt.ylabel("Total Win %")
    plt.show()


def main():
    play_episodes()


if __name__ == "__main__":
    main()

# py -m reinforcement_learning.rl5_deep_td_control
