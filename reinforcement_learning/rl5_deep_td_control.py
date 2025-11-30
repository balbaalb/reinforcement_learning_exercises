from environments.env_random_jumps import *
from models.td_methods import *
import matplotlib.pyplot as plt
from pathlib import Path
import time


def play_episodes() -> None:
    np.random.seed(45)
    torch.manual_seed(123)
    slip = 0
    env = RandomJumps(slip=slip)
    report_freq = 10
    n_episodes = 10000
    t0 = time.time()
    _, win_ratios, losses = deep_q_learning(
        env=env,
        n_episodes=n_episodes,
        gamma=0.999,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        verbose_frequency=report_freq,
        q_network_depths=[64, 128, 64],
        n_epochs=1000,
        max_n_batches=10,
        batch_size=1000,
        lr=1.0e-5,
    )
    t1 = time.time()
    print(f"Training time ={t1 - t0} s = {round((t1 - t0) / 3600, 1)} hr")
    _, ax = plt.subplots(nrows=2, figsize=(10, 8))
    ax[0].plot(np.arange(0, n_episodes, report_freq), win_ratios)
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Total Win %")
    title = (
        "Random Jumps, deep Q-learning: "
        + f" , Final win% = {round(win_ratios[-1], 1)} %"
    )
    ax[0].grid(True)
    ax[0].set_title(title)
    ax[1].plot(losses)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)
    this_dir = Path(__file__).parent.resolve()
    fig_file_name = f"images/TD.Fig{6}.png"
    plt.tight_layout()
    plt.savefig(this_dir / fig_file_name)
    plt.show()


def main():
    play_episodes()


if __name__ == "__main__":
    main()

# py -m reinforcement_learning.rl5_deep_td_control
