import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from enum import Enum, auto
from slippery_frozen_lake.mdp_slippery_frozen_lake import *
from slippery_frozen_lake.game_network import *

"""
This is a very simple reinforcement learning consisting of three steps, very simialr to Monte Carlo method:
1. Run the game 10,000 times with actions taken at random and collect data
2. Map the (stat, action) to final reward of the game by using random forest or multi-layer perceptrons
3. Act greedily by choosing the action that the model predicts will yield the highest reward. 
"""


MAX_STEPS = 100
SIZE = 4


class MODEL_TYPE(Enum):
    RNDOM_FOREST = auto()
    MLP = auto()


def print_game_stats(game_memory: pd.DataFrame) -> None:
    """
    Give a summary of the played game data, such as number of episodes, win%,
    average moves per episode and etc.
    """
    num_episodes = game_memory["episode"].values[-1] + 1
    print(f"Win % = {np.sum(game_memory['reward'].values) / num_episodes * 100} %")
    print(f"Total steps = {len(game_memory)}")
    avg_game_steps = len(game_memory) / num_episodes
    max_steps_reached = game_memory["step"].values == MAX_STEPS
    print(f"Average steps / episode = {avg_game_steps}")
    print(f"Number episode with max steps = {np.sum(max_steps_reached)}")


def play_frozen_lake(
    num_episodes: int = 10000,
    game_model: RandomForestRegressor | GameNetwork | None = None,
    explore_ratio: float = 0.0,
    train_while_playing: bool = False,
    min_episodes_before_training: int = 100,
) -> pd.DataFrame:
    """
    Play the slippery frozen lake.
    Inputs:
    - num_episodes: Number of episodes ro play the game
    - game_model: If the game model is provided it will use it in a
        greedy fashion to select a move with max predicted rewward; otherwise, it will choose actions randomly.
    - train_while_playing: Train the model while playing
    - min_episodes_before_training: if train_while_playing is true, number of episodes to play the game randomly before
        feeding the history to the model for training
    Output:
    - History of the game plays in the form of pandas DataFrame
    """
    slip = 1.0 / 3.0
    hole_pos = [5, 11]
    game = SlipperyFrozenLake(size=SIZE, hole_pos=hole_pos, slip=slip)
    discount_factor = 0.9
    game_memory = pd.DataFrame(
        columns=[
            "episode",
            "step",
            "state",
            "action",
            "new_state",
            "reward",
            "discounted_reward",
        ]
    )
    first_training = True
    for episode in range(num_episodes):
        if (episode + 1) % (num_episodes // 10) == 0:
            print(f"Episode {episode + 1} started.")
        game.reset()
        old_state = game.state
        step = 0
        episode_memory = []
        while not game.done and step < MAX_STEPS:
            step += 1
            number_toss = np.random.rand()
            if game_model is None or number_toss < explore_ratio:
                action = int(np.random.rand() * 4)
            else:
                x = np.zeros([4, 2])
                x[:, 0] = old_state
                x[:, 1] = range(4)
                y = game_model.predict(x)
                action = np.argmax(y)
            game.step(action=action)
            episode_memory.append(
                [episode, step, old_state, action, game.state, game.reward, game.reward]
            )
            old_state = game.state
        n_steps = len(episode_memory)
        if game.reward > 0:
            for step in range(n_steps - 2, 0, -1):
                episode_memory[step][-1] = (
                    episode_memory[step + 1][-1] * discount_factor
                )
        episode_memory = pd.DataFrame(episode_memory, columns=game_memory.columns)
        game_memory = pd.concat([game_memory, episode_memory], ignore_index=True)
        train_model = train_while_playing and episode > min_episodes_before_training
        if train_model:
            training_data = game_memory if first_training else episode_memory
            x = training_data[["state", "action"]].values
            y = training_data["discounted_reward"].values
            first_training = False
            game_model.fit(x, y)
    return game_memory


def gen_game_model(
    game_memory: pd.DataFrame, model_type: MODEL_TYPE = MODEL_TYPE.RNDOM_FOREST
) -> RandomForestRegressor | GameNetwork:
    """
    Train a game model using either random forest or multi-layer perceptron
    """
    x = game_memory[["state", "action"]].values
    y = game_memory["discounted_reward"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    match model_type:
        case MODEL_TYPE.RNDOM_FOREST:
            game_model = RandomForestRegressor(n_estimators=100)
        case MODEL_TYPE.MLP:
            game_model = GameNetwork(size=SIZE, epochs=1000, lr=0.001)
    game_model.fit(x_train, y_train)
    y_pred = game_model.predict(x_test)
    r2 = r2_score(y_pred=y_pred, y_true=y_test)
    print(f"R2 = {r2}")
    return game_model


def gen_model_table(game_model: RandomForestRegressor) -> pd.DataFrame:
    """
    Creates a summary of the game model, what is the greedy predicted action for each state-action pair?
    """
    df = pd.DataFrame(columns=["state", "policy_action"])
    for state in range(SIZE * SIZE):
        x = np.zeros([4, 2])
        x[:, 0] = state
        x[:, 1] = range(4)
        y = game_model.predict(x)
        action = np.argmax(y)
        df.loc[len(df)] = [state, action]
    return df


def main(model_type: MODEL_TYPE) -> None:
    """
    For a given model type, play the fame and collect data, train a model using the collected data
    and finally play the game by using predictions of the model and taking the action that is
    predicted to maximise the rewrard (winning probablity).
    """
    np.random.seed(42)
    this_dir = Path(__file__).parent.resolve()
    # ------- Gather Data ---------------------------------------
    game_memory_file = this_dir / "game_data.csv"
    if not game_memory_file.is_file():
        game_memory = play_frozen_lake(num_episodes=10000)
        game_memory.to_csv(this_dir / "game_data.csv", index=False)
    game_memory = pd.read_csv(this_dir / "game_data.csv")
    print_game_stats(game_memory)
    # ------- Train Model ---------------------------------------
    game_model = gen_game_model(game_memory=game_memory, model_type=model_type)
    gen_model_table(game_model=game_model).to_csv(this_dir / "game_model.csv")
    # ------- Replay  ---------------------------------------
    game_memory2 = play_frozen_lake(num_episodes=1000, game_model=game_model)
    print(f"After Training: ")
    print_game_stats(game_memory2)
    game_memory2.to_csv(this_dir / "game_data2.csv", index=False)


if __name__ == "__main__":
    main(model_type=MODEL_TYPE.RNDOM_FOREST)
    """
    Using randon forest as a model of the game:
        Playing randomly:
            Win % = 4.57 %
            Total steps = 100990
            Average steps / episode = 10.099
            Number episode with max steps = 0
        Model traing. Model performance:
            R2 = 0.21
        Playing using the trained model:
            Win % = 39.5 %
            Total steps = 21848
            Average steps / episode = 28.613
            Number episode with max steps = 3
    """
    main(model_type=MODEL_TYPE.MLP)
    """ 
    Using a deep neural network as a model of the game:
        Playing randomly:
            Win % = 4.57 %
            Total steps = 100990
            Average steps / episode = 10.099
            Number episode with max steps = 0
        Model traing. Model performance:
            Number of Epochs:
            Loss at the last epoch: 0.0172
            R2 = 0.17
        Playing using the trained model:
            Win % = 58.9 %
            Total steps = 21344
            Average steps / episode = 21.344
            Number episode with max steps = 0
    """
    pass

# py -m slippery_frozen_lake.rl1_gather_train_act
