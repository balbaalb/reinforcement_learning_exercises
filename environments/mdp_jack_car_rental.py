import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path
import time
from typing import Self
from environments.environment import Environment

"""
Jack's Car Rental Problem. 
Example 4.2 of Sutton & Barto 2015 Reinforcement Learning, page 98:

"Jack's Car Rental Jack manages two locations for a nationwide car rental company. 
Each day, some number of customers arrive at each location to rent cars. If Jack 
has a car available, he rents it out and is credited $10 by the national company. 
If he is out of cars at that location, then the business is lost. Cars become 
available for renting the day after they are returned. To help ensure that cars 
are available where they are needed, Jack can move them between the two locations 
overnight, at a cost of $2 per car moved. We assume that the number of cars 
requested and returned at each location are Poisson random variables, meaning that
the probability that the number is n is λ^n/n! * exp(-λ), where λ is the expected 
number. Suppose λ is 3 and 4 for rental requests at the first and second locations 
and 3 and 2 for returns. To simplify the problem slightly, we assume that there can
be no more than 20 cars at each location (any additional cars are returned to the
nationwide company, and thus disappear from the problem) and a maximum of five cars
can be moved from one location to the other in one night. We take the discount rate
to be γ = 0.9 and formulate this as a continuing finite MDP, where the time steps 
are days, the state is the number of cars at each location at the end of the day, 
and the actions are the net numbers of cars moved between the two locations overnight."

[Quoted from page 98 of Sutton & Barto 2015 Reinforcement Learning]
"""


def get_mdp_jack_car_rental():
    t0 = time.time()
    cap = 20  # garage_capacity
    mdp = dict()
    x1_dist = poisson(3)
    x2_dist = poisson(4)
    y1_dist = poisson(3)
    y2_dist = poisson(2)
    move_cost_per_car = 2
    bonus_per_car_rental = 10
    move_max = 5
    this_dir = Path(__file__).parent.resolve()
    df_file = this_dir / "jack_car_rental_mdp.csv"
    if df_file.is_file():
        df = pd.read_csv(df_file)
        s_data = df["state"].values.astype(int)
        m_index_data = df["move_index"].values.astype(int)
        p_data = df["prob"].values
        s_next_data = df["next_state"].values.astype(int)
        r_data = df["reward"].values
        for s in range((cap + 1) ** 2):
            mdp[s] = dict()
            for m_index in range(move_max * 2 + 1):
                mdp[s][m_index] = []
        for i in range(len(df)):
            s = s_data[i]
            m_index = m_index_data[i]
            mdp[s][m_index].append((p_data[i], s_next_data[i], r_data[i], False))
        return mdp
    for s in range((cap + 1) ** 2):
        mdp[s] = dict()
        c1 = s // (cap + 1)
        c2 = s % (cap + 1)
        print(f"s = ({c1} , {c2}) started")
        # c1 , c2: numbers of cars in garages 1 and 2 , respectively, at the end of the yesterday
        # m: number of cars moved overnight from garage 1 to garage 2
        m_max = min(c1, cap - c2, move_max)
        m_min = -min(c2, cap - c1, move_max)
        for m_index in range(move_max * 2 + 1):
            mdp[s][m_index] = []
            m = m_index - move_max
            if m < m_min or m_max < m:
                continue
            p_arr = np.zeros((cap + 1) ** 2, dtype=float)
            r_arr = np.zeros((cap + 1) ** 2, dtype=float)
            x1_max = c1 - m
            x2_max = c2 + m
            for x1 in range(x1_max + 1):
                p_x1 = x1_dist.pmf(x1)
                if p_x1 < 0.01:
                    continue
                for x2 in range(x2_max + 1):
                    p_x2 = x2_dist.pmf(x2)
                    if p_x2 < 0.01:
                        continue
                    y1_max = cap - (c1 - m)
                    y2_max = cap - (c2 + m)
                    for y1 in range(y1_max + 1):
                        p_y1 = y1_dist.pmf(y1)
                        if p_y1 < 0.01:
                            continue
                        for y2 in range(y2_max + 1):
                            p_y2 = y2_dist.pmf(y2)
                            if p_y2 < 0.01:
                                continue
                            prob = p_x1 * p_x2 * p_y1 * p_y2
                            reward = bonus_per_car_rental * (
                                x1 + x2
                            ) - move_cost_per_car * np.abs(m)
                            c1_next = c1 - m - x1 + y1
                            c2_next = c2 + m - x2 + y2
                            s_next = c1_next * (cap + 1) + c2_next
                            p_arr[s_next] += prob
                            r_arr[s_next] += prob * reward
            p_sum = np.sum(p_arr)
            if p_sum > 0:
                r_arr /= p_sum
                p_arr /= p_sum
                for s_next, (p, r) in enumerate(zip(p_arr, r_arr)):
                    if p > 0:
                        mdp[s][m_index].append((p, s_next, r, False))
    df = pd.DataFrame(columns=["state", "move_index", "prob", "next_state", "reward"])
    for s in mdp:
        for m_index in mdp[s]:
            for p, s_next, r, _ in mdp[s][m_index]:
                df.loc[len(df)] = [s, m_index, p, s_next, r]
    df.to_csv(df_file, index=False)
    t1 = time.time()
    print(f"Total mdp compilation time = {round(t1 - t0)} s")
    return mdp


class JackCarRentalProblem(Environment):
    """
    A class encapsulating the game mdp mimicing gym/gymnasium methods.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.cap = 20
        self.move_cost_per_car = 2
        self.bonus_per_car_rental = 10
        self.move_max = 5
        self.reset()
        self.range = (
            np.random.default_rng(seed=seed)
            if seed is not None
            else np.random.default_rng()
        )
        self.n_states = (self.cap + 1) * (self.cap + 1)
        self.n_actions = self.move_max * 2 + 1

    def reset(self):
        self.reward = 0
        self.state = 0
        self.episode = 0

    def step(self, action: int) -> Self:
        c1 = self.state // (self.cap + 1)
        c2 = self.state % (self.cap + 1)
        m = action - self.move_max
        m_max = min(c1, self.cap - c2, self.move_max)
        m_min = -min(c2, self.cap - c1, self.move_max)
        m = m_min if m < m_min else (m_max if m > m_max else m)
        x1 = self.range.poisson(3)
        x2 = self.range.poisson(4)
        y1 = self.range.poisson(3)
        y2 = self.range.poisson(2)
        x1_max = c1 - m
        x2_max = c2 + m
        y1_max = self.cap - (c1 - m)
        y2_max = self.cap - (c2 + m)
        x1 = x1_max if x1 > x1_max else x1
        x2 = x2_max if x2 > x2_max else x2
        y1 = y1_max if y1 > y1_max else y1
        y2 = y2_max if y2 > y2_max else y2
        c1_next = c1 - m - x1 + y1
        c2_next = c2 + m - x2 + y2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.state = c1_next * (self.cap + 1) + c2_next
        self.reward += self.bonus_per_car_rental * (
            x1 + x2
        ) - self.move_cost_per_car * np.abs(m)
        self.episode += 1
        return self


# py -m environments.jack_car_rental
