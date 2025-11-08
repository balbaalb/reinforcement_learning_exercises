import numpy as np
from environments.mdp_jack_car_rental import (
    get_mdp_jack_car_rental,
    JackCarRentalProblem,
)
from pytest import approx


def test_get_mdp_jack_car_rental():
    mdp = get_mdp_jack_car_rental()
    assert len(mdp) == 441
    for s in range(len(mdp)):
        assert len(mdp[s]) == 11
    # state 0: c1 = 0, c2 = 0 so the only possible move should be 0 or m_index = 5
    for m_index in range(11):
        m = m_index - 5
        if m != 0:
            assert len(mdp[0][m_index]) == 0
    # for c1 = c2 = 0 and m = 0, x1 = x2 = 0. Then the number of possible
    # states should be equal to
    # (number of possible states for y1) x (number of possible states for y1) = 8 x 7 = 56
    assert len(mdp[0][5]) == 56

    """
    For c1 = 3, c2 = 5 , s = 3 * 21 + 5 = 68
    with m = -5, c'1 = 8, c'2 = 0
    Garage 1 could end up with either 1 car or 15 car, so 15 possibilties.
    Garage 2 cannot rent out any cars so x2 = 0 but 0 <= y2 <= 6 so 7 possibilities.
    Note: since we have limited probabilities for x1,x2,y1,y2 to be higher than or equal to 0.01
    Possible values are
    0 <= x1 <= 7, 0 <= x2 <= 9, 0 <= y1 <= 7, 0 <= y2 <= 6
    """
    assert len(mdp[68][0]) == 15 * 7
    p_sum = 0
    for p, _, _, _ in mdp[68][0]:
        p_sum += p
    assert p_sum == approx(1, 1.0e-6)


def test_JackCarRentalProblem():
    seed = 11082025
    env = JackCarRentalProblem(seed=seed)
    assert env.reward == 0
    env.step(-5)
    assert env.x1 == 0
    assert env.x2 == 0
    assert env.state == env.y1 * 21 + env.y2
    assert env.reward == 0
    assert env.y1 <= 20
    assert env.y2 <= 20
    for m_order in [3, 5, -2, 0, 0, 4, 9, 10]:
        c1 = env.state // 21
        c2 = env.state % 21
        m = m_order if m_order >= -min(c2, 20 - c1) else -(c2, 20 - c1)
        m = m if m <= min(c1, 20 - c2) else min(c1, 20 - c2)
        r0 = env.reward
        env.step(m_order + 5)  # move 3 from garage 1 to garage 2
        assert env.x1 <= c1 - m
        assert env.x2 <= c2 + m
        assert env.y1 <= 20 - (c1 - m)
        assert env.y2 <= 20 - (c2 + m)
        c1_next = c1 - m - env.x1 + env.y1
        c2_next = c2 + m - env.x2 + env.y2
        assert env.state == c1_next * 21 + c2_next
        assert env.reward == r0 - 2 * np.abs(m) + 10 * (env.x1 + env.x2)
