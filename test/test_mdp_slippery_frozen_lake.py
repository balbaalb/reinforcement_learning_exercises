from slippery_frozen_lake.mdp_slippery_frozen_lake import *


def test_get_mdp_frozen_slippery_lake():
    size = 3
    hole_pos = [4]
    slip = 0.2  # if action = go-right, there is slip probability that the players go up or down instead!
    mdp = get_mdp_frozen_slippery_lake(size=size, hole_pos=hole_pos, slip=slip)
    assert set(mdp[0][0]) == set([(0.2, 3, 0, False), (0.8, 0, 0, False)])
    assert set(mdp[0][1]) == set(
        [(0.6, 3, 0, False), (0.2, 1, 0, False), (0.2, 0, 0, False)]
    )
    assert set(mdp[0][2]) == set(
        [(0.2, 3, 0, False), (0.6, 1, 0, False), (0.2, 0, 0, False)]
    )
    assert set(mdp[0][3]) == set([(0.2, 1, 0, False), (0.8, 0, 0, False)])
    assert set(mdp[1][0]) == set(
        [(0.6, 0, 0, False), (0.2, 4, 0, True), (0.2, 1, 0, False)]
    )
    assert set(mdp[1][1]) == set(
        [(0.2, 0, 0, False), (0.6, 4, 0, True), (0.2, 2, 0, False)]
    )
    assert set(mdp[1][2]) == set(
        [(0.2, 4, 0, True), (0.6, 2, 0, False), (0.2, 1, 0, False)]
    )
    assert set(mdp[1][3]) == set(
        [(0.2, 0, 0, False), (0.2, 2, 0, False), (0.6, 1, 0, False)]
    )
    assert set(mdp[2][0]) == set(
        [(0.6, 1, 0, False), (0.2, 5, 0, False), (0.2, 2, 0, False)]
    )
    assert set(mdp[2][1]) == set(
        [(0.2, 1, 0, False), (0.6, 5, 0, False), (0.2, 2, 0, False)]
    )
    assert set(mdp[2][2]) == set([(0.2, 5, 0, False), (0.8, 2, 0, False)])
    assert set(mdp[2][3]) == set([(0.2, 1, 0, False), (0.8, 2, 0, False)])
    assert set(mdp[3][0]) == set(
        [(0.2, 6, 0, False), (0.2, 0, 0, False), (0.6, 3, 0, False)]
    )
    assert set(mdp[3][1]) == set(
        [(0.6, 6, 0, False), (0.2, 4, 0, True), (0.2, 3, 0, False)]
    )
    assert set(mdp[3][2]) == set(
        [(0.2, 6, 0, False), (0.6, 4, 0, True), (0.2, 0, 0, False)]
    )
    assert set(mdp[3][3]) == set(
        [(0.2, 4, 0, True), (0.6, 0, 0, False), (0.2, 3, 0, False)]
    )
    assert set(mdp[4][0]) == set([(1.0, 4, 0.0, True)])
    assert set(mdp[4][1]) == set([(1.0, 4, 0.0, True)])
    assert set(mdp[4][2]) == set([(1.0, 4, 0.0, True)])
    assert set(mdp[4][3]) == set([(1.0, 4, 0.0, True)])
    assert set(mdp[5][0]) == set(
        [(0.6, 4, 0, True), (0.2, 8, 1.0, True), (0.2, 2, 0, False)]
    )
    assert set(mdp[5][1]) == set(
        [(0.2, 4, 0, True), (0.6, 8, 1.0, True), (0.2, 5, 0, False)]
    )
    assert set(mdp[5][2]) == set(
        [(0.2, 8, 1.0, True), (0.2, 2, 0, False), (0.6, 5, 0, False)]
    )
    assert set(mdp[5][3]) == set(
        [(0.2, 4, 0, True), (0.6, 2, 0, False), (0.2, 5, 0, False)]
    )
    assert set(mdp[6][0]) == set([(0.2, 3, 0, False), (0.8, 6, 0, False)])
    assert set(mdp[6][1]) == set([(0.2, 7, 0, False), (0.8, 6, 0, False)])
    assert set(mdp[6][2]) == set(
        [(0.6, 7, 0, False), (0.2, 3, 0, False), (0.2, 6, 0, False)]
    )
    assert set(mdp[6][3]) == set(
        [(0.2, 7, 0, False), (0.6, 3, 0, False), (0.2, 6, 0, False)]
    )
    assert set(mdp[7][0]) == set(
        [(0.6, 6, 0, False), (0.2, 4, 0, True), (0.2, 7, 0, False)]
    )
    assert set(mdp[7][1]) == set(
        [(0.2, 6, 0, False), (0.2, 8, 1.0, True), (0.6, 7, 0, False)]
    )
    assert set(mdp[7][2]) == set(
        [(0.6, 8, 1.0, True), (0.2, 4, 0, True), (0.2, 7, 0, False)]
    )
    assert set(mdp[7][3]) == set(
        [(0.2, 6, 0, False), (0.2, 8, 1.0, True), (0.6, 4, 0, True)]
    )
    assert set(mdp[8][0]) == set([(1.0, 8, 0.0, True)])
    assert set(mdp[8][1]) == set([(1.0, 8, 0.0, True)])
    assert set(mdp[8][2]) == set([(1.0, 8, 0.0, True)])
    assert set(mdp[8][3]) == set([(1.0, 8, 0.0, True)])


def test_slippery_frozen_lake_step():
    game = SlipperyFrozenLake(size=4, hole_pos=[5, 9], slip=0)
    assert game.state == 0
    assert game.reward == 0
    assert game.done == False
    game.step(1)
    assert game.state == 4
    assert game.reward == 0
    assert game.done == False
    game.step(1)
    assert game.state == 8
    assert game.reward == 0
    assert game.done == False
    game.step(2)
    assert game.state == 9
    assert game.reward == 0
    assert game.done == True
    game.step(0)
    assert game.state == 9
    assert game.reward == 0
    assert game.done == True
    game = SlipperyFrozenLake(size=4, hole_pos=[5, 9], slip=0)
    game.step(0).step(2).step(2).step(3).step(1).step(1)
    assert game.state == 10
    assert game.reward == 0
    assert game.done == False
    game.step(2).step(1)
    assert game.state == 15
    assert game.reward == 1
    assert game.done == True
    game.step(2)
    assert game.state == 15
    assert game.reward == 0
    assert game.done == True
