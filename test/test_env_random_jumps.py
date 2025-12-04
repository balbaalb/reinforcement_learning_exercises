from pytest import approx
from environments.env_random_jumps import *


def test_RandomJumps_continous():
    env = RandomJumps(slip=0)
    for _ in range(5):
        env.step(3)
    assert env.done
    assert env.reward == 0
    assert env.gain == 0
    assert env.state == -5.0
    env.reset()
    env.step(1)
    assert env.state == 1.0
    for _ in range(5):
        env.step(0)
    assert env.state == approx(1.5)
    env.step(1)
    assert env.state == approx(2.5)
    for _ in range(5):
        env.step(0)
    assert env.state == approx(3.0)
    assert env.done
    assert env.reward == 1
    assert env.gain == 1
    env.step(1)
    assert env.done
    assert env.reward == 0
    assert env.gain == 1


def test_RandomJumps_discrete():
    env = RandomJumps(slip=0, mode=RandomJumps.MODE.DISCRETE)
    assert env.state == 50
    for _ in range(5):
        env.step(3)
    assert env.done
    assert env.reward == 0
    assert env.gain == 0
    assert env.state == 0
    env.reset()
    assert env.state == 50
    env.step(1)
    assert env.state == 60
    for _ in range(5):
        env.step(0)
    assert env.state == 65
    env.step(1)
    assert env.state == 75
    for _ in range(5):
        env.step(0)
    assert env.state == 80
    assert env.done
    assert env.reward == 1
    assert env.gain == 1
    env.step(1)
    assert env.state == 80
    assert env.done
    assert env.reward == 0
    assert env.gain == 1
