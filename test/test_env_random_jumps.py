from environments.env_random_jumps import *


def test_RandomJumps():
    env = RandomJumps(slip=0)
    for _ in range(5):
        env.step(3)
    assert env.done
    assert env.reward == 0
    assert env.gain == 0
    assert env.state == -5
    env.reset()
    env.step(1).step(0).step(1).step(0)
    assert env.done
    assert env.reward == 1
    assert env.gain == 1
    env.step(1)
    assert env.done
    assert env.reward == 0
    assert env.gain == 1
