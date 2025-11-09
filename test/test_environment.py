from environments.environment import *


class TestEnviro(Environment):
    def step(self, action: int) -> Self:
        if not self.done:
            self.reward += 1 if action == 1 else 0
            self.episode_number += 1
            self.done = self.episode_number >= 10
        return self


def TestModel(env: Environment) -> None:
    env.reset()
    i = 0
    while not env.done:
        env.step(i % 2)
        i += 1


def test_environment():
    env = TestEnviro()
    env.reward = 1000
    TestModel(env=env)
    assert env.reward == 5
