from environments.environment import *


class SampleEnviro(Environment):
    def __init__(self):
        super().__init__()
        self.max_steps = 1000
        self.n_states = 1
        self.n_actions = 2
        self.reset()

    def reset(self):
        self.reward = 0
        self.done = False
        self.step_number = 0

    def step(self, action: int) -> Self:
        if not self.done and self.step_number < self.max_steps:
            self.reward += 1 if action == 1 else 0
            self.step_number += 1
            self.done = self.step_number >= 10
        return self


def TestModel(env: Environment) -> None:
    env.reset()
    i = 0
    while not env.done:
        env.step(i % 2)
        i += 1


def test_environment():
    env = SampleEnviro()
    env.reward = 1000
    TestModel(env=env)
    assert env.reward == 5
