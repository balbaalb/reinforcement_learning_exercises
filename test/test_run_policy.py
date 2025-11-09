from environments.run_policy import *


def test_run_policy():
    pi = lambda s, a: 0.1 if a == 0 else (0.7 if a == 1 else 0.2)
    actions = np.zeros(3)
    np.random.seed(42)
    for _ in range(1000):
        a = run_policy(policy=pi, state=0, n_actions=3)
        actions[a] += 1
    print(f"actions = {actions}")
    assert 90 <= actions[0] <= 110
    assert 690 <= actions[1] <= 710


def test_run_policy2():
    def pi(s: int, a: int):
        if s == 0:
            return 0.1 if a == 0 else (0.7 if a == 1 else 0.2)
        else:
            return 0.65 if a == 0 else (0.2 if a == 1 else 0.15)

    actions = np.zeros(3)
    np.random.seed(123)
    for s in range(1000):
        a = run_policy(policy=pi, state=s % 2, n_actions=3)
        actions[a] += 1
    print(f"actions = {actions}")
    assert 365 <= actions[0] <= 385
    assert 440 <= actions[1] <= 460
