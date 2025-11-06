from models.bellman_eq_methods import *
from environments.mdp_slippery_frozen_lake import *


def test_value_evaluation():
    """
    Based on Morales (2020) page 82
    """
    mdp = get_mdp_frozen_slippery_lake(size=4, hole_pos=[5, 7, 11, 12], slip=1.0 / 3.0)

    def policy(s, a):
        """
        A randomly generated policy. Taken from page 82 of Morales (2020)
        to match with estimated values in that ref.
        """
        go_left = 0
        go_down = 1
        go_right = 2
        go_up = 3
        if (s, a) in [
            (0, go_right),
            (1, go_left),
            (2, go_down),
            (3, go_up),
            (4, go_left),
            (6, go_right),
            (8, go_up),
            (9, go_down),
            (10, go_up),
            (13, go_right),
            (14, go_down),
        ]:
            return 1
        return 0

    gamma = 0.99
    v = policy_evaluation(mdp=mdp, policy=policy, gamma=gamma)
    v_morales = np.array(
        [
            [0.0955, 0.0471, 0.047, 0.0456],
            [0.1469, 0, 0.0498, 0],
            [0.2028, 0.2647, 0.1038, 0],
            [0, 0.4957, 0.7417, 0],
        ]
    )
    assert np.max(np.abs(v - v_morales.reshape(16))) < 5.0e-5


def test_policy_iteration():
    mdp = get_mdp_frozen_slippery_lake(size=4, hole_pos=[5, 7, 11, 12], slip=1.0 / 3.0)
    starting_policy = lambda s, a: 0.25  # Random policy
    _, optimal_policy = policy_iteration(
        mdp=mdp, starting_policy=starting_policy, gamma=0.99
    )
    states_with_optimal_move = [0, 1, 2, 3, 4, 8, 9, 10, 13, 14]
    optimal_moves = [0, 3, 3, 3, 0, 3, 1, 0, 2, 1]
    for i, s in enumerate(states_with_optimal_move):
        for a in range(4):
            assert optimal_policy(s, a) == (1 if a == optimal_moves[i] else 0)


# py -m test.test_bellman_eq_methods
