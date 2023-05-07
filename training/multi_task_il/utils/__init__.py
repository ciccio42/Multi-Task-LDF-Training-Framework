import numpy as np


def normalize_action(action, n_action_bin, action_ranges):
    half_action_bin = int(n_action_bin/2)
    norm_action = action.copy()
    # normalize between [-1 , 1]
    norm_action[:-1] = (2 * (norm_action[:-1] - action_ranges[:, 0]) /
                        (action_ranges[:, 1] - action_ranges[:, 0])) - 1
    # action discretization
    return (norm_action * half_action_bin).astype(np.int32).astype(np.float32) / half_action_bin


def denormalize_action(norm_action, action_ranges):
    action = np.clip(norm_action.copy(), -1, 1)
    for d in range(action_ranges.shape[0]):
        action[d] = (0.5 * (action[d] + 1) *
                     (action_ranges[d, 1] - action_ranges[d, 0])) + action_ranges[d, 0]
