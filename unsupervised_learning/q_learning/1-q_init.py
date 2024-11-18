#!/usr/bin/env python3


import numpy as np


def q_init(env):
    """
    Initializes the Q-table.

    Parameters:
    env (FrozenLakeEnv): The FrozenLakeEnv instance

    Returns:
    numpy.ndarray: The Q-table as a numpy.ndarray of zeros
    """
    action_size = env.action_space.n
    state_size = env.observation_space.n
    Q_table = np.zeros((state_size, action_size))
    return Q_table