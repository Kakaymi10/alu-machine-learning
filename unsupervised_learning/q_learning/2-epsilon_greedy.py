#!/usr/bin/env python3


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Parameters:
    Q (numpy.ndarray): The q-table.
    state (int): The current state.
    epsilon (float): The epsilon to use for the calculation.

    Returns:
    int: The next action index.
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit: choose the action with the highest Q-value
        action = np.argmax(Q[state])
    
    return action