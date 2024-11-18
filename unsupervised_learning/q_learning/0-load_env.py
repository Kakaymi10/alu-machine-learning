#!/usr/bin/env python3


import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from OpenAI's gym.

    Parameters:
    - desc: None or a list of lists containing a custom description of the map to load for the environment
    - map_name: None or a string containing the pre-made map to load
    - is_slippery: boolean to determine if the ice is slippery

    Returns:
    - env: the environment
    """
    if desc is not None:
        env = gym.make('FrozenLake-v1', desc=desc, is_slippery=is_slippery)
    elif map_name is not None:
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
    else:
        random_map = generate_random_map(size=8)
        env = gym.make('FrozenLake-v1', desc=random_map, is_slippery=is_slippery)
    
    return env