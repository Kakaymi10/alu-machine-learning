#!/usr/bin/env python3


import numpy as np


def play(env, Q, max_steps=100):
    state = env.reset()
    env.render()
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        env.render()
        total_rewards += reward
        state = new_state
        if done:
            break

    return total_rewards