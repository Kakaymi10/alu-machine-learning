#!/usr/bin/env python3


import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function.

    Parameters:
    env (gym.Env): The OpenAI environment instance.
    V (numpy.ndarray): The value estimate of shape (s,).
    policy (function): A function that takes in a state and returns the next action to take.
    episodes (int): The total number of episodes to train over.
    max_steps (int): The maximum number of steps per episode.
    alpha (float): The learning rate.
    gamma (float): The discount rate.

    Returns:
    numpy.ndarray: The updated value estimate.
    """
    for episode in range(episodes):
        state = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V