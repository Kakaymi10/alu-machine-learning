#!/usr/bin/env python3


import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs SARSA(λ)"""
    def epsilon_greedy_policy(state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(env.action_space.n)
        else:
            return np.argmax(Q[state])

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(state, epsilon)
        E = np.zeros_like(Q)
        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(next_state, epsilon)
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1
            
            for s in range(Q.shape[0]):
                for a in range(Q.shape[1]):
                    Q[s, a] += alpha * delta * E[s, a]
                    E[s, a] *= gamma * lambtha
            
            if done:
                break
            
            state, action = next_state, next_action
        
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))
    
    return Q