#!/usr/bin/env python3


import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    for episode in range(episodes):
        state = env.reset()
        eligibility_trace = np.zeros_like(V)
        
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            
            delta = reward + gamma * V[next_state] - V[state]
            eligibility_trace[state] += 1
            
            V += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha
            
            if done:
                break
            
            state = next_state
            
    return V