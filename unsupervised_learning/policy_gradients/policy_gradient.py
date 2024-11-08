#!/usr/bin/env python3
'''
Policy Gradient
'''


import numpy as np


def policy(state, weight):
    '''Function that computes to policy with a weight of a matrix'''
    z = state.dot(weight)
    exp = np.exp(z)
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    '''Function that computes the Monte-Carlo policy gradient'''
    probs = policy(state, weight)
    action = np.random.choice(len(probs[0]), p=probs[0])
    dsoftmax = probs.copy()
    dsoftmax[0, action] -= 1
    gradient = state.T.dot(dsoftmax)
    return action, gradient


