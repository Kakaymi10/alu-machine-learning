#!/usr/bin/env python3
""" Early stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determine whether to stop training early based on validation cost.
    """
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0

    if count < patience:
        return False, count
    return True, count
