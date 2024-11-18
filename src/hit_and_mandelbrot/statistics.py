"""
    Course: Stochastic Simulation
    Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
    Student IDs: 15719227, 15393879, 13392425 
    Assignement: Hit and Miss Integration

    File description:
        File to calculate statistics.
"""

import numpy as np


def mean_and_ci(arr, z=1.96, ddof=1, axis=0):
    """ Function that calculates statistics. """
    return (
        arr.mean(axis=axis),
        z * arr.std(axis=axis, ddof=ddof) / np.sqrt(arr.shape[axis]),
    )
