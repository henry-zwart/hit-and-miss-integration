"""
    Course: Stochastic Simulation
    Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
    Student IDs: 15719227, 15393879, 13392425 
    Assignement: Hit and Miss Integration

    File description:
        File is used to generate random seed.
"""

from functools import lru_cache

import numpy as np


@lru_cache
def load_rng():
    """ Function generates random seed. """
    return np.random.default_rng(42)
