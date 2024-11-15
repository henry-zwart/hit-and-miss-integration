from functools import lru_cache

import numpy as np


@lru_cache
def load_rng():
    return np.random.default_rng(42)
