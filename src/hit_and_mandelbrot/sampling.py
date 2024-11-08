from enum import StrEnum

import numpy as np
from scipy.stats.qmc import LatinHypercube


class Sampler(StrEnum):
    RANDOM = "random"
    LHS = "lhs"


def sample_lhs(xmin, xmax, n):
    normalised_samples = LatinHypercube(d=1).random(n)
    denormed_samples = (xmax - xmin) * (normalised_samples - 0.5)
    return denormed_samples


def sample_complex_uniform(
    n_samples, r_min, r_max, i_min, i_max, method=Sampler.RANDOM
):
    match method:
        case Sampler.RANDOM:
            real_samples = np.random.uniform(r_min, r_max, n_samples)
            imag_samples = np.random.uniform(i_min, i_max, n_samples) * 1.0j
        case Sampler.LHS:
            real_samples = sample_lhs(r_min, r_max, n_samples)
            imag_samples = sample_lhs(i_min, i_max, n_samples) * 1.0j
        case _:
            raise ValueError(f"Unknown sampling method: {method}")
    return real_samples + imag_samples
