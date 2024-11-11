from enum import StrEnum

import numpy as np
from scipy.stats.qmc import LatinHypercube


class Sampler(StrEnum):
    RANDOM = "random"
    LHS = "lhs"
    ORTHO = "ortho"


def sample_lhs(xmin, xmax, ymin, ymax, n, repeats, strength=1):
    # Sample from [0,1)
    lhs = LatinHypercube(d=2, strength=strength)
    normalised_real_samples = np.stack(
        [lhs.random(n) for _ in range(repeats)],
        axis=2,
    )

    # Scale up to [xmin, xmax) and [ymin, ymax)
    scaled_real_samples = normalised_real_samples
    scaled_real_samples[:, 0] = scaled_real_samples[:, 0] * (xmax - xmin) + xmin
    scaled_real_samples[:, 1] = scaled_real_samples[:, 1] * (ymax - ymin) + ymin

    # Make the second component imaginary, and sum to get complex samples
    complex_2d_samples = scaled_real_samples.astype(np.complex128)
    complex_2d_samples[:, 1] *= 1.0j
    complex_samples = complex_2d_samples.sum(axis=1)

    return complex_samples


def sample_complex_uniform(
    n_samples,
    repeats,
    r_min=-2,
    r_max=2,
    i_min=-2,
    i_max=2,
    method=Sampler.RANDOM,
):
    match method:
        case Sampler.RANDOM:
            real_samples = np.random.uniform(r_min, r_max, (n_samples, repeats))
            imag_samples = np.random.uniform(i_min, i_max, (n_samples, repeats)) * 1.0j
            samples = real_samples + imag_samples
        case Sampler.LHS:
            samples = sample_lhs(
                r_min, r_max, i_min, i_max, n_samples, repeats, strength=1
            )
        case Sampler.ORTHO:
            samples = sample_lhs(
                r_min, r_max, i_min, i_max, n_samples, repeats, strength=2
            )
        case _:
            raise ValueError(f"Unknown sampling method: {method}")
    return samples
