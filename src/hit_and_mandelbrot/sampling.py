from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.stats.qmc import LatinHypercube
from tqdm import trange


class Sampler(StrEnum):
    RANDOM = "random"
    LHS = "lhs"
    ORTHO = "ortho"


@dataclass
class Samples:
    real_lims: tuple[float, float]
    imag_lims: tuple[float, float]
    space_vol: float
    c: np.array


def sample_lhs(xmin, xmax, ymin, ymax, n, repeats, strength=1, quiet=False):
    # Sample from [0,1)
    lhs = LatinHypercube(d=2, strength=strength)
    range_fn = range if quiet else trange
    normalised_real_samples = np.stack([lhs.random(n) for _ in range_fn(repeats)])
    np.random.shuffle(normalised_real_samples)

    # Scale up to [xmin, xmax) and [ymin, ymax)
    scaled_real_samples = normalised_real_samples
    scaled_real_samples[..., 0] = scaled_real_samples[..., 0] * (xmax - xmin) + xmin
    scaled_real_samples[..., 1] = scaled_real_samples[..., 1] * (ymax - ymin) + ymin

    # Make the second component imaginary, and sum to get complex samples
    complex_2d_samples = scaled_real_samples.astype(np.complex128)
    complex_2d_samples[..., 1] *= 1.0j
    complex_samples = complex_2d_samples.sum(axis=-1)
    return complex_samples


def sample_complex_uniform(
    n_samples,
    repeats,
    r_min=-2,
    r_max=2,
    i_min=-2,
    i_max=2,
    method=Sampler.RANDOM,
    quiet=False,
):
    # Ensure coordinates are such that min <= max
    if r_min > r_max:
        r_min, r_max = r_max, r_min
    if i_min > i_max:
        i_min, i_max = i_max, i_min

    if method != "adaptive":
        space_vol = (r_max - r_min) * (i_max - i_min)

    match method:
        case Sampler.RANDOM:
            real_samples = np.random.uniform(r_min, r_max, (repeats, n_samples))
            imag_samples = np.random.uniform(i_min, i_max, (repeats, n_samples)) * 1.0j
            samples = real_samples + imag_samples
        case Sampler.LHS:
            samples = sample_lhs(
                r_min,
                r_max,
                i_min,
                i_max,
                n_samples,
                repeats,
                strength=1,
                quiet=quiet,
            )
        case Sampler.ORTHO:
            samples = sample_lhs(
                r_min,
                r_max,
                i_min,
                i_max,
                n_samples,
                repeats,
                strength=2,
                quiet=quiet,
            )
        case _:
            raise ValueError(f"Unknown sampling method: {method}")

    return Samples(
        real_lims=(r_min, r_max),
        imag_lims=(i_min, i_max),
        space_vol=space_vol,
        c=samples,
    )
