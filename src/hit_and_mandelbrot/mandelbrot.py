import time
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import numpy as np
from scipy.stats.qmc import LatinHypercube
from tqdm import trange

from hit_and_mandelbrot.statistics import mean_and_ci

from .hits import (
    calculate_hits,
    calculate_iter_hits,
    calculate_sample_hits,
    calculate_sample_iter_hits,
)


class Sampler(StrEnum):
    RANDOM = "random"
    LHS = "lhs"
    ORTHO = "ortho"
    IMPROVED = "improved"


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


@lru_cache()
def est_outer_area(
    sample_size: int,
    iterations: int,
    repeats: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    area = est_area(
        sample_size,
        iterations=iterations,
        repeats=repeats,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        sampler=Sampler.ORTHO,
        quiet=True,
    )
    exp_outer_area, outer_area_ci = mean_and_ci(area, z=2.326)
    assert 100 * (outer_area_ci / exp_outer_area) < 0.1, "CI exceeds 0.1%"
    return exp_outer_area


def sample_improved(
    outer_xmin,
    outer_xmax,
    outer_ymin,
    outer_ymax,
    n_samples,
    repeats,
    approx_iters=3,
    quiet=False,
):
    OUTER_SAMPLE_SIZE = 131**2
    # OUTER_SAMPLE_SIZE = 47**2
    outer_area = est_outer_area(
        OUTER_SAMPLE_SIZE,
        approx_iters,
        30,
        outer_xmin,
        outer_xmax,
        outer_ymin,
        outer_ymax,
    )

    collected_samples = []
    for _ in range(repeats):
        repeat_samples = []
        while (remaining := n_samples - len(repeat_samples)) > 0:
            # Sample points uniformly on rectangle
            print(remaining)
            samples = sample_complex_uniform(remaining, repeats=1, quiet=True).c

            # Filter out any which aren't hits on the outer shape
            outer_hits = calculate_sample_hits(samples, approx_iters)
            reduced_samples = samples[outer_hits]

            # Add the remaining ones to our list of samples
            repeat_samples.extend(reduced_samples)
        collected_samples.append(repeat_samples)
    collected_samples = np.array(collected_samples)

    return collected_samples, outer_area


def sample_complex_uniform(
    n_samples,
    repeats,
    r_min=-2,
    r_max=2,
    i_min=-2,
    i_max=2,
    method=Sampler.RANDOM,
    quiet=False,
    **kwargs,
):
    # Ensure coordinates are such that min <= max
    if r_min > r_max:
        r_min, r_max = r_max, r_min
    if i_min > i_max:
        i_min, i_max = i_max, i_min

    if method not in ("adaptive", "improved"):
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
        case Sampler.IMPROVED:
            samples, space_vol = sample_improved(
                r_min,
                r_max,
                i_min,
                i_max,
                n_samples,
                repeats,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unknown sampling method: {method}")

    return Samples(
        real_lims=(r_min, r_max),
        imag_lims=(i_min, i_max),
        space_vol=space_vol,
        c=samples,
    )


def est_area(
    n_samples: int | None = None,
    iterations: int | None = None,
    x_min: float = -2.0,
    x_max: float = 2.0,
    y_min: float = -2.0,
    y_max: float = 2.0,
    repeats: int = 1,
    sampler: Sampler | None = None,
    samples: Samples | None = None,
    per_sample: bool = False,
    per_iter: bool = False,
    quiet: bool = False,
):
    # x_min, x_max, y_min, y_max, v = _prepare_params(x_min, x_max, y_min, y_max)
    assert (samples is not None) or not any((n_samples is None, sampler is None))

    if samples is not None:
        repeats = samples.c.shape[0]
        n_samples = samples.c.shape[1]

    if not quiet:
        print("Running Mandelbrot area estimation:")
        print(
            f"\tInterval (xmin, xmax)x(ymin, ymax): ({x_min:.2f}, {x_max:.2f})x({y_min:.2f}, {y_max:.2f})"
        )
        print(f"\tIterations: {iterations}")
        print(f"\tSample size: {n_samples}")
        print(f"\tRepeats: {repeats}")
        print(f"\tSampling method: {str(sampler)}")

    t0 = time.time()
    if samples is None:
        samples = sample_complex_uniform(
            n_samples,
            repeats,
            x_min,
            x_max,
            y_min,
            y_max,
            method=sampler,
            quiet=quiet,
        )
    else:
        print("Samples provided, ignoring sampling parameters.")

    match (per_sample, per_iter):
        case (False, False):  # Just record the final proportion
            # Divide number of hits by number of samples
            hits_count = calculate_hits(samples.c, iterations)
            proportion_hits = hits_count / n_samples
        case (True, False):  # Proportion per individual sample
            # Calculate cumulative sum over samples, divide by 1 + idx to get
            #   per-sample proportion
            hits = calculate_sample_hits(samples.c, iterations)
            hits_count = hits.cumsum(axis=-1)
            proportion_hits = hits_count / np.arange(1, samples.c.shape[1] + 1)
        case (False, True):  # Proportion per-iteration
            # Divide count by number of samples
            hits_count = calculate_iter_hits(samples.c, iterations)
            proportion_hits = hits_count / n_samples
        case (True, True):  # Proportion per-iteration and per-sample
            # As in (True, False) case
            hits = calculate_sample_iter_hits(samples.c, iterations)
            hits_count = hits.cumsum(axis=-1)
            proportion_hits = hits_count / np.arange(1, samples.c.shape[1] + 1)

    # Calculate area as the sample space volume multiplied by proportion of hits
    area_estimates = proportion_hits * samples.space_vol
    t1 = time.time()

    t = t1 - t0
    if not quiet:
        print(f"Completed in {t:.2f}s")

    return area_estimates


def adaptive_sampling(
    n_samples,
    iterations,
    x_min,
    x_max,
    y_min,
    y_max,
    threshold,
    max_depth,
    cur_depth=0,
):
    """
    This is an adaptation to the random sampling Monte carlo technique.
    The function works recursively.

    Algorithm:
    1. It starts with a (sub)grid
    2. It takes n random sampling points within the subgrid
    3. It calculates the balance of hits and misses
    4. We check if the max depth is reached of our recursion,
        if so -> area subgrid is returned
        if not -> step 5
    5. - If the balance is disproportionate:
            the subgrid is mostly entirely inside or outside the Mandelbrot set area
            -> thus, we return the area of the subgrid
       - If the balance is proportionate:
            the subgrid includes an edge of the Mandelbrot set area (most probably)
            -> thus, we want to zoom in on this subgrid,
               we recursively call this function with 4 subgrids of the current grid
    """
    # Counts the proportion of sample points that is inside of the Mandelbrot set area
    v = (x_max - x_min) * (y_max - y_min)
    total_inside_area = est_area(n_samples, iterations, x_min, x_max, y_min, y_max)
    hits_proportion = total_inside_area / v

    # When max depth is reached the area of the current grid is returned.
    # This makes sure the algorithm doesn't go on forever.
    if cur_depth == max_depth:
        return v * hits_proportion

    # Checks balance hits and misses
    # If the balance is disproportionate (mostly hits or mostly misses):
    #         the subgrid is mostly entirely inside or outside the Mandelbrot set area
    #         -> thus, we return the area of the subgrid
    if hits_proportion < threshold or hits_proportion > 1 - threshold:
        return v * hits_proportion

    # recursively calculates area for 4 subgrids
    mid_x, mid_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    area = 0
    quadrants = (
        ((x_min, mid_x), (y_min, mid_y)),  # Bottom left
        ((mid_x, x_max), (y_min, mid_y)),  # Bottom right
        ((x_min, mid_x), (mid_y, y_max)),  # Top left
        ((mid_x, x_max), (mid_y, y_max)),  # Top right
    )
    for (x1, x2), (y1, y2) in quadrants:
        area += adaptive_sampling(
            iterations, n_samples, x1, x2, y1, y2, threshold, max_depth, cur_depth + 1
        )

    return area
