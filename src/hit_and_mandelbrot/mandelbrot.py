import time

import numba
import numpy as np

from .hits import (
    calculate_hits,
    calculate_iter_hits,
    calculate_sample_hits,
    calculate_sample_iter_hits,
)
from .sampling import Sampler, Samples, sample_complex_uniform


def _prepare_params(
    x_min, x_max, y_min, y_max
) -> tuple[float, float, float, float, float]:
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    v = (x_max - x_min) * (y_max - y_min)
    return (x_min, x_max, y_min, y_max, v)


@numba.njit
def in_mandelbrot(c: np.array, iterations: int):
    """Determine whether an array of values are in Mandelbrot after given iterations."""
    z = c.copy()
    bounded = np.abs(c) <= 2
    for _ in range(1, iterations + 1):
        for s in range(c.shape[0]):
            if bounded[s]:
                z[s] = z[s] ** 2 + c[s]
                if np.abs(z[s]) > 2:
                    bounded[s] = False
    return bounded


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
        n_samples = samples.c.shape[0]
        repeats = samples.c.shape[1]

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
            n_samples, repeats, x_min, x_max, y_min, y_max, method=sampler
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
            hits_count = hits.cumsum(axis=0)
            proportion_hits = hits_count / np.arange(1, samples.c.shape[0] + 1)[:, None]
        case (False, True):  # Proportion per-iteration
            # Divide count by number of samples
            hits_count = calculate_iter_hits(samples.c, iterations)
            proportion_hits = hits_count / n_samples
        case (True, True):  # Proportion per-iteration and per-sample
            # As in (True, False) case
            hits = calculate_sample_iter_hits(samples.c, iterations)
            hits_count = hits.cumsum(axis=1)
            proportion_hits = hits_count / np.arange(1, samples.c.shape[0] + 1)[:, None]

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
