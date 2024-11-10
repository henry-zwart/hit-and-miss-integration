import time

import numba
import numpy as np

from .sampling import Sampler, sample_complex_uniform
from .statistics import mean_and_ci


def _prepare_params(
    x_min, x_max, y_min, y_max
) -> tuple[float, float, float, float, float]:
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    v = (x_max - x_min) * (y_max - y_min)
    return (x_min, x_max, y_min, y_max, v)


def calculate_hits_slow(
    c: np.ndarray,
    iterations: int,
):
    """
    For each iteration and each sample, determines whether the sample is a hit.

    On each iteration:
        Check if any previously-bounded samples are outside threshold.
        For each such sample, set the entry in "bounded" to False, and set the entry in
        "unbounded_at" to the current iteration.

        For all remaining points in "bounded", iterate the mandelbrot function again.

    At the end:
        For each iteration, record which samples had exceeded the threshold.
    """
    z = c.copy()

    # Keep a record of the samples which are still within bounds
    bounded = np.abs(c) <= 2

    # Record the iteration at which each point exceeds threshold
    unbounded_at = np.full(c.shape, np.inf)
    unbounded_at[~bounded] = 0

    for i in range(1, iterations + 1):  # We want to consider i=x as (0, ..., x)
        # Iterate on the samples which remain bounded
        z[bounded] = z[bounded] ** 2 + c[bounded]

        # Identify the samples which were previously bounded, and now exceed threshold
        just_unbounded = bounded & (np.abs(z) > 2)

        # Record these samples as unbounded
        bounded[just_unbounded] = False
        unbounded_at[just_unbounded] = i

    # For each iteration, record the hits. i.e. the samples which were still bound
    hits = unbounded_at > np.arange(iterations + 1)[:, None, None]

    return hits


@numba.njit
def calculate_hits(c: np.ndarray, iterations: int):
    """
    For each iteration and each sample, determines whether the sample is a hit.
    """
    z = c.copy()

    # Keep a boolean record of the samples which are still within bounds
    bounded = np.abs(c) <= 2

    # Record the iteration at which each point exceeds threshold
    unbounded_at = np.full(c.shape, iterations + 1, dtype=np.int64)

    # Initialise the record for samples which are unbounded at iteration 0
    for s in range(c.shape[0]):
        for r in range(c.shape[1]):
            if not bounded[s, r]:
                unbounded_at[s, r] = 0

    # For each iteration,
    for i in range(1, iterations + 1):
        # ... for each sample,
        for s in range(c.shape[0]):
            # ... in each repeat of the estimation:
            for r in range(c.shape[1]):
                # If the sample was previously within the bounds.
                # Note, this avoids extra computation on already-excluded samples.
                if bounded[s, r]:
                    # Iterate on the sample
                    z[s, r] = z[s, r] ** 2 + c[s, r]
                    # If it is now outside the bounds, record that.
                    if np.abs(z[s, r]) > 2:
                        bounded[s, r] = False
                        unbounded_at[s, r] = i

    # Prepare boolean array to say which samples were included, and at which iterations
    hits = np.zeros((iterations + 1, c.shape[0], c.shape[1]), dtype=np.bool)
    for i in range(iterations + 1):
        for s in range(c.shape[0]):
            for r in range(c.shape[1]):
                hits[i, s, r] = unbounded_at[s, r] > i

    return hits


@numba.njit
def calculate_final_hits(c: np.ndarray, iterations: int):
    """
    Calculate hits without storing intermediary information
    """
    z = c.copy()
    bounded = np.abs(c) <= 2
    for _ in range(1, iterations + 1):
        for s in range(c.shape[0]):
            for r in range(c.shape[1]):
                if bounded[s, r]:
                    z[s, r] = z[s, r] ** 2 + c[s, r]
                    if np.abs(z[s, r]) > 2:
                        bounded[s, r] = False

    return (np.abs(z) <= 2).sum(axis=0)


def estimate_area_per_sample(
    n_samples,
    iterations,
    x_min=-2,
    x_max=2,
    y_min=-2,
    y_max=2,
    repeats=1,
    sampler=Sampler.RANDOM,
    ddof=1,
    z=1.96,
):
    x_min, x_max, y_min, y_max, v = _prepare_params(x_min, x_max, y_min, y_max)
    c = sample_complex_uniform(
        n_samples, repeats, x_min, x_max, y_min, y_max, method=sampler
    )
    hits = calculate_hits(c, iterations)

    # For each repeat, get prop. hits in (0, ..., s) samples, for each s
    cumulative_prop_hits = hits.cumsum(axis=1) / np.arange(1, c.shape[0] + 1)[:, None]

    # Multiply by volume of sample space to get area
    per_sample_area_est = cumulative_prop_hits * v

    # Reduce this to an expected value and ci per sample size
    per_sample_area_exp, per_sample_area_ci = mean_and_ci(
        per_sample_area_est, axis=2, ddof=ddof, z=z
    )

    return (
        per_sample_area_exp,
        per_sample_area_ci,
    )


def estimate_area(
    n_samples,
    iterations,
    x_min=-2,
    x_max=2,
    y_min=-2,
    y_max=2,
    repeats=1,
    sampler=Sampler.RANDOM,
):
    x_min, x_max, y_min, y_max, v = _prepare_params(x_min, x_max, y_min, y_max)
    print("Running Mandelbrot area estimation:")
    print(
        f"\tInterval (xmin, xmax)x(ymin, ymax): ({x_min:.2f}, {x_max:.2f})x({y_min:.2f}, {y_max:.2f})"
    )
    print(f"\tIterations: {iterations}")
    print(f"\tSample size: {n_samples}")
    print(f"\tRepeats: {repeats}")
    print(f"\tSampling method: {str(sampler)}")

    t0 = time.time()
    c = sample_complex_uniform(
        n_samples, repeats, x_min, x_max, y_min, y_max, method=sampler
    )
    hits = calculate_final_hits(c, iterations)

    # Calculate area as the sample space volume multiplied by proportion of hits
    prop_hits = hits / n_samples
    area_estimates = prop_hits * v
    t1 = time.time()

    t = t1 - t0
    print(f"Completed in {t:.2f}s")

    if repeats == 1:
        area_estimates = area_estimates[0]

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
    total_inside_area = estimate_area(n_samples, iterations, x_min, x_max, y_min, y_max)
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
