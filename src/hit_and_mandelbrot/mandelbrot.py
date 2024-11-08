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


def calculate_hits(
    c: np.ndarray,
    iterations: int,
):
    z = c.copy()
    for _ in range(iterations + 1):  # We want to consider i=x as (0, ..., x)
        still_bounded = np.abs(z) < 2
        z[still_bounded] = np.pow(z[still_bounded], 2) + c[still_bounded]

    return np.abs(z) < 2


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
    cumulative_prop_hits = hits.cumsum(axis=0) / np.arange(1, c.shape[0] + 1)[:, None]

    # Multiply by volume of sample space to get area
    per_sample_area_est = cumulative_prop_hits * v

    # Reduce this to an expected value and ci per sample size
    per_sample_area_exp, per_sample_area_ci = mean_and_ci(
        per_sample_area_est, axis=1, ddof=ddof, z=z
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
    c = sample_complex_uniform(
        n_samples, repeats, x_min, x_max, y_min, y_max, method=sampler
    )
    hits = calculate_hits(c, iterations)

    # Calculate area as the sample space volume multiplied by proportion of hits
    prop_hits = hits.sum(axis=0) / n_samples
    area_estimates = prop_hits * v

    if repeats == 1:
        area_estimates = area_estimates[0]

    return area_estimates
