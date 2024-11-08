import numpy as np

from .sampling import Sampler, sample_complex_uniform


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
    # Calculate area of the sample space
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    v = (x_max - x_min) * (y_max - y_min)

    results = np.zeros(repeats, dtype=np.float64)

    for i in range(repeats):
        # Run mandlebrot iterations
        c0 = sample_complex_uniform(
            n_samples, x_min, x_max, y_min, y_max, method=sampler
        )
        z = c0.copy()
        for _ in range(iterations):
            still_bounded = np.abs(z) < 2
            z[still_bounded] = np.pow(z[still_bounded], 2) + c0[still_bounded]

        # Return the estimated area
        proportion_bounded = (np.abs(z) < 2).sum() / n_samples
        results[i] = v * proportion_bounded

    return results
