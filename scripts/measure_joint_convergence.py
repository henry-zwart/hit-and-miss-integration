import json
from pathlib import Path

import numpy as np

from hit_and_mandelbrot.mandelbrot import est_area
from hit_and_mandelbrot.sampling import Sampler
from hit_and_mandelbrot.statistics import mean_and_ci

np.random.seed(39)


def measure_equiv_errors(area, ε_min, ε_max, steps, repeats, z, ddof):
    # Get error due to finite iterations, and error due to finite samples
    ε_i = area[..., -1] - area[-1, -1]
    ε_s = area[-1, ...] - area[-1, -1]
    ε_measure = np.linspace(ε_min, ε_max, steps)
    corresponding_i = np.argmax(ε_i < ε_measure[:, None], axis=1)
    corresponding_s = np.argmax(ε_s < ε_measure[:, None], axis=1)
    corresponding_areas = []
    for i, s in zip(corresponding_i, corresponding_s):
        a = est_area(s, i, repeats=repeats, quiet=True)
        expected_area, _ = mean_and_ci(a, z, ddof)
        corresponding_areas.append(expected_area)
    return (ε_measure, corresponding_areas)


if __name__ == "__main__":
    RESULTS_ROOT = Path("data") / "joint_convergence"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    n_samples = 100000
    iterations = 128  # To-do: Set this equal to the minimum iterations for convergence
    repeats = 100
    sampler = Sampler.RANDOM
    ddof = 0
    z = 1.96

    equiv_ε_min = 0.001
    equiv_ε_max = 0.003
    equiv_ε_steps = 100

    area = est_area(
        n_samples,
        iterations,
        repeats=repeats,
        sampler=sampler,
        per_sample=True,
        per_iter=True,
    )
    expected_area, confidence_interval = mean_and_ci(area, axis=2, ddof=ddof, z=z)

    np.save(RESULTS_ROOT / "expected_area.npy", expected_area)
    np.save(RESULTS_ROOT / "confidence_intervals.npy", confidence_interval)

    ε_measure, ε_equiv_areas = measure_equiv_errors(
        expected_area, equiv_ε_min, equiv_ε_max, equiv_ε_steps, repeats, z, ddof
    )

    np.save(RESULTS_ROOT / "equiv_errors_measured.npy", ε_measure)
    np.save(RESULTS_ROOT / "equiv_error_areas.npy", ε_equiv_areas)

    metadata = {
        "max_samples": n_samples,
        "max_iterations": iterations,
        "repeats": repeats,
        "sampling_method": str(sampler),
        "ddof": ddof,
        "z": z,
        "equiv_err_min": equiv_ε_min,
        "equiv_err_max": equiv_ε_max,
        "equiv_err_steps": equiv_ε_steps,
    }

    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
