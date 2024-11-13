import json
from pathlib import Path

import numpy as np

from hit_and_mandelbrot.mandelbrot import Sampler, est_area
from hit_and_mandelbrot.statistics import mean_and_ci

np.random.seed(39)


if __name__ == "__main__":
    RESULTS_ROOT = Path("data") / "joint_convergence"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    with (Path("data") / "shape_convergence/metadata.json").open("r") as f:
        convergent_iters = json.load(f)["min_convergent_iters"]

    n_samples = 100000
    convergent_iters = 256
    iterations = np.arange(
        convergent_iters, step=10
    )  # To-do: Set this equal to the minimum iterations for convergence
    repeats = 100
    sampler = Sampler.RANDOM
    ddof = 1
    z = 1.96

    area = est_area(
        n_samples,
        iterations,
        repeats=repeats,
        sampler=sampler,
        per_sample=True,
    )
    print(area.shape)
    expected_area, confidence_interval = mean_and_ci(area, ddof=ddof, z=z)

    np.save(RESULTS_ROOT / "expected_area.npy", expected_area)
    np.save(RESULTS_ROOT / "confidence_intervals.npy", confidence_interval)

    metadata = {
        "max_samples": n_samples,
        "iterations": iterations.tolist(),
        "repeats": repeats,
        "sampling_method": str(sampler),
        "ddof": ddof,
        "z": z,
    }

    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
