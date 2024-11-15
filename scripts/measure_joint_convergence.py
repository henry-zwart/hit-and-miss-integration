import json
import random
from pathlib import Path

import numpy as np

from hit_and_mandelbrot.mandelbrot import Sampler, est_area
from hit_and_mandelbrot.random_seed import load_rng
from hit_and_mandelbrot.statistics import mean_and_ci

if __name__ == "__main__":
    # Set random seeds
    load_rng()
    random.seed(42)

    # Establish directory to write results
    RESULTS_ROOT = Path("data") / "joint_convergence"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with (Path("data") / "shape_convergence/metadata.json").open("r") as f:
        convergent_iters = json.load(f)["min_convergent_iters"]

    # Parameters
    n_samples = 100000
    iterations = np.arange(
        convergent_iters, step=10
    )  # To-do: Set this equal to the minimum iterations for convergence
    repeats = 50
    sampler = Sampler.RANDOM
    ddof = 1
    z = 1.96

    # Calculate per-sample-size, per-iteration area of Mandelbrot
    area = est_area(
        n_samples,
        iterations,
        repeats=repeats,
        sampler=sampler,
        per_sample=True,
    )

    # Calculate expected area and CI for each iteration and sample-size
    expected_area, confidence_interval = mean_and_ci(area, ddof=ddof, z=z)

    # Save results and experiment metadata
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
