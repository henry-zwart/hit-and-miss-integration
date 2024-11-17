import json
import random
from pathlib import Path

import numpy as np
from tqdm import trange

from hit_and_mandelbrot import mean_and_ci
from hit_and_mandelbrot.mandelbrot import (
    adaptive_sampling,
)
from hit_and_mandelbrot.random_seed import load_rng

if __name__ == "__main__":
    # Set random seeds
    load_rng()
    random.seed(42)

    # Establish directory to write results
    RESULTS_ROOT = Path("data") / "sample_adaptive"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with (Path("data") / "shape_convergence/metadata.json").open("r") as f:
        convergent_iters = json.load(f)["min_convergent_iters"]

    # Parameters
    # iterations = 3069
    iterations = convergent_iters
    repeats = 30
    threshold = 0.05
    max_depth = 8
    ddof = 1
    z = 1.96
    initial_samples = [
        500,
        1000,
        2000,
        3000,
        4000,
        5000,
        7000,
        9000,
        11000,
        13000,
        15000,
        17000,
        19000,
        21000,
        23000,
        25000,
    ]

    area = np.zeros((repeats, len(initial_samples)))
    total_samples = np.zeros((repeats, len(initial_samples)))
    for i in trange(repeats):
        area_repeat = np.zeros(len(initial_samples))
        samples_repeat = np.zeros(len(initial_samples))
        for j, n_samples in enumerate(initial_samples):
            area_estimate, count_samples = adaptive_sampling(
                n_samples,
                iterations,
                x_min=-2,
                x_max=2,
                y_min=-2,
                y_max=2,
                threshold=threshold,
                max_depth=max_depth,
            )
            area_repeat[j] = area_estimate[0]
            samples_repeat[j] = count_samples
        area[i] = area_repeat
        total_samples[i] = samples_repeat

    expected_area, confidence_interval = mean_and_ci(area, ddof=ddof, z=z)
    confidence_interval *= np.sqrt(repeats)

    np.save(RESULTS_ROOT / "expected_area.npy", expected_area)
    np.save(RESULTS_ROOT / "confidence_intervals.npy", confidence_interval)
    np.save(RESULTS_ROOT / "area.npy", area)
    np.save(RESULTS_ROOT / "total_samples.npy", total_samples)
    metadata = {
        "max_samples": np.max(total_samples),
        "iterations": iterations,
        "repeats": repeats,
        "initial_samples": initial_samples,
        "sampling_method": "Adaptive_sampling",
        "ddof": ddof,
        "z": z,
    }
    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
