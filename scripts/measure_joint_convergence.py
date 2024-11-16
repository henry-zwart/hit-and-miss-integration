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
        shape_conv_meta = json.load(f)
        convergent_iters = shape_conv_meta["min_convergent_iters"]
        target_area = shape_conv_meta["best_estimate_area"]

    # Parameters
    N_SAMPLES = 450000
    ITER_STEP_SIZE = 25
    ITERATIONS = np.arange(convergent_iters, step=ITER_STEP_SIZE)
    REPEATS = 30
    SAMPLER = Sampler.RANDOM
    DDOF = 1
    Z = 1.96

    # Calculate per-sample-size, per-iteration area of Mandelbrot
    area = est_area(
        N_SAMPLES,
        ITERATIONS,
        repeats=REPEATS,
        sampler=SAMPLER,
        per_sample=True,
    )

    # Calculate expected area and CI for each iteration and sample-size
    expected_area, confidence_interval = mean_and_ci(area, ddof=DDOF, z=Z)

    # Calculate expected error (against A_M), and CIs
    error = np.abs(area - target_area)
    expected_err, err_ci = mean_and_ci(error, ddof=DDOF, z=Z)

    # Save results and experiment metadata
    # np.save(RESULTS_ROOT / "expected_area.npy", expected_area)
    # np.save(RESULTS_ROOT / "confidence_intervals.npy", confidence_interval)
    np.save(RESULTS_ROOT / "expected_err.npy", expected_err)
    np.save(RESULTS_ROOT / "err_confidence.npy", err_ci)
    metadata = {
        "max_samples": N_SAMPLES,
        "iterations": ITERATIONS.tolist(),
        "iter_step_size": ITER_STEP_SIZE,
        "repeats": REPEATS,
        "sampling_method": str(SAMPLER),
        "ddof": DDOF,
        "z": Z,
    }
    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
