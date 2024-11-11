import json
from pathlib import Path

import numpy as np

from hit_and_mandelbrot.mandelbrot import est_area
from hit_and_mandelbrot.sampling import Sampler
from hit_and_mandelbrot.statistics import mean_and_ci

if __name__ == "__main__":
    RESULTS_ROOT = Path("data") / "sample_convergence"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    MAX_SAMPLES = 101**2  # a little prime
    # MAX_SAMPLES = 1051**2  # a big prime
    REPEATS = 300
    z = 1.96
    ddof = 1

    # Load data from the iteration convergence experiment
    SHAPE_CONVERGENCE_RESULTS_ROOT = Path("data") / "shape_convergence"
    with (SHAPE_CONVERGENCE_RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    # Determine the minimum iteration number we recorded as "convergent"
    min_convergent_iters = metadata["min_convergent_iters"]

    # Calculate per-sample-size area and CI for each sampling algorithm
    for sampler in Sampler:
        area = est_area(
            MAX_SAMPLES,
            min_convergent_iters,
            repeats=REPEATS,
            per_sample=True,
            sampler=sampler,
        )
        expected_area, ci = mean_and_ci(area, z=z, ddof=ddof, axis=1)
        np.save(RESULTS_ROOT / f"{sampler}_area.npy", expected_area)
        np.save(RESULTS_ROOT / f"{sampler}_ci.npy", ci)
        np.save(RESULTS_ROOT / f"{sampler}_sample_size.npy", np.arange(MAX_SAMPLES))

    metadata = {
        "max_samples": MAX_SAMPLES,
        "repeats": REPEATS,
        "z": z,
        "ddof": ddof,
        "iters": min_convergent_iters,
    }
    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
