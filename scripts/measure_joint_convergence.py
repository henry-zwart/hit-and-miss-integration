import json
from pathlib import Path

import numpy as np

from hit_and_mandelbrot.mandelbrot import estimate_area_per_sample
from hit_and_mandelbrot.sampling import Sampler

if __name__ == "__main__":
    RESULTS_ROOT = Path("data") / "joint_convergence"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    n_samples = 100000
    iterations = 128  # To-do: Set this equal to the minimum iterations for convergence
    repeats = 100
    sampler = Sampler.RANDOM
    ddof = 0
    z = 1.96

    expected_area, confidence_interval = estimate_area_per_sample(
        n_samples,
        iterations,
        repeats=repeats,
        sampler=sampler,
        ddof=ddof,
        z=z,
    )

    np.save(RESULTS_ROOT / "expected_area.npy", expected_area)
    np.save(RESULTS_ROOT / "confidence_intervals.npy", confidence_interval)

    metadata = {
        "max_samples": n_samples,
        "max_iterations": iterations,
        "repeats": repeats,
        "sampling_method": str(sampler),
        "ddof": ddof,
        "z": z,
    }

    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
