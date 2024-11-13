import json
import random
from pathlib import Path

import numpy as np
from sympy import sieve
from tqdm import tqdm

from hit_and_mandelbrot.mandelbrot import Sampler, est_area
from hit_and_mandelbrot.statistics import mean_and_ci

if __name__ == "__main__":
    # Random seeds
    np.random.seed(42)
    random.seed(42)

    # Establish directory to write results
    RESULTS_ROOT = Path("data") / "sample_convergence"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Parameters
    sieve.extend(367)
    primes = np.array(sieve._list)
    sample_sizes = primes**2
    MAX_SAMPLES_IDX = {
        Sampler.RANDOM: len(sample_sizes),
        Sampler.LHS: len(sample_sizes),
        Sampler.ORTHO: np.argmax(sample_sizes >= 139**2),
        Sampler.SHADOW: len(sample_sizes),
    }
    REPEATS = 30
    z = 1.96
    ddof = 1

    # Load data from the iteration convergence experiment
    SHAPE_CONVERGENCE_RESULTS_ROOT = Path("data") / "shape_convergence"
    with (SHAPE_CONVERGENCE_RESULTS_ROOT / "metadata.json").open("r") as f:
        min_convergent_iters = json.load(f)["min_convergent_iters"]

    # Calculate per-sample-size area and CI for each sampling algorithm
    with tqdm(total=sum(MAX_SAMPLES_IDX[sampler] for sampler in Sampler)) as pbar:
        for sampler in Sampler:
            pbar.set_description(f"{sampler.title()} sampler")
            iter_sample_sizes = sample_sizes[: MAX_SAMPLES_IDX[sampler]]
            measured_areas = np.empty(
                (len(iter_sample_sizes), REPEATS), dtype=np.float64
            )
            expected_areas = np.empty_like(iter_sample_sizes, dtype=np.float64)
            cis = np.empty_like(iter_sample_sizes, dtype=np.float64)
            for i, sample_size in enumerate(iter_sample_sizes):
                area = est_area(
                    sample_size,
                    min_convergent_iters,
                    repeats=REPEATS,
                    sampler=sampler,
                    quiet=True,
                )
                expected_area, ci = mean_and_ci(area, z=z, ddof=ddof)
                expected_areas[i] = expected_area
                measured_areas[i] = area
                cis[i] = ci
                pbar.update()

            # Save per-sampler results
            np.save(RESULTS_ROOT / f"{sampler}_measured_area.npy", measured_areas)
            np.save(RESULTS_ROOT / f"{sampler}_expected_area.npy", expected_areas)
            np.save(RESULTS_ROOT / f"{sampler}_ci.npy", cis * np.sqrt(REPEATS))
            np.save(RESULTS_ROOT / f"{sampler}_sample_size.npy", iter_sample_sizes)

    # Write out experiment metadata
    metadata = {
        "max_samples": int(sample_sizes[-1]),
        "repeats": REPEATS,
        "z": z,
        "ddof": ddof,
        "iters": min_convergent_iters,
    }
    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
