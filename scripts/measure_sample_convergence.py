"""
    Course: Stochastic Simulation
    Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
    Student IDs: 15719227, 15393879, 13392425 
    Assignement: Hit and Miss Integration

    File description:
        Script to run the code that measures the sample convergence.
"""

import json
import random
from pathlib import Path

import numpy as np
from sympy import sieve
from tqdm import tqdm

from hit_and_mandelbrot.hits import calculate_sample_hits
from hit_and_mandelbrot.mandelbrot import Sampler, est_area, sample_complex_uniform
from hit_and_mandelbrot.random_seed import load_rng
from hit_and_mandelbrot.statistics import mean_and_ci


def create_sampler_example_data(n_samples: int, iterations: int, result_dir: Path):
    """
    We can measure sample convergence for random + shadow sampling via bootstrapping.
    Record hits for largest sample size. To measure the sample convergence with k samples,
    select k rows at random. Repeat for N_REPEATS.
    """
    for sampler in Sampler:
        samples = sample_complex_uniform(
            n_samples,
            repeats=1,
            method=sampler,
            quiet=True,
        )
        hits = calculate_sample_hits(samples.c, iterations=iterations)
        np.save(result_dir / f"{sampler}_example_samples.npy", samples.c[0])
        np.save(result_dir / f"{sampler}_example_hits.npy", hits[0])


if __name__ == "__main__":
    # Random seeds
    load_rng()
    random.seed(42)

    # Establish directory to write results
    RESULTS_ROOT = Path("data") / "sample_convergence"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Parameters
    sieve.extend(499)
    primes = np.array(sieve._list)
    sample_sizes = primes[1:] ** 2  # Exclude p=2, incompatible with shadow sampler i=4
    MAX_SAMPLES_IDX = {
        Sampler.RANDOM: len(sample_sizes),
        Sampler.LHS: np.argmax(sample_sizes >= 431**2),
        Sampler.ORTHO: np.argmax(sample_sizes >= 139**2),
        Sampler.SHADOW: np.argmax(sample_sizes >= 257**2),
    }
    EXAMPLE_ITERS = 16
    EXAMPLE_SAMPLE_SIZE = 7**2
    REPEATS = 50
    SHADOW_ITERS = 4
    Z = 1.96
    DDOF = 1

    create_sampler_example_data(EXAMPLE_SAMPLE_SIZE, EXAMPLE_ITERS, RESULTS_ROOT)

    # Load data from the iteration convergence experiment
    SHAPE_CONVERGENCE_RESULTS_ROOT = Path("data") / "shape_convergence"
    with (SHAPE_CONVERGENCE_RESULTS_ROOT / "metadata.json").open("r") as f:
        min_convergent_iters = json.load(f)["min_convergent_iters"]

    # Calculate per-sample-size area and CI for each sampling algorithm
    with tqdm(total=sum(MAX_SAMPLES_IDX[sampler] for sampler in Sampler)) as pbar:
        for sampler in Sampler:
            pbar.set_description(f"{sampler.title()} sampler")
            iter_sample_sizes = sample_sizes[: MAX_SAMPLES_IDX[sampler]]
            if sampler in (Sampler.RANDOM, Sampler.SHADOW):
                measured_areas = est_area(
                    iter_sample_sizes.max(),
                    min_convergent_iters,
                    repeats=REPEATS,
                    sampler=sampler,
                    quiet=True,
                    per_sample=True,
                    shadow_iters=SHADOW_ITERS,
                )[:, iter_sample_sizes - 1]

                expected_areas, cis = mean_and_ci(measured_areas, z=Z, ddof=DDOF)

                # Plotting code expects (sample_sizes, repeats)
                measured_areas = measured_areas.T
                pbar.update(len(iter_sample_sizes))

            else:
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
                    expected_area, ci = mean_and_ci(area, z=Z, ddof=DDOF)
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
        "max_samples": {
            sampler: int(sample_sizes[MAX_SAMPLES_IDX[sampler] - 1])
            for sampler in Sampler
        },
        "global_max_samples": int(sample_sizes[-1]),
        "example_iters": EXAMPLE_ITERS,
        "example_sample_size": EXAMPLE_SAMPLE_SIZE,
        "shadow_iters": SHADOW_ITERS,
        "repeats": REPEATS,
        "z": Z,
        "ddof": DDOF,
        "iters": min_convergent_iters,
    }
    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
