import json
import random
from pathlib import Path

import numpy as np

from hit_and_mandelbrot.hits import calculate_sample_hits

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    random.seed(42)

    # Establish directory to write results
    RESULTS_ROOT = Path("data") / "mandelbrot"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Parameters
    STEPS = 5000
    X_MIN, X_MAX = (-2, 2)
    Y_MIN, Y_MAX = (-2, 2)
    ITERATIONS = [0, 1, 2, 4, 16, 256]

    # Evenly spaced points on the (-2,2) by (-2i, 2i) rectangle
    x_points = np.linspace(X_MIN, X_MAX, STEPS)
    y_points = np.linspace(Y_MIN, Y_MAX, STEPS) * 1.0j
    c = (x_points + y_points[:, None]).flatten()

    # Determine which points are inside Mandelbrot
    hits = np.zeros((len(ITERATIONS), len(c)), dtype=bool)
    for i, iters in enumerate(ITERATIONS):
        hits[i] = calculate_sample_hits(c[None, :], iters)[0]

    # Write results and metadata
    np.save(RESULTS_ROOT / "hits.npy", hits)
    metadata = {
        "x_min": X_MIN,
        "x_max": X_MAX,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "steps_per_dim": STEPS,
        "iterations": ITERATIONS,
    }
    with (RESULTS_ROOT / "metadata.json").open("w") as f:
        json.dump(metadata, f)
