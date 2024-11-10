import json
from pathlib import Path

import numpy as np

from hit_and_mandelbrot.mandelbrot import in_mandelbrot

if __name__ == "__main__":
    RESULTS_ROOT = Path("data") / "mandelbrot"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    STEPS = 5000
    X_MIN, X_MAX = (-2, 2)
    Y_MIN, Y_MAX = (-2, 2)
    ITERATIONS = [0, 1, 2, 4, 16, 256]

    x_points = np.linspace(X_MIN, X_MAX, STEPS)
    y_points = np.linspace(Y_MIN, Y_MAX, STEPS) * 1.0j
    c = (x_points + y_points[:, None]).flatten()

    hits = np.zeros((len(ITERATIONS), len(c)), dtype=bool)
    for i, iters in enumerate(ITERATIONS):
        hits[i] = in_mandelbrot(c, iters)

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
