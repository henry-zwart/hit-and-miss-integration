import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    random.seed(42)

    # Set parameters for plotting
    MIN_ITERATIONS = 2
    MIN_SAMPLES = 10_000

    # Load experiment data and prepare figure directory for plots
    RESULTS_ROOT = Path("data") / "joint_convergence"
    FIGURES_ROOT = Path("figures") / "limit_convergence"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    expected_area = np.load(RESULTS_ROOT / "expected_area.npy")
    confidence_interval = np.load(RESULTS_ROOT / "confidence_intervals.npy")

    # Plot of area estimates for max samples and varying iters, and vice versa.
    fig, ax = plt.subplots(2)

    # Make plot pretty
    ax[0].set_ylabel("Area of Mandelbrot set")
    ax[0].set_xlabel("Number of iterations")
    ax[1].set_ylabel("Area of Mandelbrot set")
    ax[1].set_xlabel("Number of sample points")
    plt.subplots_adjust(hspace=0.3)

    area_i_lower = expected_area[..., -1] - confidence_interval[..., -1]
    area_i_upper = expected_area[..., -1] + confidence_interval[..., -1]

    ax[0].plot(
        metadata["iterations"][MIN_ITERATIONS:],
        expected_area[MIN_ITERATIONS:, -1],
    )
    ax[0].fill_between(
        metadata["iterations"][MIN_ITERATIONS:],
        area_i_lower[MIN_ITERATIONS:],
        area_i_upper[MIN_ITERATIONS:],
        alpha=0.25,
    )
    area_s_lower = expected_area[-1, ...] - confidence_interval[-1, ...]
    area_s_upper = expected_area[-1, ...] + confidence_interval[-1, ...]
    ax[1].plot(
        np.arange(MIN_SAMPLES, metadata["max_samples"]),
        expected_area[-1, MIN_SAMPLES:],
    )
    ax[1].fill_between(
        np.arange(MIN_SAMPLES, metadata["max_samples"]),
        area_s_lower[MIN_SAMPLES:],
        area_s_upper[MIN_SAMPLES:],
        alpha=0.25,
    )
    fig.savefig(FIGURES_ROOT / "limit_area.png", dpi=500, bbox_inches="tight")

    fig, ax = plt.subplots(2)

    # Make plot pretty
    ax[0].set_ylabel("Error")
    ax[0].set_xlabel("Number of Iterations")
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Number of sample points")
    plt.subplots_adjust(hspace=0.3)

    # Plot error due to finite iterations, with "infinite" samples
    ε_i = expected_area[..., -1] - expected_area[-1, -1]
    ε_i_lower = ε_i - confidence_interval[..., -1]
    ε_i_upper = ε_i + confidence_interval[..., -1]
    ax[0].plot(
        metadata["iterations"][MIN_ITERATIONS:],
        ε_i[MIN_ITERATIONS:],
    )
    ax[0].fill_between(
        metadata["iterations"][MIN_ITERATIONS:],
        ε_i_lower[MIN_ITERATIONS:],
        ε_i_upper[MIN_ITERATIONS:],
        alpha=0.25,
    )
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")

    # Plot error due to finite samples, with "infinite" iterations
    for iters in (1, 4, 16):
        ε_s = expected_area[iters, ...] - expected_area[iters, -1]
        ε_s_lower = ε_s - confidence_interval[iters]
        ε_s_upper = ε_s + confidence_interval[iters]
        ax[1].plot(np.arange(MIN_SAMPLES, metadata["max_samples"]), ε_s[MIN_SAMPLES:])
        ax[1].fill_between(
            np.arange(MIN_SAMPLES, metadata["max_samples"]),
            ε_s_lower[MIN_SAMPLES:],
            ε_s_upper[MIN_SAMPLES:],
            alpha=0.25,
        )

    fig.savefig(FIGURES_ROOT / "limit_error.png", dpi=500, bbox_inches="tight")
