import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    MIN_ITERATIONS = 20
    MIN_SAMPLES = 10_000

    RESULTS_ROOT = Path("data") / "joint_convergence"
    FIGURES_ROOT = Path("figures") / "joint_error"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    expected_area = np.load(RESULTS_ROOT / "expected_area.npy")
    confidence_interval = np.load(RESULTS_ROOT / "confidence_intervals.npy")

    # Plot area for equivalent errors
    equiv_errors = np.load(RESULTS_ROOT / "equiv_errors_measured.npy")
    equiv_error_areas = np.load(RESULTS_ROOT / "equiv_error_areas.npy")

    fig, ax = plt.subplots()
    ax.scatter(equiv_errors, equiv_error_areas)
    ax.set_xlabel("Error")
    ax.set_ylabel("Area estimate")
    ax.set_title(r"Mandelbrot area estimate for $\epsilon_i = \epsilon_s$")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "equiv_error_areas.png", dpi=500, bbox_inches="tight")

    # Plot heatmap / contour for error given sample size and iters
    fig, ax = plt.subplots(2)
    iteration_numbers = np.arange(metadata["max_iterations"] + 1)
    sample_sizes = np.arange(0, metadata["max_samples"], 100)
    Samples, Iters = np.meshgrid(sample_sizes, iteration_numbers)
    Error = expected_area[:, ::100] - 1.506
    contour = ax[0].contour(
        Iters, Samples, Error, levels=np.arange(25, dtype=np.float64) / 100
    )
    ax[0].clabel(contour, fontsize=10)
    ax[1].pcolor(Iters, Samples, Error, vmin=0, vmax=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "error_contour.png", dpi=500, bbox_inches="tight")

    # Plot a histogram of the flattened error matrix (all sample sizes and iterations)
    fig, ax = plt.subplots()
    ax.hist(Error.flatten(), bins=40)
    ax.set_title("Histogram over error matrix")
    fig.savefig(FIGURES_ROOT / "error_hist.png")
