import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hit_and_mandelbrot.random_seed import load_rng


def plot_shape_convergence():
    # Load experiment data and prepare figures directory for plots
    RESULTS_ROOT = Path("data") / "shape_convergence"
    FIGURES_ROOT = Path("figures") / "shape_convergence"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    iterations = np.load(RESULTS_ROOT / "iterations.npy")
    relchange_cis = np.load(RESULTS_ROOT / "relchange_cis.npy")
    area_cis = np.load(RESULTS_ROOT / "area_cis.npy")

    convergence_threshold = metadata["convergence_threshold"]
    min_convergent_idx = metadata["min_convergent_idx"]
    min_convergent_area = metadata["min_convergent_area"]
    min_convergent_iters = metadata["min_convergent_iters"]

    # Two stacked plots:
    #   1. Relative change in estimate as iterations increase
    #   2. Estimated area for varying iterations
    #       2b. (Inset): Zoom in on the area estimates after convergence to show CI's
    fig, axes = plt.subplots(2, sharex=True)

    # Scatterplot of the relative change in estimated area, with confidence intervals
    axes[0].scatter(iterations, 100 * relchange_cis.mean(axis=1), s=10, marker="x")
    axes[0].vlines(iterations, 100 * relchange_cis[:, 0], 100 * relchange_cis[:, 1])

    # Add:
    #   - Horizontal red line at threshold,
    #   - Vertical line at first iteration which achieves convergence
    axes[0].axhline(
        y=100 * convergence_threshold, color="red", linewidth=0.5, linestyle="dashed"
    )
    axes[0].axvline(min_convergent_iters, color="grey", linewidth=0.5)

    # Make plot pretty
    axes[0].set_xlim(1, None)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative change")

    # Scatterplot of the estimated area for each of the tested i's
    axes[1].scatter(iterations, area_cis.mean(axis=1), s=10)
    axes[1].vlines(iterations, area_cis[:, 0], area_cis[:, 1])

    # Add horizontal red line at minimum convergent area
    axes[1].axhline(
        area_cis[min_convergent_idx, 1], linestyle="dashed", color="red", linewidth=0.5
    )
    axes[1].axvline(min_convergent_iters, color="grey", linewidth=0.5)

    # Write the minimum convergent area to the right of the plot
    axes[1].text(
        x=1.01,
        y=min_convergent_area,
        s=f"{min_convergent_area:.4f}",
        va="center",
        ha="left",
        color="red",
        transform=axes[1].get_yaxis_transform(),
    )

    # Make plot pretty
    axes[1].set_ylabel("Estimated area of Mandelbrot set")
    axes[1].set_ylim(1, None)
    axes[1].set_xlabel("Number of iterations")

    # Add zoomed-in section to show area after convergence
    zoom_ax = axes[1].inset_axes(
        [0.4, 0.37, 0.5, 0.5],
        xlim=(min_convergent_iters - 50, iterations[-1] + 50),
        ylim=(1.505, 1.510),
        xticks=[],
    )

    for spine in zoom_ax.spines.values():
        spine.set_linewidth(0.5)

    rect, leders = axes[1].indicate_inset_zoom(zoom_ax, edgecolor="black")
    rect.set_edgecolor("none")
    for leder in leders:
        leder.set_linewidth(0.5)

    zoom_ax.scatter(
        iterations[min_convergent_idx:],
        area_cis[min_convergent_idx:].mean(axis=1),
        s=10,
    )
    zoom_ax.vlines(
        iterations[min_convergent_idx:],
        area_cis[min_convergent_idx:, 0],
        area_cis[min_convergent_idx:, 1],
    )
    zoom_ax.axhline(
        area_cis[min_convergent_idx, 1], linestyle="dashed", color="red", linewidth=0.5
    )

    # Prepare and save figure
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "relchange_and_area.png", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    # Set random seeds
    load_rng()
    random.seed(42)

    plot_shape_convergence()
