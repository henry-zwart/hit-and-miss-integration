import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_true_area_convergence():
    # Load data:
    #   - Evaluated iteration numbers
    #   - Relative change confidence intervals
    #   - Estimated area confidence intervals
    #   - Relative convergence threshold
    RESULTS_ROOT = Path("results") / "data" / "true_area_convergence"
    FIGURES_ROOT = Path("results") / "figures"

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    iterations = np.load(RESULTS_ROOT / "iterations.npy")
    relchange_cis = np.load(RESULTS_ROOT / "relchange_cis.npy")
    area_cis = np.load(RESULTS_ROOT / "area_cis.npy")

    convergence_threshold = metadata["convergence_threshold"]
    min_convergent_idx = np.argmax(relchange_cis[:, 1] < convergence_threshold)
    min_convergent_iters = iterations[min_convergent_idx]
    min_convergent_area = area_cis[min_convergent_idx].mean()

    fig, axes = plt.subplots(2, sharex=True)

    # Scatterplot of the relative change in estimated area, with confidence intervals
    axes[0].scatter(iterations, 100 * relchange_cis.mean(axis=1), s=10, marker="x")
    axes[0].vlines(iterations, 100 * relchange_cis[:, 0], 100 * relchange_cis[:, 1])

    # Horizontal line at threshold, vertical red line at first i which achieves convergence
    axes[0].axhline(y=100 * convergence_threshold, color="grey", linewidth=0.5)
    axes[0].axvline(
        min_convergent_iters, linestyle="dashed", color="red", linewidth=0.5
    )

    # Make plot pretty
    axes[0].set_xlim(1, None)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative change")
    axes[0].set_title("Percentage relative change from A(i-1) -> A(i)")

    # Scatterplot of the estimated area for each of the tested i's, with confidence intervals
    axes[1].scatter(iterations, area_cis.mean(axis=1), s=10)
    axes[1].vlines(iterations, area_cis[:, 0], area_cis[:, 1])

    # Horizontal red line at minimum convergent area
    axes[1].axhline(min_convergent_area, linestyle="dashed", color="red", linewidth=0.5)
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
    axes[1].set_ylabel("Estimated area")
    axes[1].set_ylim(0, None)
    axes[1].set_xlabel("Iterations")
    axes[1].set_title("Convergence of A(i)")

    # Prepare and save figure
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "true_area_conv.png", dpi=500, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.scatter(
        iterations[min_convergent_idx:],
        area_cis[min_convergent_idx:].mean(axis=1),
        s=10,
    )
    ax.vlines(
        iterations[min_convergent_idx:],
        area_cis[min_convergent_idx:, 0],
        area_cis[min_convergent_idx:, 1],
    )
    ax.axhline(min_convergent_area, linestyle="dashed", color="red", linewidth=0.5)
    ax.text(
        x=1.01,
        y=min_convergent_area,
        s=f"{min_convergent_area:.4f}",
        va="center",
        ha="left",
        color="red",
        transform=ax.get_yaxis_transform(),
    )
    final_area = area_cis.mean(axis=1)[-1]
    ax.text(
        x=1.01,
        y=final_area,
        s=f"{final_area:.4f}",
        va="center",
        ha="left",
        color="red",
        transform=ax.get_yaxis_transform(),
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_ROOT / "true_area_conv_closeup.png",
        dpi=500,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    plot_true_area_convergence()
