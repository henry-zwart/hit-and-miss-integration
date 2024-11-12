import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hit_and_mandelbrot.sampling import Sampler

if __name__ == "__main__":
    MIN_SAMPLES = 10
    ABSOLUTE_THRESHOLD = 1 / 100

    RESULTS_ROOT = Path("data") / "sample_convergence"
    FIGURES_ROOT = Path("figures") / "sample_convergence"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    with (Path("data") / "shape_convergence" / "metadata.json").open("r") as f:
        target_area = json.load(f)["min_convergent_area"]

    fig, axes = plt.subplots(3, sharey=True)

    for ax in axes:
        ax.axhline(target_area, color="grey")
        ax.axhline(
            target_area * (1 + ABSOLUTE_THRESHOLD), linestyle="dashed", color="grey"
        )
        ax.axhline(
            target_area * (1 - ABSOLUTE_THRESHOLD), linestyle="dashed", color="grey"
        )

    # For each sampling method, plot the lower and upper confidence interval, at each sample size
    colours = {
        "random": "blue",
        "lhs": "orange",
        "ortho": "green",
    }
    for i, sampler in enumerate(Sampler):
        area = np.load(RESULTS_ROOT / f"{sampler}_area.npy")[MIN_SAMPLES:]
        ci = np.load(RESULTS_ROOT / f"{sampler}_ci.npy")[MIN_SAMPLES:]
        sample_size = np.load(RESULTS_ROOT / f"{sampler}_sample_size.npy")[MIN_SAMPLES:]
        lower = area - ci
        upper = area + ci
        axes[i].fill_between(
            sample_size, lower, upper, color=colours[sampler], alpha=0.5, label=sampler
        )
        axes[i].plot(
            sample_size, area, color=colours[sampler], linestyle="dashed", linewidth=1
        )

    fig.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "area.png", dpi=700)

    # Plot distribution over samples-till-convergence.
    # i.e. for each repeat, on each sampling method, record the number of sampling such
    #       that the confidence intervals afterward are totally within the bounds.
    #       Plot this as a histogram or KDE.
