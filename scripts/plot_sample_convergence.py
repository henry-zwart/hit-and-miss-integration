import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

    print(target_area)
    fig, ax = plt.subplots()
    ax.axhline(target_area, color="grey")

    ax.axhline(target_area * (1 + ABSOLUTE_THRESHOLD), linestyle="dashed", color="grey")
    ax.axhline(target_area * (1 - ABSOLUTE_THRESHOLD), linestyle="dashed", color="grey")

    # For each sampling method, plot the lower and upper confidence interval, at each sample size
    colours = {
        "random": "blue",
        "lhs": "orange",
        "ortho": "green",
    }
    for sampler in ("random", "lhs"):
        area = np.load(RESULTS_ROOT / f"{sampler}_area.npy")[2000:]
        ci = np.load(RESULTS_ROOT / f"{sampler}_ci.npy")[2000:]
        sample_size = np.load(RESULTS_ROOT / f"{sampler}_sample_size.npy")[2000:]
        lower = area - ci
        upper = area + ci
        # ax.plot(
        #     sample_size,
        #     lower,
        #     label=sampler,
        #     color=colours[sampler],
        #     linewidth=1,
        # )
        # ax.plot(
        #     sample_size,
        #     upper,
        #     color=colours[sampler],
        #     linewidth=1,
        # )
        ax.fill_between(
            sample_size, lower, upper, color=colours[sampler], alpha=0.5, label=sampler
        )
        ax.plot(
            sample_size, area, color=colours[sampler], linestyle="dashed", linewidth=1
        )

    fig.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "area.png", dpi=700)
