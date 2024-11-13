import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hit_and_mandelbrot.sampling import Sampler


def final_true(mask, axis, invalid_val=-1):
    """Return indices of last True element in array.
    
    Adapted from: https://stackoverflow.com/questions/47269390/how-to-find-first-non-\
        zero-value-in-every-column-of-a-numpy-array/47269413#47269413
    """
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


if __name__ == "__main__":
    MIN_SAMPLES = 20
    ABS_THRESHOLD = 1.5 / 100

    RESULTS_ROOT = Path("data") / "sample_convergence"
    FIGURES_ROOT = Path("figures") / "sample_convergence"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    with (Path("data") / "shape_convergence" / "metadata.json").open("r") as f:
        target_area = json.load(f)["min_convergent_area"]

    fig, axes = plt.subplots(3, sharey=True, sharex=True)

    for ax in axes:
        ax.axhline(target_area, color="grey", linewidth=1)
        ax.axhline(
            target_area * (1 + ABS_THRESHOLD),
            linestyle="dashed",
            color="grey",
            linewidth=1,
        )
        ax.axhline(
            target_area * (1 - ABS_THRESHOLD),
            linestyle="dashed",
            color="grey",
            linewidth=1,
        )

    # For each sampling method, plot the lower and upper confidence interval, at each sample size
    colours = {
        "random": "blue",
        "lhs": "orange",
        "ortho": "green",
    }
    for i, sampler in enumerate(Sampler):
        area = np.load(RESULTS_ROOT / f"{sampler}_expected_area.npy")[MIN_SAMPLES:]

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
    fig, ax = plt.subplots()
    results = []
    for i, sampler in enumerate(Sampler):
        area = np.load(RESULTS_ROOT / f"{sampler}_measured_area.npy")
        sample_size = np.load(RESULTS_ROOT / f"{sampler}_sample_size.npy")

        # Final sample size with estimated area outside threshold
        max_nonconvergent_idx = final_true(
            np.abs(area - target_area) >= target_area * ABS_THRESHOLD,
            axis=0,
            invalid_val=-1,
        )

        # Translate to min convergent idx such that subsequent sample sizes in thresh.
        min_convergent_idx = max_nonconvergent_idx + 1

        # Get distribution of min convergence samples, treating non-convergent as NaN
        min_convergent_sample_size = np.zeros_like(min_convergent_idx, dtype=np.float64)
        mask = min_convergent_idx < area.shape[0]
        min_convergent_sample_size[mask] = sample_size[min_convergent_idx[mask]]
        min_convergent_sample_size[~mask] = np.nan

        results.append(min_convergent_sample_size)

        # sns.histplot(min_convergent_sample_size, stat="density", ax=ax, alpha=0.5)
        # sns.kdeplot(min_convergent_sample_size, fill=True, ax=ax, alpha=0.5)

    # sns.histplot(
    #     results, stat="density", binwidth=2000, common_bins=False, ax=ax, alpha=0.3
    # )
    sns.boxplot(results, ax=ax)
    sns.swarmplot(data=results, color="k", size=3, ax=ax)

    # ax.set_xlim(0, metadata["max_samples"])
    ax.set_xlabel("Minimum convergent sample size")
    ax.set_ylabel("Probability density")
    fig.legend(Sampler)
    fig.tight_layout()
    fig.savefig(
        FIGURES_ROOT / "convergent_samplesize_dist.png", bbox_inches="tight", dpi=500
    )
