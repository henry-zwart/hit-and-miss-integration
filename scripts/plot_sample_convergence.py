import json
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hit_and_mandelbrot import Sampler
from hit_and_mandelbrot.random_seed import load_rng


def final_true(mask, axis, invalid_val=-1):
    """Return indices of last True element in array.
    
    Adapted from: https://stackoverflow.com/questions/47269390/how-to-find-first-non-\
        zero-value-in-every-column-of-a-numpy-array/47269413#47269413
    """
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def get_white_to_blue_cmap():
    colors = plt.cm.Blues(np.linspace(0, 1, 256))
    colors[:128] = mcolors.to_rgba("white")  # Modify lower half to be white
    custom_blues = mcolors.LinearSegmentedColormap.from_list("CustomBlues", colors)
    return custom_blues


def plot_sampler_examples(results_dir: Path, figures_dir: Path):
    overlay_hits = np.load(Path("data") / "mandelbrot" / "hits.npy")[4]
    mask = np.reshape(overlay_hits, (int(np.sqrt(len(overlay_hits))), -1))
    cmap = get_white_to_blue_cmap()

    fig, axes = plt.subplots(
        1,
        len(Sampler),
        sharex=True,
        sharey=True,
        figsize=(12, 5),
        subplot_kw=dict(box_aspect=1),
    )

    for i, sampler in enumerate(Sampler):
        samples = np.load(results_dir / f"{sampler}_example_samples.npy")
        hits = np.load(results_dir / f"{sampler}_example_hits.npy")

        samples_x = samples.real
        samples_y = samples.imag
        axes[i].scatter(samples_x[hits], samples_y[hits], s=5, color="green")
        axes[i].scatter(samples_x[~hits], samples_y[~hits], s=5, color="red")
        axes[i].set_title(sampler.title())
        axes[i].imshow(
            mask,
            alpha=0.35,
            cmap=cmap,
            norm=mpl.colors.Normalize(vmin=-0, vmax=1),
            extent=[-2, 2, -2, 2],
        )

    fig.savefig(figures_dir / "sampler_examples.png", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    # Set random seeds
    load_rng()
    random.seed(42)

    # Parameters
    MIN_SAMPLES = 20
    ABS_THRESHOLD = 1.5 / 100

    # Load experiment data and prepare figure directory for plots
    RESULTS_ROOT = Path("data") / "sample_convergence"
    FIGURES_ROOT = Path("figures") / "sample_convergence"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    with (Path("data") / "shape_convergence" / "metadata.json").open("r") as f:
        target_area = json.load(f)["min_convergent_area"]

    plot_sampler_examples(RESULTS_ROOT, FIGURES_ROOT)

    # Show convergence behaviour of different samplers.
    #   - Plot two sample std. for each sampler, for varying sample sizes.
    #   - Show the target (~A_M) area and 1.5% error thresholds for comparison.
    fig, axes = plt.subplots(len(Sampler) + 1, sharey=True, sharex=True)

    # True area
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

    # 2 sample standard deviations
    colours = {"random": "blue", "lhs": "orange", "ortho": "green", "shadow": "red"}
    for i, sampler in enumerate(Sampler):
        area = np.load(RESULTS_ROOT / f"{sampler}_expected_area.npy")[MIN_SAMPLES:]
        ci = np.load(RESULTS_ROOT / f"{sampler}_ci.npy")[MIN_SAMPLES:]
        sample_size = np.load(RESULTS_ROOT / f"{sampler}_sample_size.npy")[MIN_SAMPLES:]

        lower = area - ci
        upper = area + ci
        axes[i].fill_between(
            sample_size, lower, upper, color=colours[sampler], alpha=0.5
        )
        axes[i].plot(
            sample_size, area, color=colours[sampler], linestyle="dashed", linewidth=1
        )
        axes[i].set_title(sampler.title())

    RESULTS_ROOT_2 = Path("data") / "sample_adaptive"
    area = np.load(RESULTS_ROOT_2 / "area.npy")
    total_samples = np.load(RESULTS_ROOT_2 / "total_samples.npy")
    exp_area = np.load(RESULTS_ROOT_2 / "expected_area.npy")
    ci = np.load(RESULTS_ROOT_2 / "confidence_intervals.npy")

    average_samples = np.mean(total_samples, axis=0)

    lower = exp_area - ci
    upper = exp_area + ci

    axes[4].plot(average_samples, exp_area, linestyle="dashed", linewidth=1)
    axes[4].fill_between(
        average_samples, lower, upper, alpha=0.5, color="yellow", label="Stratified"
    )

    # fig.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "area.png", dpi=700)

    # Plot distribution over samples-till-convergence.
    # i.e. for each repeat, on each sampling method, record the number of samples such
    #       that for all subsequent sample sizes, the estimate is totally within the
    #       1.5% error bounds.
    # Plot this as a box plot, with convergent sample sizes overlaid.

    fig, ax = plt.subplots()
    results = []
    for i, sampler in enumerate(Sampler):
        area = np.load(RESULTS_ROOT / f"{sampler}_measured_area.npy")
        sample_size = np.load(RESULTS_ROOT / f"{sampler}_sample_size.npy")

        # Shuffle results for each sample size along 'repeats' axis
        ind = np.ones_like(area, dtype=np.int64) * np.arange(area.shape[1])
        load_rng().permuted(ind, axis=1, out=ind)
        area = np.take_along_axis(area, ind, axis=1)

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

    sns.boxplot(results, ax=ax)
    sns.swarmplot(data=results, color="k", size=1.5, ax=ax)

    ax.set_xlabel("Sampler")
    ax.set_ylabel("Convergent sample size")
    fig.legend(Sampler)
    fig.tight_layout()
    fig.savefig(
        FIGURES_ROOT / "convergent_samplesize_dist.png", bbox_inches="tight", dpi=500
    )
