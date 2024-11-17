import json
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

from hit_and_mandelbrot.mandelbrot import Sampler
from hit_and_mandelbrot.random_seed import load_rng


def save_fig(fig, filename, output_dir, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(output_dir / f"{filename}.png", bbox_inches="tight")


def get_white_to_blue_cmap():
    colors = plt.cm.Blues(np.linspace(0, 1, 256))
    colors[:128] = mcolors.to_rgba("white")  # Modify lower half to be white
    custom_blues = mcolors.LinearSegmentedColormap.from_list("CustomBlues", colors)
    return custom_blues


def plot_mandelbrot(fp: str, data: Path, figures: Path):
    # Load data
    with (data / "mandelbrot" / "metadata.json").open("r") as f:
        metadata = json.load(f)
        side_length = metadata["steps_per_dim"]

    hits = np.load(data / "mandelbrot" / "hits.npy")

    # Overlay each iteration of Mandelbrot some some transparency
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = get_white_to_blue_cmap()
    for iter_hits in hits:
        mask = np.reshape(iter_hits, (side_length, side_length))
        ax.imshow(
            mask,
            alpha=0.2,
            cmap=cmap,
            norm=mpl.colors.Normalize(vmin=0, vmax=1),
        )

    ax.set_axis_off()
    save_fig(fig, fp, output_dir=figures)


def plot_relative_change(fp: str, data: Path, figures: Path):
    # Load data
    data_path = data / "shape_convergence"
    with (data_path / "metadata.json").open("r") as f:
        metadata = json.load(f)
        threshold = metadata["convergence_threshold"] * 100
        converge_idx = metadata["min_convergent_idx"]
        converge_iters = metadata["min_convergent_iters"]
        converge_area = metadata["min_convergent_area"]

    iterations = np.load(data_path / "iterations.npy")
    relchange_cis = np.load(data_path / "relchange_cis.npy") * 100  # Scale to pct.
    area_cis = np.load(data_path / "area_cis.npy")

    # Make plot
    fig, axes = plt.subplots(2, sharex=True, figsize=(8.7, 6))

    # Error plot
    expected_relchange = relchange_cis.mean(axis=1)
    axes[0].scatter(iterations, expected_relchange, s=10, marker="x")
    axes[0].vlines(iterations, relchange_cis[:, 0], relchange_cis[:, 1])
    axes[0].axhline(y=threshold, color="red", linewidth=0.5, linestyle="dashed")
    axes[0].axvline(converge_iters, color="grey", linewidth=0.5)
    axes[0].set_xlim(0, None)  # To-do: Check. used to be 1,None
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative change")

    # Area plot
    upper_conv_area = area_cis[converge_idx, 1]
    axes[1].scatter(iterations, area_cis.mean(axis=1), s=10)
    axes[1].vlines(iterations, area_cis[:, 0], area_cis[:, 1])
    axes[1].axhline(upper_conv_area, linestyle="dashed", color="red", linewidth=0.5)
    axes[1].axvline(converge_iters, color="grey", linewidth=0.5)
    axes[1].text(
        x=1.01,
        y=converge_area,
        s=f"{converge_area:.4f}",
        va="center",
        ha="left",
        color="red",
        transform=axes[1].get_yaxis_transform(),
    )
    axes[1].set_ylabel("Estimated area")
    axes[1].set_ylim(1, None)
    axes[1].set_xlabel("Number of iterations")

    # Inset "zoom" on points after convergence
    zoom_ax = axes[1].inset_axes(
        [0.4, 0.37, 0.5, 0.5],
        xlim=(converge_iters - 50, iterations[-1] + 50),
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
        iterations[converge_idx:],
        area_cis[converge_idx:].mean(axis=1),
        s=10,
    )
    zoom_ax.vlines(
        iterations[converge_idx:],
        area_cis[converge_idx:, 0],
        area_cis[converge_idx:, 1],
    )
    zoom_ax.axhline(
        area_cis[converge_idx, 1], linestyle="dashed", color="red", linewidth=0.5
    )

    save_fig(fig, fp, figures)


def plot_convergence_error(
    fp: str,
    min_iters: int,
    min_samples: int,
    plot_iters: list[int],
    data: Path,
    figures: Path,
):
    # Load data
    data_path = data / "joint_convergence"
    with (data_path / "metadata.json").open("r") as f:
        metadata = json.load(f)
        iterations_step = metadata["iter_step_size"]
        iterations = metadata["iterations"][min_iters:]
        samples_step = metadata["sample_step_size"]
        max_samples = metadata["max_samples"]

    expect_error = np.load(data_path / "expected_err.npy")
    confidence = np.load(data_path / "err_confidence.npy")

    # Make plots
    fig, axes = plt.subplots(2, figsize=(8.7, 6))

    # Error due to finite iterations
    error_iter = expect_error[min_iters:, -1]
    error_iter_lower = error_iter - confidence[min_iters:, -1]
    error_iter_upper = error_iter + confidence[min_iters:, -1]
    axes[0].plot(iterations, error_iter)
    axes[0].fill_between(iterations, error_iter_lower, error_iter_upper, alpha=0.25)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Error")
    axes[0].set_xlabel("Number of iterations")

    # Error due to finite samples
    sample_sizes = np.arange(min_samples, max_samples, samples_step)
    min_samples_idx = min_samples // samples_step
    for iters in plot_iters:
        iter_idx = iters // iterations_step
        error_sample = expect_error[iter_idx, min_samples_idx:]
        error_sample_lower = error_sample - confidence[iter_idx, min_samples_idx:]
        error_sample_upper = error_sample + confidence[iter_idx, min_samples_idx:]
        axes[1].plot(sample_sizes, error_sample)
        axes[1].fill_between(
            sample_sizes, error_sample_lower, error_sample_upper, alpha=0.25
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Number of samples")

    save_fig(fig, fp, figures)


def plot_sampler_example(fp: str, data: Path, figures: Path):
    overlay_hits = np.load(data / "mandelbrot" / "hits.npy")[4]
    mask = np.reshape(overlay_hits, (int(np.sqrt(len(overlay_hits))), -1))
    cmap = get_white_to_blue_cmap()

    fig, axes = plt.subplots(
        1,
        len(Sampler),
        sharex=True,
        sharey=True,
        # figsize=(12, 5),
        figsize=(8, 4),
        subplot_kw=dict(box_aspect=1),
    )

    data_path = data / "sample_convergence"
    for i, sampler in enumerate(Sampler):
        samples = np.load(data_path / f"{sampler}_example_samples.npy")
        hits = np.load(data_path / f"{sampler}_example_hits.npy")

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
        axes[i].set_xticks([-2, 0, 2])

    axes[0].set_yticks([-2, 0, 2])
    save_fig(fig, fp, figures)


def plot_sampler_estimates(
    fp: str, threshold: float, min_samples: int, data: Path, figures: Path
):
    # Load data
    with (data / "shape_convergence/metadata.json").open("r") as f:
        target_area = json.load(f)["min_convergent_area"]

    with (data / "sample_convergence/metadata.json").open("r") as f:
        max_samples = json.load(f)["global_max_samples"]

    fig, axes = plt.subplots(len(Sampler) + 1, sharey=True, sharex=True, figsize=(6, 4))

    # Target area and absolute convergence bounds
    lower = target_area * (1 - threshold)
    upper = target_area * (1 + threshold)
    for ax in axes:
        ax.axhline(target_area, color="grey", linewidth=1)
        ax.axhline(
            lower,
            linestyle="dashed",
            color="grey",
            linewidth=0.5,
        )
        ax.axhline(
            upper,
            linestyle="dashed",
            color="grey",
            linewidth=0.5,
        )

    # Plot non-adaptive samplers
    colours = {"random": "blue", "lhs": "orange", "ortho": "green", "shadow": "red"}
    data_path = data / "sample_convergence"
    for ax, sampler in zip(axes, Sampler):
        area = np.load(data_path / f"{sampler}_expected_area.npy")[min_samples:]
        ci = np.load(data_path / f"{sampler}_ci.npy")[min_samples:]
        n_samples = np.load(data_path / f"{sampler}_sample_size.npy")[min_samples:]
        lower = area - ci
        upper = area + ci
        color = colours[sampler]

        # Mean
        ax.plot(n_samples, area, color=color, linestyle="dashed", linewidth=1)

        # Plus/minus 2 std
        ax.fill_between(n_samples, lower, upper, color=color, alpha=0.5)
        ax.set_ylabel(sampler.title(), rotation=0, fontsize=10, va="center", ha="left")
        ax.yaxis.set_label_position("right")

    # Plot adaptive sampler
    data_path = data / "sample_adaptive"
    area = np.load(data_path / "area.npy")
    total_samples = np.load(data_path / "total_samples.npy")
    exp_area = np.load(data_path / "expected_area.npy")
    ci = np.load(data_path / "confidence_intervals.npy")

    average_samples = np.mean(total_samples, axis=0)

    lower = exp_area - ci
    upper = exp_area + ci

    axes[4].plot(average_samples, exp_area, linestyle="dashed", linewidth=1)
    axes[4].fill_between(
        average_samples, lower, upper, alpha=0.5, color="yellow", label="Adaptive"
    )
    axes[4].set_ylabel("Adaptive", rotation=0, fontsize=10, va="center", ha="left")
    axes[4].yaxis.set_label_position("right")

    # Set axis limits consistently
    axes[0].set_ylim(1.4, 1.6)
    axes[0].set_xlim(0, max_samples + 1000)

    # Clean up axes
    for i, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i != 4:
            ax.tick_params(
                axis="x",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
            )

    save_fig(fig, fp, figures)


def final_true(mask, axis, invalid_val=-1):
    """Return indices of last True element in array.
    
    Adapted from: https://stackoverflow.com/questions/47269390/how-to-find-first-non-\
        zero-value-in-every-column-of-a-numpy-array/47269413#47269413
    """
    val = mask.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def plot_sampler_convergence(
    fp: str, threshold: float, data: Path, figures: Path, results_dir: Path
):
    # Load data
    with (data / "shape_convergence/metadata.json").open("r") as f:
        target_area = json.load(f)["min_convergent_area"]

    # with (data / "sample_convergence/metadata.json").open("r") as f:
    #     max_samples = json.load(f)["global_max_samples"]

    data_path = data / "sample_convergence"

    fig, ax = plt.subplots(figsize=(6, 3.5))
    results = []
    for sampler in Sampler:
        area = np.load(data_path / f"{sampler}_measured_area.npy")
        sample_size = np.load(data_path / f"{sampler}_sample_size.npy")

        # Shuffle results for each sample size along 'repeats' axis
        ind = np.ones_like(area, dtype=np.int64) * np.arange(area.shape[1])
        load_rng().permuted(ind, axis=1, out=ind)
        area = np.take_along_axis(area, ind, axis=1)

        # Final sample size with estimated area outside threshold
        t = target_area * threshold
        max_nonconvergent_idx = final_true(
            np.abs(area - target_area) >= t, axis=0, invalid_val=-1
        )

        # Translate to min convergent idx such that subsequent sample sizes in thresh.
        min_convergent_idx = max_nonconvergent_idx + 1

        # Get distribution of min convergence samples, treating non-convergent as NaN
        min_convergent_sample_size = np.zeros_like(min_convergent_idx, dtype=np.float64)
        mask = min_convergent_idx < area.shape[0]
        min_convergent_sample_size[mask] = sample_size[min_convergent_idx[mask]]
        min_convergent_sample_size[~mask] = np.nan

        results.append(min_convergent_sample_size)

    # Welch's t-test on minimum convergent sample sizes between samplers
    test_results = {str(s): {} for s in Sampler}
    for i, s1 in enumerate(Sampler):
        for j, s2 in enumerate(Sampler):
            if i == j:
                continue
            tr = ttest_ind(results[i], results[j], equal_var=False)
            test_results[s1][str(s2)] = {
                "statistic": tr.statistic,
                "p": tr.pvalue,
                "df": tr.df,
            }

    with (results_dir / "sampler_comparison.json").open("w") as f:
        json.dump(test_results, f)

    sns.boxplot(results, orient="h", ax=ax, whis=(0, 100), width=0.5)
    sns.swarmplot(data=results, orient="h", color="k", size=1, ax=ax)

    ax.set_xlabel("Convergent sample size")
    ax.set_yticks([0, 1, 2, 3], labels=[s.title() for s in Sampler])
    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.tick_right()
    ax.set_xlim(0, None)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    save_fig(fig, fp, figures)


if __name__ == "__main__":
    # Set random seeds
    load_rng()
    random.seed(42)

    # === Default matplotlib settings
    FONT_SIZE_SMALL = 10
    FONT_SIZE_DEFAULT = 12
    FONT_SIZE_LARGE = 14

    plt.rc("font", family="Georgia")
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("mathtext", fontset="stix")
    plt.rc("font", size=FONT_SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=FONT_SIZE_LARGE)  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT_SIZE_DEFAULT)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=FONT_SIZE_SMALL)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=FONT_SIZE_SMALL)  # fontsize of the tick labels

    # plt.rc("axes", titlepad=10)  # add space between title and plot
    plt.rc("figure", dpi=500)  # fix output resolution

    # Base paths
    DATA = Path("data")
    RESULTS = Path("results")
    FIGURES = RESULTS / "figures"

    # Deterministic overlayed mandelbrot image
    plot_mandelbrot("mandelbrot", DATA, FIGURES)

    # Estimating A_M
    plot_relative_change("relative_change", DATA, FIGURES)

    # Convergence error due to iterations vs. sample sizes
    min_iters = 2
    convergence_min_samples = 500
    plot_iters = [25, 250, 2500]
    plot_convergence_error(
        "convergence_error",
        min_iters=min_iters,
        min_samples=convergence_min_samples,
        plot_iters=plot_iters,
        data=DATA,
        figures=FIGURES,
    )

    # Comparing sampling methods
    abs_convergence_threshold = 1.5 / 100
    sampler_min_samples = 20
    plot_sampler_example(
        "sampler_examples",
        DATA,
        FIGURES,
    )
    plot_sampler_estimates(
        "sampler_estimates",
        abs_convergence_threshold,
        sampler_min_samples,
        DATA,
        FIGURES,
    )
    plot_sampler_convergence(
        "sampler_convergence",
        abs_convergence_threshold,
        DATA,
        FIGURES,
        RESULTS,
    )

    with (RESULTS / "plot_metadata.json").open("w") as f:
        plot_meta = {
            "convergence_error": {
                "min_iters": min_iters,
                "min_samples": convergence_min_samples,
                "plot_iters": plot_iters,
            },
            "samplers": {
                "min_samples": sampler_min_samples,
                "convergence_threshold": abs_convergence_threshold,
            },
        }
        json.dump(plot_meta, f)
