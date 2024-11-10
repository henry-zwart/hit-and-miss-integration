import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    RESULTS_ROOT = Path("data") / "joint_convergence"
    FIGURES_ROOT = Path("figures") / "limit_convergence"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    expected_area = np.load(RESULTS_ROOT / "expected_area.npy")
    confidence_interval = np.load(RESULTS_ROOT / "confidence_intervals.npy")

    MIN_ITERATIONS = 20
    MIN_SAMPLES = 10_000

    fig, ax = plt.subplots(2)
    area_i_lower = expected_area[..., -1] - confidence_interval[..., -1]
    area_i_upper = expected_area[..., -1] + confidence_interval[..., -1]
    ax[0].plot(np.arange(metadata["max_iterations"] + 1), expected_area[..., -1])
    ax[0].fill_between(
        np.arange(MIN_ITERATIONS, metadata["max_iterations"] + 1),
        area_i_lower[MIN_ITERATIONS:],
        area_i_upper[MIN_ITERATIONS:],
        alpha=0.25,
    )
    area_s_lower = expected_area[-1, ...] - confidence_interval[-1, ...]
    area_s_upper = expected_area[-1, ...] + confidence_interval[-1, ...]
    ax[1].plot(
        np.arange(MIN_SAMPLES, metadata["max_samples"]), expected_area[-1, MIN_SAMPLES:]
    )
    ax[1].fill_between(
        np.arange(MIN_SAMPLES, metadata["max_samples"]),
        area_s_lower[MIN_SAMPLES:],
        area_s_upper[MIN_SAMPLES:],
        alpha=0.25,
    )
    fig.savefig(FIGURES_ROOT / "limit_area.png", dpi=500, bbox_inches="tight")

    fig, ax = plt.subplots(2)

    # Plot error due to finite iterations, with "infinite" samples
    ε_i = expected_area[..., -1] - expected_area[-1, -1]
    ε_i_lower = ε_i - confidence_interval[..., -1]
    ε_i_upper = ε_i + confidence_interval[..., -1]
    ax[0].plot(
        np.arange(MIN_ITERATIONS, metadata["max_iterations"] + 1),
        ε_i[MIN_ITERATIONS:],
    )
    ax[0].fill_between(
        np.arange(MIN_ITERATIONS, metadata["max_iterations"] + 1),
        ε_i_lower[MIN_ITERATIONS:],
        ε_i_upper[MIN_ITERATIONS:],
        alpha=0.25,
    )

    # Plot error due to finite samples, with "infinite" iterations
    ε_s = expected_area[-1, ...] - expected_area[-1, -1]
    ε_s_lower = ε_s - confidence_interval[-1]
    ε_s_upper = ε_s + confidence_interval[-1]
    ax[1].plot(np.arange(MIN_SAMPLES, metadata["max_samples"]), ε_s[MIN_SAMPLES:])
    ax[1].fill_between(
        np.arange(MIN_SAMPLES, metadata["max_samples"]),
        ε_s_lower[MIN_SAMPLES:],
        ε_s_upper[MIN_SAMPLES:],
        alpha=0.25,
    )

    fig.savefig(FIGURES_ROOT / "limit_error.png", dpi=500, bbox_inches="tight")
