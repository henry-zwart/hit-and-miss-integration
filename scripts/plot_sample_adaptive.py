# Plot sample adaptive

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    RESULTS_ROOT = Path("data") / "sample_adaptive"
    FIGURES_ROOT = Path("figures") / "sample_convergence"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    # with (Path("data") / "shape_convergence" / "metadata.json").open("r") as f:
    #   target_area = json.load(f)["min_convergent_area"]

    target_area = 1.506

    area = np.load(RESULTS_ROOT / "area.npy")
    total_samples = np.load(RESULTS_ROOT / "total_samples.npy")
    exp_area = np.load(RESULTS_ROOT / "expected_area.npy")
    ci = np.load(RESULTS_ROOT / "confidence_intervals.npy")

    average_samples = np.mean(total_samples, axis=0)

    lower = exp_area - ci
    upper = exp_area + ci

    fig, axes = plt.subplots()

    print(f"Lower: {lower}, upper: {upper}")
    axes.plot(average_samples, exp_area)
    print(total_samples.shape)
    for i in range(len(total_samples[0, :])):
        axes.scatter(total_samples[:, i], area[:, i])
    axes.fill_between(average_samples, lower, upper, alpha=0.5)
    plt.show()
    print(area.shape)
