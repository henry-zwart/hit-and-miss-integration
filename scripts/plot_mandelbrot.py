import json
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def get_white_to_blue_cmap():
    colors = plt.cm.Blues(np.linspace(0, 1, 256))
    colors[:128] = mcolors.to_rgba("white")  # Modify lower half to be white
    custom_blues = mcolors.LinearSegmentedColormap.from_list("CustomBlues", colors)
    return custom_blues


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    random.seed(42)

    # Load data, prepare figure directory for plotting
    RESULTS_ROOT = Path("data") / "mandelbrot"
    FIGURES_ROOT = Path("figures") / "mandelbrot"
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    side_length = metadata["steps_per_dim"]
    hits = np.load(RESULTS_ROOT / "hits.npy")
    iterations = metadata["iterations"]

    # Overlay each iteration of Mandelbrot some some transparency
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = get_white_to_blue_cmap()
    for i, step in enumerate(iterations):
        mask = np.reshape(hits[i], (side_length, side_length))
        ax.imshow(
            mask,
            alpha=0.2,
            cmap=cmap,
            norm=mpl.colors.Normalize(vmin=0, vmax=1),
        )

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "mandelbrot.png", dpi=800, bbox_inches="tight")
