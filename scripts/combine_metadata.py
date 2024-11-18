"""
    Course: Stochastic Simulation
    Names: Petr Chalupský, Henry Zwart, Tika van Bennekum
    Student IDs: 15719227, 15393879, 13392425 
    Assignement: Hit and Miss Integration

    File description:
        Script to combine metadata.
"""
import json
from pathlib import Path


def main(data_dir: Path, results_dir: Path):
    big_meta = {}

    with (data_dir / "mandelbrot/metadata.json").open("r") as f:
        big_meta["mandelbrot"] = json.load(f)

    with (data_dir / "shape_convergence/metadata.json").open("r") as f:
        big_meta["relative_change"] = json.load(f)

    with (data_dir / "joint_convergence/metadata.json").open("r") as f:
        big_meta["convergence_error"] = json.load(f)

    with (data_dir / "sample_convergence/metadata.json").open("r") as f:
        big_meta["samplers"] = json.load(f)

    with (data_dir / "sample_adaptive/metadata.json").open("r") as f:
        big_meta["adaptive"] = json.load(f)

    with (results_dir / "experiment_metadata.json").open("w") as f:
        json.dump(big_meta, f)


if __name__ == "__main__":
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    main(DATA_DIR, RESULTS_DIR)
