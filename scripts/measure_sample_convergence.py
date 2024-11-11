import json
from pathlib import Path

if __name__ == "__main__":
    # Load data from the iteration convergence experiment
    RESULTS_ROOT = Path("data") / "shape_convergence"
    with (RESULTS_ROOT / "metadata.json").open("r") as f:
        metadata = json.load(f)

    # Determine the minimum iteration number we recorded as "convergent"
    min_convergent_iters = metadata["min_convergent_iters"]

    # Calculate per-sample-size area and CI for each sampling algorithm

    # Save data
    ...
